"""黒子筆の MCP サーバー（設計書 §4.2、§16）。stdio で動く。

Claude に渡すのは、塗った図面のパスと地番の一覧だけ。生画像のバイト列、塗る前の図面のパス、
氏名の文字列はどのツールも返さない。
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from .config import DEFAULT_CONFIG_PATH, load_config
from .pipeline import mask_file
from .review import app as review

PROJECT_ROOT = DEFAULT_CONFIG_PATH.parents[1]
CONFIGS_DIR_NAME = "configs"
MASKED_DIR = Path("data") / "masked"
CACHE_DIR = Path("data") / "cache"
TIFF_SUFFIXES = {".tif", ".tiff"}
CONFIG_SUFFIX = ".yaml"
RESULT_SUFFIX = ".chiban.json"
WORLD_FILE_SUFFIX = ".tfw"
STATUS_DETECTED = "detected"
CENTER_DECIMALS = 1
SCORE_DECIMALS = 3
REVIEW_HINT = "open_review で確認画面（人が見る画面）をブラウザで開けます。塗り漏れと塗りすぎを人が確かめてから使ってください"

server = MCPServer(
    name="kurokofude",
    instructions=(
        "黒子筆：測量図面の地番の下にある氏名だけを手元で塗りつぶす。"
        "塗る前の図面（data/raw/）は直接読まず、mask_drawing に渡すこと。"
        "返ってくるのは塗った図面のパスと地番の一覧だけ。"
    ),
)


class GateError(ValueError):
    """関所で止めた。メッセージに入力のパスを含めない。"""


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _resolve(path: str, root: Path) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve()


def _protected_dirs(config: dict, root: Path) -> list[Path]:
    return [(root / Path(d)).resolve() for d in config["protected_dirs"]]


def _check_raw_drawing(path: str, config: dict, root: Path) -> Path:
    """塗る前の図面が保護ディレクトリの下にある TIFF であることを確かめる（フックとの二重の守り）。"""
    resolved = _resolve(path, root)
    if not any(resolved.is_relative_to(d) for d in _protected_dirs(config, root)):
        raise GateError("保護ディレクトリ（protected_dirs）の下にある図面だけを受け付けます")
    if resolved.suffix.lower() not in TIFF_SUFFIXES:
        raise GateError("TIFF（.tif / .tiff）の図面だけを受け付けます")
    if not resolved.is_file():
        raise GateError("指定された図面が見つかりません")
    return resolved


def _check_masked_drawing(path: str, root: Path) -> Path:
    resolved = _resolve(path, root)
    if not resolved.is_relative_to((root / MASKED_DIR).resolve()):
        raise GateError("data/masked/ の下にある塗った図面だけを受け付けます")
    if resolved.suffix.lower() not in TIFF_SUFFIXES or not resolved.is_file():
        raise GateError("塗った図面（TIFF）が見つかりません")
    return resolved


def _load_preset(name: str | None, root: Path) -> dict:
    """設定は configs/ の下のプリセット名だけ受け付ける。任意のパスは読まない。"""
    if not name:
        return load_config()
    stem = Path(name).stem if Path(name).suffix == CONFIG_SUFFIX else name
    configs_dir = (root / CONFIGS_DIR_NAME).resolve()
    preset = (configs_dir / f"{stem}{CONFIG_SUFFIX}").resolve()
    if preset.parent != configs_dir or not preset.is_file():
        available = sorted(p.stem for p in configs_dir.glob(f"*{CONFIG_SUFFIX}"))
        raise GateError(f"設定は configs/ のプリセット名で指定してください（{', '.join(available)}）")
    return load_config(preset)


def _read_result(masked_tif: Path) -> dict | None:
    result_path = masked_tif.with_name(f"{masked_tif.stem}{RESULT_SUFFIX}")
    if not result_path.is_file():
        return None
    return json.loads(result_path.read_text(encoding="utf-8"))


def _mask_counts(result: dict) -> dict:
    by_status = Counter(mask["status"] for mask in result["masks"])
    return {"total": len(result["masks"]), "by_status": dict(sorted(by_status.items()))}


def _is_reviewed(result: dict) -> bool:
    return bool(result.get("confirmed"))


def _summarize(result: dict) -> dict:
    """chiban.json から、Claude に渡してよいものだけを取り出す。四角形の頂点は渡さない。"""
    return {
        "chiban": [
            {
                "text": item["text"],
                "center": [round(v, CENTER_DECIMALS) for v in item["center"]],
                "score": round(item["score"], SCORE_DECIMALS),
                "status": item.get("status", STATUS_DETECTED),
                "masked_below": item["masked_below"],
            }
            for item in result["chiban"]
        ],
        "uncertain": [
            {
                "text": item["text"],  # 数字と □ だけ（§11.1）
                "center": [round(sum(p[i] for p in item["quad"]) / len(item["quad"]), CENTER_DECIMALS) for i in (0, 1)],
            }
            for item in result.get("uncertain", [])
        ],
        "title_meta": [
            {
                "text": item["text"],  # 数字・記号・図名の定型語と □ だけ（§18.1）
                "center": [round(sum(p[i] for p in item["quad"]) / len(item["quad"]), CENTER_DECIMALS) for i in (0, 1)],
            }
            for item in result.get("title_meta", [])
        ],
        "masks": _mask_counts(result),
        "frame_quad": result.get("frame_quad"),  # 図郭線（画素座標4点）。なければ null
    }


def do_mask_drawing(path: str, config: str | None = None, root: Path = PROJECT_ROOT) -> dict:
    root = root.resolve()
    settings = _load_preset(config, root)
    raw = _check_raw_drawing(path, settings, root)
    try:
        summary = mask_file(raw, settings, root / MASKED_DIR, root / CACHE_DIR)
    except Exception as exc:  # 例外のメッセージにはパスが入りうるので、種類だけ返す
        raise GateError(f"塗りに失敗しました（{type(exc).__name__}）") from None
    masked = summary.out_path.resolve()
    result = _read_result(masked)
    world_file = masked.with_suffix(WORLD_FILE_SUFFIX)
    return {
        "masked_path": _relative(masked, root),
        "result_path": _relative(masked.with_name(f"{masked.stem}{RESULT_SUFFIX}"), root),
        "world_file": _relative(world_file, root) if world_file.is_file() else None,
        "image_size": result["image_size"],
        **_summarize(result),
        "review": REVIEW_HINT,
    }


def do_open_review(masked_path: str, root: Path = PROJECT_ROOT) -> dict:
    """確認画面は人の道具。URL は返さず、ブラウザを直接開く（§3.1）。"""
    root = root.resolve()
    masked = _check_masked_drawing(masked_path, root)
    result = _read_result(masked)
    if result is None:
        raise GateError("塗った図面の結果（chiban.json）が見つかりません")
    config = load_config()
    port = config["review"]["port"]
    try:
        review.ensure_running(root, port, config)
    except RuntimeError as exc:
        raise GateError(f"確認画面を起動できませんでした（{exc}）") from None
    return {"opened": review.open_in_browser(port, masked.stem), "confirmed": bool(result.get("confirmed"))}


def do_list_masked(root: Path = PROJECT_ROOT) -> dict:
    root = root.resolve()
    masked_dir = root / MASKED_DIR
    drawings = []
    for tif in sorted(p for p in masked_dir.glob("*") if p.suffix.lower() in TIFF_SUFFIXES):
        result = _read_result(tif)
        drawings.append(
            {
                "masked_path": _relative(tif.resolve(), root),
                "has_world_file": tif.with_suffix(WORLD_FILE_SUFFIX).is_file(),
                "chiban": len(result["chiban"]) if result else None,
                "masks": _mask_counts(result) if result else None,
                "confirmed": bool(result and _is_reviewed(result)),
            }
        )
    return {"masked_dir": MASKED_DIR.as_posix(), "drawings": drawings}


@server.tool()
def mask_drawing(path: str, config: str | None = None) -> dict:
    """塗る前の図面（data/raw/ の下の TIFF）の氏名を塗り、塗った図面のパスと地番の一覧を返す。

    path: 塗る前の図面のパス（リポジトリのルートからの相対パス、または絶対パス）
    config: configs/ のプリセット名（例 bottom_strip）。省略すると既定の設定
    """
    return do_mask_drawing(path, config)


@server.tool()
def open_review(masked_path: str) -> dict:
    """塗った図面（data/masked/ の下）の確認画面を、人が見るためにブラウザで開く。URL は返さない。

    返すもの：opened（ブラウザを開けたか）、confirmed（人が確定済みか）
    """
    return do_open_review(masked_path)


@server.tool()
def list_masked() -> dict:
    """data/masked/ にある塗った図面の一覧と、それぞれの状態を返す。"""
    return do_list_masked()


def main() -> None:
    server.run("stdio")


if __name__ == "__main__":
    main()
