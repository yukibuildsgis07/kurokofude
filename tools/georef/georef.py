"""④ 地番の照合でワールドファイルを作る（設計書 §5、§18.3）。生の図面は使わない。

入力は塗った図面の地番一覧（data/masked/<図面>.chiban.json）と公図（data/ref/<図面>.geojson）だけ。
図郭線（画像上の四角形）があれば、反復の2回目以降に使う。

python tools/georef/georef.py data/masked/<図面>.chiban.json data/ref/<図面>.geojson [--frame <json>]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import Point, Polygon, shape

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
HYPHEN = "-"
TFW_DECIMALS = 10
REPORT_DECIMALS = 3
PERCENT = 100.0
CLIP_NONE = "none"
CLIP_IMAGE = "image"
CLIP_FRAME = "frame"
IRLS_NONE = "none"
MAD_TO_SIGMA = 1.4826  # 正規分布のとき、中央絶対偏差を標準偏差に直す係数


@dataclass
class Pair:
    text: str  # 公図で引き当てた地番
    read_as: str  # OCR が読んだ地番
    pixel: tuple[float, float]  # 画素座標（列, 行）
    world: tuple[float, float]  # 筆の重心（東, 北）
    score: float
    reread: bool  # ハイフンを挿して読み替えたか
    by_prediction: bool = False  # 重複した地番を、予測した位置で選んだか


@dataclass
class Fit:
    params: np.ndarray  # a, b, c, d（東 = a·列 + b·行 + c、北 = b·列 − a·行 + d）
    inliers: list[Pair]
    residuals_m: np.ndarray
    outliers: list[Pair] = field(default_factory=list)
    weights: np.ndarray | None = None  # IRLS の最後の重み（inliers と同じ順）


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def parcel_shapes(geojson: dict) -> dict[str, list]:
    """地番 → 筆のポリゴンの一覧。同じ地番の筆が複数あることもある（§18.3）。"""
    shapes: dict[str, list] = defaultdict(list)
    for feature in geojson["features"]:
        shapes[feature["properties"]["chiban"]].append(shape(feature["geometry"]).buffer(0))
    return dict(shapes)


def visible_centroid(polygon, clip) -> tuple[float, float] | None:
    """筆の見えている部分（clip の内側）の重心。clip がなければ筆全体の重心。"""
    visible = polygon.intersection(clip) if clip is not None else polygon
    if visible.is_empty:
        return None
    point = visible.centroid
    return point.x, point.y


def hyphen_candidates(text: str) -> list[str]:
    """ハイフンが落ちた地番の候補（2739 → 2-739、27-39、273-9）。"""
    if HYPHEN in text or len(text) < 2:
        return []
    return [f"{text[:i]}{HYPHEN}{text[i:]}" for i in range(1, len(text))]


def design_rows(pixel: np.ndarray, flip_rows: bool) -> tuple[np.ndarray, np.ndarray]:
    """相似変換の最小二乗の係数行列。flip_rows=False は鏡映を許した変換（向きの確認用）。"""
    col, row = pixel[:, 0], pixel[:, 1]
    ones, zeros = np.ones_like(col), np.zeros_like(col)
    if flip_rows:  # 東 = a·列 + b·行 + c、北 = b·列 − a·行 + d
        return np.column_stack([col, row, ones, zeros]), np.column_stack([-row, col, zeros, ones])
    # 行を反転しない（鏡映）：東 = a·列 − b·行 + c、北 = b·列 + a·行 + d
    return np.column_stack([col, -row, ones, zeros]), np.column_stack([row, col, zeros, ones])


def solve(pixel: np.ndarray, world: np.ndarray, flip_rows: bool = True, weights: np.ndarray | None = None) -> np.ndarray:
    """（重み付き）最小二乗。weights はペアごとの重みで、東・北の両方の式に同じ重みを掛ける。"""
    east, north = design_rows(pixel, flip_rows)
    matrix = np.vstack([east, north])
    target = np.concatenate([world[:, 0], world[:, 1]])
    if weights is not None:
        root = np.sqrt(np.concatenate([weights, weights]))
        matrix, target = matrix * root[:, None], target * root
    params, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return params


def robust_weights(residuals: np.ndarray, irls_cfg: dict) -> np.ndarray:
    """残差が大きいペアほど重みを下げる（Huber）。尺度は中央絶対偏差（§5、Q14）。"""
    scale = max(MAD_TO_SIGMA * float(np.median(np.abs(residuals))), irls_cfg["min_scale_m"])
    ratio = np.abs(residuals) / scale
    k = irls_cfg["k"]
    return np.where(ratio <= k, 1.0, k / np.maximum(ratio, k))


def irls(pixel: np.ndarray, world: np.ndarray, irls_cfg: dict, flip_rows: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """重み付き最小二乗を、重みが落ち着くまで繰り返す。返すのは変換と最後の重み。"""
    weights = np.ones(len(pixel))
    params = solve(pixel, world, flip_rows)
    if irls_cfg["method"] == IRLS_NONE:
        return params, weights
    for _ in range(irls_cfg["iterations"]):
        residuals = np.linalg.norm(apply(params, pixel, flip_rows) - world, axis=1)
        new_weights = robust_weights(residuals, irls_cfg)
        params = solve(pixel, world, flip_rows, new_weights)
        if np.max(np.abs(new_weights - weights)) < irls_cfg["tolerance"]:
            weights = new_weights
            break
        weights = new_weights
    return params, weights


def apply(params: np.ndarray, pixel: np.ndarray, flip_rows: bool = True) -> np.ndarray:
    east, north = design_rows(pixel, flip_rows)
    return np.column_stack([east @ params, north @ params])


def scale_of(params: np.ndarray) -> float:
    return float(np.hypot(params[0], params[1]))


def footprint(params: np.ndarray, quad) -> Polygon:
    """画像上の四角形を、解いた変換で公共座標に写す。"""
    return Polygon(apply(params, np.asarray(quad, dtype=float)))


def image_quad(image_size: list[int]) -> list[list[float]]:
    width, height = image_size
    return [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]


def make_pairs(
    chiban: list[dict], shapes: dict[str, list], config: dict, params: np.ndarray | None = None, clip=None
) -> tuple[list[Pair], list[dict]]:
    """地番の文字列で筆を引き当てる（§5、§18.3）。

    公図にない地番は、ハイフンを挿した候補で読み替える。地番が図面か公図で重複していれば、
    params（前の解）がないときは使わず、あるときは予測した位置に最も近い筆を採る。
    """
    pairs, unmatched = [], []
    drawn = Counter(item["text"] for item in chiban)
    for item in chiban:
        text = item["text"]
        reread = text not in shapes
        candidates = [c for c in hyphen_candidates(text) if c in shapes] if reread else [text]
        if not candidates:
            unmatched.append({"text": text, "reason": "公図にない"})
            continue
        options = [(c, polygon) for c in candidates for polygon in shapes[c]]
        ambiguous = len(options) > 1 or drawn[text] > 1
        if ambiguous and params is None:
            unmatched.append({"text": text, "reason": "重複（1回目の解では使わない）"})
            continue
        located = [(c, visible_centroid(polygon, clip)) for c, polygon in options]
        located = [(c, xy) for c, xy in located if xy is not None]
        if not located:
            unmatched.append({"text": text, "reason": "筆が図面の範囲の外"})
            continue
        if ambiguous:
            predicted = apply(params, np.array([item["center"]], dtype=float))[0]
            matched, world = min(located, key=lambda cw: np.hypot(*(np.array(cw[1]) - predicted)))
        else:
            matched, world = located[0]
        score = item["score"] * (config["reread_score_factor"] if reread else 1.0)
        pairs.append(Pair(matched, text, tuple(item["center"]), world, score, reread, ambiguous))
    return pairs, unmatched


def fit_pairs(pairs: list[Pair], config: dict, flip_rows: bool = True) -> Fit | None:
    """2組から変換を作って一致する組を数え（RANSAC 風）、一致した組で最小二乗をやり直す。"""
    if len(pairs) < config["min_pairs"]:
        return None
    pixel = np.array([p.pixel for p in pairs], dtype=float)
    world = np.array([p.world for p in pairs], dtype=float)
    candidates = list(itertools.combinations(range(len(pairs)), 2))
    rng = random.Random(config["seed"])
    if len(candidates) > config["max_pair_candidates"]:
        candidates = rng.sample(candidates, config["max_pair_candidates"])
    tolerance = config["ransac_tolerance_m"]
    best: np.ndarray | None = None
    best_count = -1
    for i, j in candidates:
        params = solve(pixel[[i, j]], world[[i, j]], flip_rows)
        errors = np.linalg.norm(apply(params, pixel, flip_rows) - world, axis=1)
        count = int((errors <= tolerance).sum())
        if count > best_count:
            best, best_count = errors <= tolerance, count
    if best is None or best_count < config["min_pairs"]:
        return None
    for _ in range(len(pairs)):  # 一致した組で解き直し、一致の範囲が変わらなくなるまで繰り返す
        params = solve(pixel[best], world[best], flip_rows)
        errors = np.linalg.norm(apply(params, pixel, flip_rows) - world, axis=1)
        updated = errors <= tolerance
        if updated.sum() < config["min_pairs"] or (updated == best).all():
            break
        best = updated
    # RANSAC 風の除外で残った組の内側で、残差で重みを決める最小二乗を繰り返す（Q14）
    params, weights = irls(pixel[best], world[best], config["irls"], flip_rows)
    errors = np.linalg.norm(apply(params, pixel, flip_rows) - world, axis=1)
    inliers = [p for p, ok in zip(pairs, best) if ok]
    outliers = [p for p, ok in zip(pairs, best) if not ok]
    return Fit(params, inliers, errors[best], outliers, weights)


def world_file(params: np.ndarray) -> list[float]:
    """.tfw の並び（A, D, B, E, C, F）。東 = A·列 + B·行 + C、北 = D·列 + E·行 + F。"""
    a, b, c, d = params
    return [a, b, b, -a, c, d]


def iterate(chiban_json: dict, shapes: dict, config: dict, frame_quad=None) -> tuple[Fit | None, list[Pair], list[dict], list[dict]]:
    """1回目は筆全体の重心（重複なし）で解く。次は画像の範囲で切った重心、その次から図郭線の範囲で
    切った重心で解き直す。縮尺の変化が小さくなったら止める（§18.3）。"""
    chiban = chiban_json["chiban"]
    pairs, unmatched = make_pairs(chiban, shapes, config)
    fit = fit_pairs(pairs, config)
    history = []
    if fit is None:
        return None, pairs, unmatched, history
    history.append(_history(CLIP_NONE, fit, len(pairs)))
    for step in range(config["max_iterations"]):
        use_frame = frame_quad is not None and step > 0
        clip = footprint(fit.params, frame_quad if use_frame else image_quad(chiban_json["image_size"]))
        new_pairs, new_unmatched = make_pairs(chiban, shapes, config, fit.params, clip)
        refit = fit_pairs(new_pairs, config)
        if refit is None:
            break
        change_pct = abs(scale_of(refit.params) / scale_of(fit.params) - 1) * PERCENT
        fit, pairs, unmatched = refit, new_pairs, new_unmatched
        history.append(_history(CLIP_FRAME if use_frame else CLIP_IMAGE, fit, len(pairs)))
        if step > 0 and change_pct < config["converge_scale_pct"]:
            break
    return fit, pairs, unmatched, history


def _history(clip: str, fit: Fit, found: int) -> dict:
    return {
        "clip": clip,
        "pixel_size_m": round(scale_of(fit.params), 7),
        "pairs_used": len(fit.inliers),
        "pairs_found": found,
        "residual_rms_m": round(float(np.sqrt(np.mean(fit.residuals_m**2))), REPORT_DECIMALS),
    }


def georeference(chiban_json: dict, geojson: dict, config: dict, frame_quad=None) -> dict:
    shapes = parcel_shapes(geojson)
    fit, pairs, unmatched, history = iterate(chiban_json, shapes, config, frame_quad)
    mirrored = fit_pairs(pairs, config, flip_rows=False)
    report: dict = {
        "pairs_found": len(pairs),
        "unmatched": unmatched,
        "rereadings": [{"read_as": p.read_as, "matched": p.text} for p in pairs if p.reread],
        "chosen_by_prediction": [p.text for p in pairs if p.by_prediction],
        "iterations": history,
        "used_frame": frame_quad is not None,
    }
    if fit is None:
        report["status"] = "failed"
        report["message"] = f"一致するペアが {config['min_pairs']} 組に足りません"
        return report
    a, b = fit.params[0], fit.params[1]
    report.update(
        {
            "status": "ok",
            "world_file": world_file(fit.params),
            "pairs_used": len(fit.inliers),
            "pixel_size_m": round(scale_of(fit.params), 7),
            "rotation_deg": round(float(np.degrees(np.arctan2(b, a))), REPORT_DECIMALS),
            "residual_rms_m": round(float(np.sqrt(np.mean(fit.residuals_m**2))), REPORT_DECIMALS),
            "residual_max_m": round(float(fit.residuals_m.max()), REPORT_DECIMALS),
            "residuals": [
                {
                    "chiban": p.text,
                    "read_as": p.read_as,
                    "residual_m": round(float(r), REPORT_DECIMALS),
                    "weight": round(float(w), REPORT_DECIMALS),
                    "score": round(p.score, 3),
                }
                for p, r, w in sorted(
                    zip(fit.inliers, fit.residuals_m, fit.weights if fit.weights is not None else np.ones(len(fit.inliers))),
                    key=lambda x: -x[1],
                )
            ],
            "irls": config["irls"]["method"],
            "excluded": [p.text for p in fit.outliers],
            # 向きの確認：鏡映を許した変換のほうが合うなら、行の向きを取り違えている
            "mirrored_residual_rms_m": (
                round(float(np.sqrt(np.mean(mirrored.residuals_m**2))), REPORT_DECIMALS) if mirrored else None
            ),
            "mirrored_pairs_used": len(mirrored.inliers) if mirrored else 0,
        }
    )
    return report


def write_world_file(path: Path, values: list[float]) -> None:
    path.write_text("\n".join(f"{v:.{TFW_DECIMALS}f}" for v in values) + "\n", encoding="ascii")


def read_frame(path: Path | None) -> list | None:
    """図郭線（画像上の四角形）。ダミーでは .truth.json の frame_quad、または frame_quad だけの JSON。"""
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))["frame_quad"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="地番の照合でワールドファイルを作る")
    parser.add_argument("chiban_json", type=Path, help="data/masked/<図面>.chiban.json")
    parser.add_argument("geojson", type=Path, help="data/ref/<図面>.geojson")
    parser.add_argument("--frame", type=Path, help="図郭線（frame_quad を持つ JSON）")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no-write", action="store_true", help=".tfw を書かずに結果だけ出す")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    chiban_json = json.loads(args.chiban_json.read_text(encoding="utf-8"))
    geojson = json.loads(args.geojson.read_text(encoding="utf-8"))
    # 図郭線は、指定があればそれを、なければ mask_drawing の結果（chiban.json の frame_quad）を使う
    frame = read_frame(args.frame) if args.frame else chiban_json.get("frame_quad")
    report = georeference(chiban_json, geojson, config, frame)
    stem = args.chiban_json.name.removesuffix(".chiban.json")
    if report["status"] == "ok" and not args.no_write:
        tfw = args.chiban_json.with_name(f"{stem}.tfw")
        write_world_file(tfw, report["world_file"])
        report["tfw"] = tfw.as_posix()
    report_path = args.chiban_json.with_name(f"{stem}.georef.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    summary = {k: v for k, v in report.items() if k not in {"residuals", "world_file"}}
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
