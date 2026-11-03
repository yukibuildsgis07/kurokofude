"""MCP サーバーのツール関数を直接呼ぶ単体テスト。OCR エンジンは呼ばない。"""

import json
import shutil
from pathlib import Path

import anyio
import numpy as np
import pytest
import tifffile

from kurokofude import mcp_server, pipeline

ROOT = Path(__file__).resolve().parents[1]


def rect(cx, cy, w, h):
    return [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]


BOXES = [
    {"text": "12-3", "score": 0.97, "quad": rect(300, 300, 60, 20)},
    {"text": "", "score": 0.95, "quad": rect(300, 326, 90, 18)},  # 地番の下の氏名（文字列は OCR 直後に捨てられている）
    {"text": "", "score": 0.90, "quad": rect(600, 200, 90, 18)},  # ペアにならない文字
    {"text": "273:19", "score": 0.80, "quad": rect(150, 500, 60, 20)},  # 数字主体
]


@pytest.fixture
def project(tmp_path, monkeypatch):
    """ルートの構成をまねた一時フォルダ。configs/ は本物をコピーする。"""
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    tifffile.imwrite(raw / "zumen.tif", np.full((700, 1000, 3), 240, dtype=np.uint8), photometric="rgb")
    (raw / "zumen.tfw").write_text("0.5\n0\n0\n-0.5\n100\n200\n", encoding="ascii")
    (raw / "note.txt").write_text("x", encoding="utf-8")
    (tmp_path / "data" / "other").mkdir()
    tifffile.imwrite(tmp_path / "data" / "other" / "b.tif", np.zeros((10, 10, 3), dtype=np.uint8), photometric="rgb")
    monkeypatch.setattr(pipeline, "run_ocr", lambda *_args: [dict(b) for b in BOXES])
    return tmp_path


def test_exactly_three_tools_are_exposed():
    tools = anyio.run(mcp_server.server.list_tools)
    assert sorted(t.name for t in tools) == ["list_masked", "mask_drawing", "open_review"]


def test_mask_drawing_returns_only_safe_data(project):
    result = mcp_server.do_mask_drawing("data/raw/zumen.tif", root=project)
    assert result["masked_path"] == "data/masked/zumen.tif"
    assert result["result_path"] == "data/masked/zumen.chiban.json"
    assert result["world_file"] == "data/masked/zumen.tfw"
    assert (project / "data" / "masked" / "zumen.tif").is_file()
    assert [c["text"] for c in result["chiban"]] == ["12-3"]
    assert set(result["chiban"][0]) == {"text", "center", "score", "status", "masked_below"}
    assert result["chiban"][0]["status"] == "detected"
    assert result["chiban"][0]["masked_below"] is True
    assert [u["text"] for u in result["uncertain"]] == ["273:19"]
    assert result["masks"] == {"total": 2, "by_status": {"auto": 1, "unpaired": 1}}
    assert result["review_url"] is None
    dumped = json.dumps(result, ensure_ascii=False)
    assert "raw" not in dumped  # 塗る前の図面のパスを返さない
    assert "quad" not in dumped  # 四角形の頂点や画像のバイト列も返さない


def test_mask_drawing_accepts_absolute_path_and_preset(project):
    result = mcp_server.do_mask_drawing(str(project / "data" / "raw" / "zumen.tif"), "bottom_strip", root=project)
    assert result["masked_path"] == "data/masked/zumen.tif"


@pytest.mark.parametrize(
    "path",
    [
        "data/other/b.tif",  # 保護ディレクトリの外
        "data/raw/../other/b.tif",  # .. で外に出る
        "data/masked/zumen.tif",  # 塗った側
        "data/raw/note.txt",  # TIFF でない
        "data/raw/nothing.tif",  # 存在しない
        "data/raw",  # フォルダ
    ],
)
def test_mask_drawing_rejects_anything_but_protected_tiff(project, path):
    with pytest.raises(mcp_server.GateError) as caught:
        mcp_server.do_mask_drawing(path, root=project)
    assert path not in str(caught.value)
    assert "zumen" not in str(caught.value) and "note" not in str(caught.value)


def test_mask_drawing_follows_symlink_to_its_real_place(project):
    link = project / "data" / "raw" / "link.tif"
    link.symlink_to(project / "data" / "other" / "b.tif")
    with pytest.raises(mcp_server.GateError):
        mcp_server.do_mask_drawing("data/raw/link.tif", root=project)


@pytest.mark.parametrize("preset", ["../pyproject", "/etc/passwd", "nothing", "configs/../../x"])
def test_config_accepts_only_presets_in_configs(project, preset):
    with pytest.raises(mcp_server.GateError):
        mcp_server.do_mask_drawing("data/raw/zumen.tif", preset, root=project)


def test_mask_failure_hides_details(project, monkeypatch):
    def broken(*_args):
        raise RuntimeError(f"cannot open {project / 'data' / 'raw' / 'zumen.tif'}")

    monkeypatch.setattr(mcp_server, "mask_file", broken)
    with pytest.raises(mcp_server.GateError) as caught:
        mcp_server.do_mask_drawing("data/raw/zumen.tif", root=project)
    assert "RuntimeError" in str(caught.value)
    assert "raw" not in str(caught.value)


def test_list_masked_and_open_review(project):
    assert mcp_server.do_list_masked(root=project)["drawings"] == []
    mcp_server.do_mask_drawing("data/raw/zumen.tif", root=project)
    listed = mcp_server.do_list_masked(root=project)["drawings"]
    assert listed == [
        {
            "masked_path": "data/masked/zumen.tif",
            "has_world_file": True,
            "chiban": 1,
            "masks": {"total": 2, "by_status": {"auto": 1, "unpaired": 1}},
            "reviewed": False,
        }
    ]
    review = mcp_server.do_open_review("data/masked/zumen.tif", root=project)
    assert review["review_url"] is None and review["reviewed"] is False

    result_path = project / "data" / "masked" / "zumen.chiban.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["masks"][0]["status"] = "confirmed"
    result_path.write_text(json.dumps(result), encoding="utf-8")
    assert mcp_server.do_open_review("data/masked/zumen.tif", root=project)["reviewed"] is True


@pytest.mark.parametrize("path", ["data/raw/zumen.tif", "data/other/b.tif", "data/masked/none.tif"])
def test_open_review_accepts_only_masked_drawings(project, path):
    with pytest.raises(mcp_server.GateError):
        mcp_server.do_open_review(path, root=project)


def test_mcp_json_starts_server_with_project_venv():
    config = json.loads((ROOT / ".mcp.json").read_text(encoding="utf-8"))
    server = config["mcpServers"]["kurokofude"]
    assert server["command"] == "uv"
    assert server["args"] == ["run", "--no-sync", "python", "-m", "kurokofude.mcp_server"]
