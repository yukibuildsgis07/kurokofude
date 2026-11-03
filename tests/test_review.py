"""② 確認画面の API を TestClient で確かめる。画面の JS はテストしない。OCR は呼ばない。"""

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import tifffile
from fastapi.testclient import TestClient

from kurokofude import mcp_server, pipeline
from kurokofude.config import load_config
from kurokofude.review import app as review

ROOT = Path(__file__).resolve().parents[1]
GRAY = 240
BLACK = [0, 0, 0]


def rect(cx, cy, w, h):
    return [[cx - w / 2, cy - h / 2], [cx + w / 2, cy - h / 2], [cx + w / 2, cy + h / 2], [cx - w / 2, cy + h / 2]]


BOXES = [
    {"text": "12-3", "score": 0.97, "quad": rect(300, 300, 60, 20)},
    {"text": "", "score": 0.95, "quad": rect(300, 326, 90, 18)},
    {"text": "", "score": 0.90, "quad": rect(600, 200, 90, 18)},
    {"text": "273:19", "score": 0.80, "quad": rect(150, 500, 60, 20)},
]


@pytest.fixture
def project(tmp_path, monkeypatch):
    shutil.copytree(ROOT / "configs", tmp_path / "configs")
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    image = np.full((700, 1000, 3), GRAY, dtype=np.uint8)
    tifffile.imwrite(raw / "zumen.tif", image, photometric="rgb")
    (raw / "zumen.tfw").write_text("0.5\n0\n0\n-0.5\n100\n200\n", encoding="ascii")
    monkeypatch.setattr(pipeline, "run_ocr", lambda *_args: [dict(b) for b in BOXES])
    mcp_server.do_mask_drawing("data/raw/zumen.tif", root=tmp_path)
    return tmp_path


@pytest.fixture
def client(project):
    return TestClient(review.create_app(project, load_config()))


def result_of(project):
    return json.loads((project / "data" / "masked" / "zumen.chiban.json").read_text(encoding="utf-8"))


def masked_pixels(project):
    return tifffile.imread(project / "data" / "masked" / "zumen.tif")


def test_list_and_get(client):
    assert client.get("/api/drawings").json() == {"drawings": [{"name": "zumen", "confirmed": False, "masks": 2}]}
    data = client.get("/api/drawings/zumen").json()
    assert {m["status"] for m in data["masks"]} == {"auto", "unpaired"}
    assert data["uncertain"][0]["status"] == "uncertain_chiban"
    assert client.get(f"{review.HEALTH_PATH}").json() == {"app": review.HEALTH_MARKER}
    assert client.get("/").status_code == 200


@pytest.mark.parametrize("name", ["nothing", "..", "zumen.chiban", "a b"])
def test_unknown_or_odd_names_are_404(client, name):
    assert client.get(f"/api/drawings/{name}").status_code == 404


def test_image_is_png_of_unmasked_drawing_kept_in_protected_cache(client, project):
    response = client.get("/api/drawings/zumen/image")
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"
    previews = list((project / review.PREVIEW_DIR).glob("*.png"))
    assert len(previews) == 1
    assert "raw" in previews[0].parent.name  # フックの部品名の判定が効く置き場所
    assert (project / "data" / "raw" / "zumen.tif").read_bytes()  # 元の TIFF はそのまま


def test_update_records_removed_added_and_uncertain(client, project):
    data = client.get("/api/drawings/zumen").json()
    masks = [{**m, "origin": m["status"]} for m in data["masks"]]
    masks[1]["status"] = "removed"
    masks.append({"quad": rect(800, 600, 40, 20), "status": "added", "origin": "added"})
    masks.append({"quad": data["uncertain"][0]["quad"], "status": "added", "origin": "uncertain_chiban", "uncertain_index": 0})
    saved = client.put("/api/drawings/zumen", json={"masks": masks}).json()
    assert [m["status"] for m in saved["masks"]] == ["auto", "removed", "added", "added"]
    assert saved["masks"][1]["origin"] == "unpaired"
    assert saved["uncertain"][0]["decision"] == "mask"
    assert saved["confirmed"] is False
    assert result_of(project)["masks"] == saved["masks"]


@pytest.mark.parametrize(
    "mask",
    [
        {"quad": rect(10, 10, 4, 4), "status": "auto", "origin": "nonsense"},
        {"quad": rect(10, 10, 4, 4), "status": "deleted", "origin": "auto"},
        {"quad": [[0, 0], [1, 1]], "status": "auto"},
        {"quad": [[0, 0, 0], [1, 1, 1], [2, 2, 2]], "status": "auto"},
        {"quad": rect(10, 10, 4, 4), "status": "added", "origin": "uncertain_chiban", "uncertain_index": 5},
    ],
)
def test_update_rejects_bad_masks(client, mask):
    assert client.put("/api/drawings/zumen", json={"masks": [mask]}).status_code == 422


def test_confirm_repaints_from_unmasked_drawing(client, project):
    data = client.get("/api/drawings/zumen").json()
    masks = [{**m, "origin": m["status"]} for m in data["masks"]]
    unpaired = next(m for m in masks if m["status"] == "unpaired")
    unpaired["status"] = "removed"
    masks.append({"quad": rect(800, 600, 40, 20), "status": "added", "origin": "added"})
    client.put("/api/drawings/zumen", json={"masks": masks})
    assert masked_pixels(project)[200, 600].tolist() == BLACK  # 確定前は自動の塗りのまま

    confirmed = client.post("/api/drawings/zumen/confirm").json()
    assert confirmed["confirmed"] is True and confirmed["confirmed_at"]
    assert sorted(m["status"] for m in confirmed["masks"]) == ["added", "confirmed", "removed"]
    assert next(m for m in confirmed["masks"] if m["status"] == "confirmed")["origin"] == "auto"
    pixels = masked_pixels(project)
    assert pixels[200, 600].tolist() == [GRAY] * 3  # 外した塗りは元に戻る
    assert pixels[600, 800].tolist() == BLACK  # 追加した塗り
    assert pixels.shape == (700, 1000, 3)
    tfw = project / "data" / "masked" / "zumen.tfw"
    assert tfw.read_bytes() == (project / "data" / "raw" / "zumen.tfw").read_bytes()
    assert client.get("/api/drawings").json()["drawings"][0]["confirmed"] is True

    client.put("/api/drawings/zumen", json={"masks": confirmed["masks"]})  # 確定後に保存し直すと未確定に戻る
    assert result_of(project)["confirmed"] is False


def test_confirm_fails_when_unmasked_drawing_is_gone(client, project):
    (project / "data" / "raw" / "zumen.tif").unlink()
    assert client.post("/api/drawings/zumen/confirm").status_code == 404


def test_server_binds_only_to_loopback():
    assert review.HOST == "127.0.0.1"
    assert review.review_url(8765, "zumen") == "http://127.0.0.1:8765/?drawing=zumen"
    assert load_config()["review"]["port"] == 8765


def test_html_uses_no_external_resources():
    html = (review.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "http://" not in html and "https://" not in html  # オフラインの PC で動く
