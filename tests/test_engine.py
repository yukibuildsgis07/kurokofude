"""detect / pair / mask / evaluate / ocr の単体テスト。OCR エンジンは呼ばない。"""

import json
import re
from pathlib import Path

import numpy as np
import pytest
import tifffile

from kurokofude import ocr as ocr_module
from kurokofude.config import load_config
from kurokofude.detect import detect_chiban, normalize_text
from kurokofude.evaluate import evaluate
from kurokofude.geometry import angle_difference_deg, quad_frame
from kurokofude.mask import fill_polygons, write_outputs
from kurokofude.pair import cut_at_chiban_bottom, pair_names

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = load_config()
IMAGE_SIZE = (1000, 700)


def rect(cx, cy, w, h, angle_deg=0.0):
    theta = np.radians(angle_deg)
    u = np.array([np.cos(theta), -np.sin(theta)])
    v = np.array([np.sin(theta), np.cos(theta)])
    c = np.array([cx, cy], dtype=float)
    corners = [c - u * w / 2 - v * h / 2, c + u * w / 2 - v * h / 2, c + u * w / 2 + v * h / 2, c - u * w / 2 + v * h / 2]
    return [p.tolist() for p in corners]


def box(quad, text=""):
    return {"text": text, "score": 0.9, "quad": quad}


# detect


@pytest.mark.parametrize(
    "raw,expected", [("１２３－４", "123-4"), ("l2O", "120"), ("273ー9", "273-9"), ("12 3", "123")]
)
def test_normalize_text(raw, expected):
    assert normalize_text(raw) == expected


def test_detect_keeps_chiban_and_rejects_others():
    boxes = [
        box(rect(100, 100, 60, 20), "123-4"),
        box(rect(100, 200, 60, 20), "S=1:500"),
        box(rect(100, 300, 60, 20), ""),
        box(rect(900, 650, 60, 20), "2024"),  # 表題欄の領域
        box(rect(100, 400, 90, 20), "123456"),  # 桁数オーバー
    ]
    assert [b["text"] for b in detect_chiban(boxes, IMAGE_SIZE, CONFIG)] == ["123-4"]


def test_detect_rejects_number_next_to_scale_symbol():
    boxes = [box(rect(300, 100, 40, 20), "500"), box(rect(250, 100, 40, 20), "S=1:")]
    assert detect_chiban(boxes, IMAGE_SIZE, CONFIG) == []


# geometry


def test_angle_difference_wraps_half_turn():
    assert angle_difference_deg(89, -89) == pytest.approx(2)


def test_quad_frame_of_slanted_box_points_down():
    frame = quad_frame(rect(100, 100, 60, 20, angle_deg=30))
    assert frame.angle_deg == pytest.approx(30)
    assert frame.width == pytest.approx(60)
    assert frame.height == pytest.approx(20)
    assert frame.v[1] > 0


# pair


@pytest.mark.parametrize("angle", [0, 25, -35])
def test_pair_selects_box_directly_below_along_the_slant(angle):
    theta = np.radians(angle)
    down = np.array([np.sin(theta), np.cos(theta)])
    chiban = box(rect(300, 300, 60, 20, angle), "12-3")
    name_center = np.array([300, 300]) + down * 26
    name = box(rect(*name_center, 90, 18, angle))
    other = box(rect(600, 300, 90, 18, angle))
    pairs = pair_names([chiban, name, other], [chiban], CONFIG["pair"])
    assert len(pairs) == 1
    assert pairs[0].chiban is chiban
    assert np.mean(pairs[0].quad, axis=0) == pytest.approx(name_center, abs=1.5)


def test_pair_rejects_angle_mismatch_box_above_and_far_box():
    chiban = box(rect(300, 300, 60, 20), "12-3")
    rotated = box(rect(300, 326, 90, 18, 30))
    above = box(rect(300, 274, 90, 18))
    far = box(rect(300, 400, 90, 18))
    sideways = box(rect(500, 326, 90, 18))
    assert pair_names([chiban, rotated, above, far, sideways], [chiban], CONFIG["pair"]) == []


def test_pair_never_masks_another_chiban_and_takes_nearest():
    upper = box(rect(300, 300, 60, 20), "12-3")
    lower = box(rect(300, 326, 60, 20), "12-4")
    near = box(rect(300, 352, 90, 18))
    pairs = pair_names([upper, lower, near], [upper, lower], CONFIG["pair"])
    assert [p.chiban["text"] for p in pairs] == ["12-4"]


def test_cut_removes_overlap_with_chiban_on_a_slant():
    angle = 20
    chiban_frame = quad_frame(rect(300, 300, 60, 20, angle))
    theta = np.radians(angle)
    down = np.array([np.sin(theta), np.cos(theta)])
    tall_name = rect(*(np.array([300, 300]) + down * 18), 90, 30, angle)
    cut = np.array(cut_at_chiban_bottom(tall_name, chiban_frame, margin_px=2))
    depth = (cut - chiban_frame.center) @ chiban_frame.v
    assert depth.min() == pytest.approx(chiban_frame.height / 2 + 2)


# mask


def test_fill_changes_only_inside_and_keeps_shape():
    image = np.full((100, 200, 3), 255, dtype=np.uint8)
    painted = fill_polygons(image, [rect(100, 50, 40, 20)], [0, 0, 0])
    assert painted.shape == image.shape
    assert painted[50, 100].tolist() == [0, 0, 0]
    assert painted[5, 5].tolist() == [255, 255, 255]
    assert image[50, 100].tolist() == [255, 255, 255]


def test_outputs_keep_pixels_world_file_and_geotiff_tags(tmp_path):
    source = tmp_path / "in" / "a.tif"
    source.parent.mkdir()
    image = np.full((50, 80, 3), 200, dtype=np.uint8)
    pixel_scale = (0.5, 0.5, 0.0)
    tifffile.imwrite(source, image, photometric="rgb", extratags=[(33550, "d", 3, pixel_scale, True)])
    source.with_suffix(".tfw").write_text("0.5\n0\n0\n-0.5\n10\n20\n", encoding="ascii")

    chiban = box(rect(40, 15, 20, 8), "12")
    pairs = pair_names([chiban, box(rect(40, 27, 30, 8))], [chiban], CONFIG["pair"])
    out_tif, result = write_outputs(source, tmp_path / "out", [chiban], pairs, CONFIG["mask"])

    assert out_tif.with_suffix(".tfw").read_bytes() == source.with_suffix(".tfw").read_bytes()
    with tifffile.TiffFile(out_tif) as tif:
        painted = tif.pages[0].asarray()
        assert tif.pages[0].tags[33550].value == pixel_scale
    assert painted.shape == image.shape
    assert painted[27, 40].tolist() == [0, 0, 0]
    assert painted[2, 2].tolist() == [200, 200, 200]
    assert result["image_size"] == [80, 50]
    assert result["chiban"][0]["masked_below"] is True
    saved = json.loads((tmp_path / "out" / "a.chiban.json").read_text(encoding="utf-8"))
    assert saved["masks"][0]["status"] == "auto"


def test_refuses_to_overwrite_the_input(tmp_path):
    source = tmp_path / "a.tif"
    tifffile.imwrite(source, np.zeros((10, 10, 3), dtype=np.uint8), photometric="rgb")
    with pytest.raises(ValueError):
        write_outputs(source, tmp_path, [], [], CONFIG["mask"])


# evaluate


def truth_item(chiban_quad, name_quad):
    return {"chiban": {"text": "1", "quad": chiban_quad}, "name": {"quad": name_quad}}


def test_evaluate_exact_mask_is_perfect():
    name, chiban = rect(50, 50, 40, 10), rect(50, 30, 20, 10)
    truth = {"items": [truth_item(chiban, name)]}
    result = {
        "chiban": [{"text": "1", "center": [50, 30], "quad": chiban, "score": 1.0}],
        "masks": [{"quad": name, "chiban": "1", "status": "auto"}],
    }
    scores = evaluate(truth, result)
    assert scores["name_coverage_pct"] == 100.0
    assert scores["overmask_pct"] == 0.0
    assert scores["chiban_survival_pct"] == 100.0
    assert scores["chiban_detection_pct"] == 100.0
    assert scores["false_positives"] == 0


def test_evaluate_counts_half_coverage_overmask_and_painted_chiban():
    chiban = rect(50, 30, 20, 10)
    name = [[30, 45], [70, 45], [70, 55], [30, 55]]
    half_outside = [[50, 45], [90, 45], [90, 55], [50, 55]]
    truth = {"items": [truth_item(chiban, name)]}
    result = {
        "chiban": [{"text": "9", "center": [200, 200], "quad": chiban, "score": 1.0}],
        "masks": [{"quad": half_outside, "chiban": "9", "status": "auto"}, {"quad": chiban, "chiban": "9", "status": "added"}],
    }
    scores = evaluate(truth, result)
    assert scores["name_coverage_pct"] == 50.0
    assert scores["chiban_survival_pct"] == 0.0
    assert scores["chiban_detection_pct"] == 0.0
    assert scores["false_positives"] == 1
    assert 0 < scores["overmask_pct"] < 100


def test_evaluate_ignores_removed_masks():
    name, chiban = rect(50, 50, 40, 10), rect(50, 30, 20, 10)
    truth = {"items": [truth_item(chiban, name)]}
    result = {"chiban": [], "masks": [{"quad": name, "chiban": "1", "status": "removed"}]}
    assert evaluate(truth, result)["name_coverage_pct"] == 0.0


# ocr（エンジンなし）


def test_scrub_drops_names_but_keeps_numbers():
    pattern = CONFIG["ocr"]["keep_text_pattern"]
    assert ocr_module.scrub_text("架空 太郎", pattern) == ""
    assert ocr_module.scrub_text("273-4", pattern) == "273-4"
    assert ocr_module.scrub_text("S=1:500", pattern) == "S=1:500"


def test_tile_origins_cover_image_with_overlap():
    origins = ocr_module.tile_origins(10000, 4000, 200)
    assert origins[0] == 0
    assert origins[-1] == 6000
    assert all(b - a <= 3800 for a, b in zip(origins, origins[1:]))
    assert ocr_module.tile_origins(3000, 4000, 200) == [0]


def test_merge_duplicates_keeps_best_score():
    weak = {"text": "1", "score": 0.5, "quad": rect(100, 100, 40, 20)}
    strong = {"text": "1", "score": 0.9, "quad": rect(101, 100, 40, 20)}
    apart = {"text": "2", "score": 0.7, "quad": rect(300, 100, 40, 20)}
    merged = ocr_module.merge_duplicates([weak, strong, apart], 0.5)
    assert strong in merged and apart in merged and weak not in merged


def test_cache_prevents_second_run(tmp_path, monkeypatch):
    image = tmp_path / "a.tif"
    tifffile.imwrite(image, np.zeros((10, 10, 3), dtype=np.uint8), photometric="rgb")
    calls = []

    def fake_engine(_image, _cfg):
        calls.append(1)
        return [{"text": "", "score": 0.5, "quad": rect(5, 5, 4, 2)}]

    monkeypatch.setattr(ocr_module, "run_engine", fake_engine)
    first = ocr_module.run_ocr(image, CONFIG["ocr"], tmp_path / "cache")
    second = ocr_module.run_ocr(image, CONFIG["ocr"], tmp_path / "cache")
    assert first == second
    assert len(calls) == 1


# フィクスチャ（実際の OCR 結果）


@pytest.fixture
def small_ocr():
    return json.loads((FIXTURES / "sample_small.ocr.json").read_text(encoding="utf-8"))


def test_fixture_has_no_name_strings(small_ocr):
    assert not any(re.search(r"[぀-鿿]", b["text"]) for b in small_ocr)


def test_pipeline_on_fixture_never_paints_chiban(small_ocr):
    truth = json.loads((FIXTURES / "sample_small.truth.json").read_text(encoding="utf-8"))
    chiban_boxes = detect_chiban(small_ocr, tuple(truth["image_size"]), CONFIG)
    assert chiban_boxes
    pairs = pair_names(small_ocr, chiban_boxes, CONFIG["pair"])
    assert pairs
    result = {
        "chiban": [{"text": b["text"], "center": np.mean(b["quad"], axis=0).tolist()} for b in chiban_boxes],
        "masks": [{"quad": p.quad, "chiban": p.chiban["text"], "status": "auto"} for p in pairs],
    }
    assert evaluate(truth, result)["chiban_survival_pct"] == 100.0
