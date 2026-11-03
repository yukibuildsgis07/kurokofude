"""detect / pair / mask / evaluate / ocr の単体テスト。OCR エンジンは呼ばない。"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile

from kurokofude import ocr as ocr_module
from kurokofude.config import load_config
from kurokofude.detect import classify_boxes, normalize_text
from kurokofude.evaluate import average_scores, evaluate
from kurokofude.geometry import FrameResolver, angle_difference_deg, quad_frame
from kurokofude.mask import fill_polygons, write_outputs
from kurokofude.pair import pair_names
from kurokofude.refine import cut_offset, refine_pair, tighten

FIXTURES = Path(__file__).parent / "fixtures"
CONFIG = load_config()
IMAGE_SIZE = (1000, 700)
FIXTURE_SURVIVAL_FLOOR = 80.0


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
    "ocr_text,expected",
    [
        ("１２３－４", "123-4"),
        ("l2O", "120"),
        ("273ー9", "273-9"),
        ("12 3", "123"),
        ("273/9", "273-9"),
        ("273\\9", "273-9"),
        ("273|9", "273-9"),
        ("273ノ9", "273-9"),
        ("273‐9", "273-9"),
        ("3-", "3"),
        (".12:", "12"),
        ("S=1:500", "S=1:500"),
        ("273□", "273□"),
    ],
)
def test_normalize_text(ocr_text, expected):
    assert normalize_text(ocr_text) == expected


def test_normalized_chiban_gets_lower_score():
    clean = box(rect(100, 100, 60, 20), "273-9")
    fixed = box(rect(300, 100, 60, 20), "273/9")
    wide = box(rect(500, 100, 60, 20), "２７３－９")
    result = classify_boxes([clean, fixed, wide], IMAGE_SIZE, CONFIG)
    factor = CONFIG["chiban"]["normalized_score_factor"]
    assert [b["score"] for b in result.chiban] == pytest.approx([0.9, 0.9 * factor, 0.9])
    assert [b["text"] for b in result.chiban] == ["273-9"] * 3


def test_classify_splits_into_three_kinds():
    chiban = box(rect(100, 100, 60, 20), "123-4")
    misread = box(rect(100, 200, 60, 20), "273.19")
    too_long = box(rect(100, 400, 90, 20), "123456")
    name = box(rect(100, 300, 60, 20), "")
    title = box(rect(900, 650, 60, 20), "2024")  # 表題欄の領域
    title_name = box(rect(900, 620, 60, 20), "")
    result = classify_boxes([chiban, misread, too_long, name, title, title_name], IMAGE_SIZE, CONFIG)
    assert [b["text"] for b in result.chiban] == ["123-4"]
    assert [b["text"] for b in result.uncertain] == ["273.19", "123456"]
    assert result.text == [name]


def test_number_next_to_scale_symbol_is_not_chiban_but_is_kept():
    number, symbol = box(rect(300, 100, 40, 20), "500"), box(rect(250, 100, 40, 20), "S=1:")
    result = classify_boxes([number, symbol], IMAGE_SIZE, CONFIG)
    assert result.chiban == []
    assert number["text"] in [b["text"] for b in result.uncertain]


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


def test_pair_takes_every_fragment_of_a_split_name():
    chiban = box(rect(300, 300, 80, 20), "12-3")
    family = box(rect(270, 326, 40, 18))
    given = box(rect(330, 326, 40, 18))
    pairs = pair_names([chiban, family, given], [chiban], CONFIG["pair"])
    assert {id(p.source) for p in pairs} == {id(family), id(given)}


@pytest.mark.parametrize("angle", [0, 25])
def test_one_digit_chiban_borrows_angle_from_neighbors(angle):
    """縦長の1桁の地番でも、近傍の細長い枠の角度で真下の氏名を選べる。"""
    theta = np.radians(angle)
    down = np.array([np.sin(theta), np.cos(theta)])
    center = np.array([300.0, 300.0])
    chiban = box(rect(*center, 12, 22, angle), "5")
    name = box(rect(*(center + down * 24), 80, 18, angle))
    neighbors = [box(rect(300 + dx, 150, 80, 18, angle)) for dx in (-100, 0, 100)]
    boxes = [chiban, name, *neighbors]
    assert pair_names([name], [chiban], CONFIG["pair"]) == []  # 自分の長辺だと倒れて選べない
    frame_of = FrameResolver(boxes, CONFIG["angle"]).frame
    pairs = pair_names([name], [chiban], CONFIG["pair"], frame_of)
    assert [p.source for p in pairs] == [name]


def test_resolver_uses_overall_median_then_zero_and_keeps_long_boxes():
    long_box = rect(100, 100, 80, 18, 20)
    assert FrameResolver([box(long_box)], CONFIG["angle"]).frame(long_box).angle_deg == pytest.approx(20)
    square = rect(600, 600, 20, 20, 45)
    assert FrameResolver([box(long_box)], CONFIG["angle"]).frame(square).angle_deg == pytest.approx(20)
    assert FrameResolver([], CONFIG["angle"]).frame(square).angle_deg == 0.0


# refine


def blank_gray(width=400, height=300):
    return np.full((height, width), 240, dtype=np.uint8)


def draw_ink(gray, quad):
    cv2.fillPoly(gray, [np.round(np.array(quad)).astype(np.int32)], 20)


@pytest.mark.parametrize("angle", [0, 20, -30])
def test_cut_falls_in_the_ink_gap_between_chiban_and_name(angle):
    """OCR の枠が重なっていても、地番と氏名のインクの間で切れる。"""
    theta = np.radians(angle)
    down = np.array([np.sin(theta), np.cos(theta)])
    center = np.array([200.0, 150.0])
    gray = blank_gray()
    draw_ink(gray, rect(*center, 50, 16, angle))  # 地番のインク（高さ16）
    draw_ink(gray, rect(*(center + down * 24), 70, 16, angle))  # 氏名のインク（上端は16、地番の下端は8）
    chiban_box = rect(*center, 54, 28, angle)  # OCR の枠は下に余白がある（下端は14）
    name_box = rect(*(center + down * 24), 74, 24, angle)  # 上端は12で、地番の枠と重なる
    offset = cut_offset(quad_frame(chiban_box), name_box, gray, CONFIG["mask"])
    assert 8 <= offset <= 16


def test_cut_uses_midpoint_when_band_is_too_narrow():
    chiban_box = rect(200, 150, 54, 20)  # 下端は 160
    name_box = rect(200, 170.5, 74, 20)  # 上端は 160.5
    offset = cut_offset(quad_frame(chiban_box), name_box, blank_gray(), CONFIG["mask"])
    assert offset == pytest.approx(10.25)


def test_cut_uses_midpoint_when_band_has_no_valley():
    chiban_box = rect(200, 150, 54, 20)
    name_box = rect(200, 180, 74, 20)  # 帯は 160〜170 で、インクなし
    offset = cut_offset(quad_frame(chiban_box), name_box, blank_gray(), CONFIG["mask"])
    assert offset == pytest.approx(15)


def test_tighten_shrinks_to_ink_with_padding_and_never_grows():
    gray = blank_gray()
    draw_ink(gray, [[180, 140], [220, 140], [220, 160], [180, 160]])
    box_quad = rect(200, 150, 80, 40)
    frame = quad_frame(box_quad)
    result = np.array(tighten(box_quad, frame, gray, CONFIG["mask"]))
    assert result[:, 0].min() == pytest.approx(177, abs=1.5)
    assert result[:, 0].max() == pytest.approx(224, abs=1.5)
    assert result[:, 1].min() == pytest.approx(137, abs=1.5)
    assert result[:, 1].max() == pytest.approx(164, abs=1.5)
    assert result[:, 0].min() >= 160 and result[:, 0].max() <= 240


def test_tighten_keeps_axis_with_ink_across_it():
    """筆界線が縦に貫く枠は、上下を詰めない。横は詰める。"""
    gray = blank_gray()
    draw_ink(gray, [[198, 100], [202, 100], [202, 200], [198, 200]])
    box_quad = rect(200, 150, 80, 40)
    result = np.array(tighten(box_quad, quad_frame(box_quad), gray, CONFIG["mask"]))
    assert result[:, 1].min() == pytest.approx(130)
    assert result[:, 1].max() == pytest.approx(170)
    assert result[:, 0].max() - result[:, 0].min() < 20


def test_tighten_leaves_box_without_ink_unchanged():
    box_quad = rect(200, 150, 80, 40)
    assert tighten(box_quad, quad_frame(box_quad), blank_gray(), CONFIG["mask"]) == box_quad


def test_refine_pair_does_not_paint_chiban_ink():
    gray = blank_gray()
    draw_ink(gray, rect(200, 150, 50, 16))
    draw_ink(gray, rect(200, 174, 70, 16))
    chiban = box(rect(200, 150, 54, 28), "12")
    pairs = pair_names([box(rect(200, 174, 74, 24))], [chiban], CONFIG["pair"])
    refine_pair(pairs[0], gray, CONFIG["mask"])
    quad = np.array(pairs[0].quad)
    assert quad[:, 1].min() >= 158
    assert quad[:, 1].max() >= 182


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


def test_result_records_unpaired_and_uncertain(tmp_path):
    source = tmp_path / "a.tif"
    tifffile.imwrite(source, np.full((60, 60, 3), 200, dtype=np.uint8), photometric="rgb")
    uncertain = box(rect(10, 10, 8, 8), "273.19")
    lone = rect(40, 40, 10, 10)
    out_tif, result = write_outputs(source, tmp_path / "out", [], [], CONFIG["mask"], [lone], [uncertain])
    assert result["masks"] == [{"quad": lone, "chiban": None, "status": "unpaired"}]
    assert result["uncertain"][0]["status"] == "uncertain_chiban"
    painted = tifffile.imread(out_tif)
    assert painted[40, 40].tolist() == [0, 0, 0]
    assert painted[10, 10].tolist() == [200, 200, 200]  # (b) は塗らない


# evaluate


def truth_item(chiban_quad, name_quad):
    return {"chiban": {"text": "1", "quad": chiban_quad}, "name": {"quad": name_quad}}


THRESHOLD = CONFIG["mask"]["ink_threshold"]


def gray_with_ink(*quads):
    gray = np.full((100, 120), 240, dtype=np.uint8)
    for quad in quads:
        cv2.fillPoly(gray, [np.round(np.array(quad)).astype(np.int32)], 20)
    return gray


def test_evaluate_exact_mask_is_perfect():
    name, chiban = rect(50, 50, 40, 10), rect(50, 30, 20, 10)
    truth = {"items": [truth_item(chiban, name)]}
    result = {
        "chiban": [{"text": "1", "center": [50, 30], "quad": chiban, "score": 1.0}],
        "masks": [{"quad": name, "chiban": "1", "status": "auto"}],
    }
    scores = evaluate(truth, result, gray_with_ink(chiban, name), THRESHOLD)
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
        "chiban": [{"text": "9", "center": [100, 90], "quad": chiban, "score": 1.0}],
        "masks": [{"quad": half_outside, "chiban": "9", "status": "auto"}, {"quad": chiban, "chiban": "9", "status": "added"}],
    }
    scores = evaluate(truth, result, gray_with_ink(chiban, name), THRESHOLD)
    assert scores["name_coverage_pct"] == 50.0
    assert scores["chiban_survival_pct"] == 0.0
    assert scores["chiban_detection_pct"] == 0.0
    assert scores["false_positives"] == 1
    assert 0 < scores["overmask_pct"] < 100


def test_evaluate_survival_ignores_paint_on_blank_margin_of_chiban_box():
    """正解の枠の余白に塗りが触れただけでは、地番は生きている（§11.5）。"""
    chiban_box = [[40, 20], [60, 20], [60, 40], [40, 40]]
    ink_only = [[44, 22], [56, 22], [56, 32], [44, 32]]
    touching_margin = [[30, 36], [70, 36], [70, 46], [30, 46]]
    truth = {"items": [truth_item(chiban_box, rect(50, 41, 40, 10))]}
    result = {"chiban": [], "masks": [{"quad": touching_margin, "chiban": None, "status": "unpaired"}]}
    assert evaluate(truth, result, gray_with_ink(ink_only), THRESHOLD)["chiban_survival_pct"] == 100.0
    touching_ink = [[30, 30], [70, 30], [70, 46], [30, 46]]
    result["masks"][0]["quad"] = touching_ink
    assert evaluate(truth, result, gray_with_ink(ink_only), THRESHOLD)["chiban_survival_pct"] == 0.0


def test_evaluate_ignores_removed_masks_and_counts_review_items():
    name, chiban = rect(50, 50, 40, 10), rect(50, 30, 20, 10)
    truth = {"items": [truth_item(chiban, name)]}
    result = {
        "chiban": [],
        "uncertain": [{"text": "273.19", "quad": chiban, "score": 0.5, "status": "uncertain_chiban"}],
        "masks": [
            {"quad": name, "chiban": "1", "status": "removed"},
            {"quad": rect(100, 80, 10, 10), "chiban": None, "status": "unpaired"},
        ],
    }
    scores = evaluate(truth, result, gray_with_ink(chiban, name), THRESHOLD)
    assert scores["name_coverage_pct"] == 0.0
    assert scores["unpaired_masks"] == 1
    assert scores["uncertain_chiban"] == 1


def test_evaluate_reports_scores_excluding_line_overlaps():
    clear_chiban, clear_name = rect(30, 20, 20, 10), rect(30, 40, 40, 10)
    hard_chiban, hard_name = rect(90, 20, 20, 10), rect(90, 40, 40, 10)
    truth = {
        "items": [
            {**truth_item(clear_chiban, clear_name), "overlaps_line": False},
            {**truth_item(hard_chiban, hard_name), "overlaps_line": True},
        ]
    }
    result = {"chiban": [], "masks": [{"quad": clear_name, "chiban": None, "status": "unpaired"}]}
    scores = evaluate(truth, result, gray_with_ink(clear_chiban, hard_chiban), THRESHOLD)
    assert scores["name_coverage_pct"] == 50.0
    assert scores["excluding_line_overlaps"]["name_coverage_pct"] == 100.0
    assert scores["excluding_line_overlaps"]["truth_items"] == 1


def test_average_scores_handles_nested_values():
    rows = [{"a": 1.0, "b": {"c": 2.0}}, {"a": 3.0, "b": {"c": 4.0}}]
    assert average_scores(rows) == {"a": 2.0, "b": {"c": 3.0}}


# ocr（エンジンなし）


def test_scrub_drops_names_but_keeps_numbers():
    keep = CONFIG["ocr"]["keep_char_class"], CONFIG["ocr"]["digit_ratio_min"]
    assert ocr_module.scrub_text("架空 太郎", *keep) == ""
    assert ocr_module.scrub_text("273-4", *keep) == "273-4"
    assert ocr_module.scrub_text("S=1:500", *keep) == "S=1:500"
    assert ocr_module.scrub_text("273ノ9", *keep) == "273ノ9"
    # 数字が主体なら数字の割合だけ分かる形で残し、許可外の文字は保存しない
    assert ocr_module.scrub_text("273田", *keep) == "273□"
    assert ocr_module.scrub_text("3田中", *keep) == ""


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


# batch


def test_batch_continues_after_failure_and_reports_it(tmp_path, monkeypatch, capsys):
    from kurokofude import cli
    from kurokofude.pipeline import MaskSummary

    for name in ("a.tif", "b.tif", "c.tiff", "note.txt"):
        (tmp_path / name).write_bytes(b"")
    seen = []

    def fake_mask_file(path, _config, out_dir, _cache):
        seen.append(path.name)
        if path.name == "b.tif":
            raise ValueError("壊れた図面")
        return MaskSummary(out_dir / path.name, 1, 1, 1)

    monkeypatch.setattr(cli, "mask_file", fake_mask_file)
    assert cli.main(["batch", str(tmp_path), "--out", str(tmp_path / "out")]) == 1
    assert seen == ["a.tif", "b.tif", "c.tiff"]
    assert "2/3" in capsys.readouterr().out


# フィクスチャ（実際の OCR 結果）


@pytest.fixture
def small_ocr():
    return json.loads((FIXTURES / "sample_small.ocr.json").read_text(encoding="utf-8"))


def test_fixture_has_no_name_strings(small_ocr):
    assert not any(re.search(r"[぀-鿿]", b["text"]) for b in small_ocr)


def test_pipeline_on_fixture_keeps_most_chiban(small_ocr):
    truth = json.loads((FIXTURES / "sample_small.truth.json").read_text(encoding="utf-8"))
    classified = classify_boxes(small_ocr, tuple(truth["image_size"]), CONFIG)
    assert classified.chiban
    pairs = pair_names(classified.text, classified.chiban, CONFIG["pair"])
    assert pairs
    gray = cv2.cvtColor(tifffile.imread(FIXTURES / "sample_small.tif"), cv2.COLOR_RGB2GRAY)
    for pair in pairs:
        refine_pair(pair, gray, CONFIG["mask"])
    result = {
        "chiban": [{"text": b["text"], "center": np.mean(b["quad"], axis=0).tolist()} for b in classified.chiban],
        "masks": [{"quad": p.quad, "chiban": p.chiban["text"], "status": "auto"} for p in pairs],
    }
    # 線と重なる難しい注記を含む1枚の現状値を下回らないこと。実物化の後で見直す
    assert evaluate(truth, result, gray, THRESHOLD)["chiban_survival_pct"] >= FIXTURE_SURVIVAL_FLOOR
