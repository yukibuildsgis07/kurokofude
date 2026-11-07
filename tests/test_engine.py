"""detect / pair / mask / evaluate / ocr の単体テスト。OCR エンジンは呼ばない。"""

import json
import re
from pathlib import Path

import cv2
import numpy as np
import pytest
import tifffile
from shapely.geometry import Polygon

from kurokofude import ocr as ocr_module
from kurokofude.config import load_config
from kurokofude.detect import classify_boxes, normalize_text
from kurokofude.evaluate import average_scores, evaluate
from kurokofude.geometry import FrameResolver, angle_difference_deg, quad_frame
from kurokofude.mixed import resolve_mixed
from kurokofude.mask import fill_polygons, protected_area, subtract_protected, write_outputs
from kurokofude.pair import Pair, pair_names
from kurokofude.refine import cut_offset, refine_lone_text, refine_pair, tight_box, tighten

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
        ("273.19", "273-19"),
        ("273,19", "273-19"),
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
    misread = box(rect(100, 200, 60, 20), "273:19")
    too_long = box(rect(100, 400, 90, 20), "123456")
    name = box(rect(100, 300, 60, 20), "")
    result = classify_boxes([chiban, misread, too_long, name], IMAGE_SIZE, CONFIG)
    assert [b["text"] for b in result.chiban] == ["123-4"]
    assert [b["text"] for b in result.uncertain] == ["273:19", "123456"]
    assert result.text == [name]
    assert result.title_text == []


def test_exclude_region_stops_only_chiban_by_default():
    """表題欄の中の数字は地番にしない。数字を含む文字は残し、それ以外は塗る（§12.1、§18.1）。"""
    number = box(rect(900, 650, 60, 20), "2024")
    author = box(rect(900, 620, 60, 20), "")
    result = classify_boxes([number, author], IMAGE_SIZE, CONFIG)
    assert result.chiban == [] and result.uncertain == []
    assert [b["text"] for b in result.title_meta] == ["2024"]
    assert result.title_text == [author]
    assert result.text == []


@pytest.mark.parametrize(
    "text",
    ["図郭番号J1□□□", "□□R6.4.1", "S=1:500", "□□□", "現況平面図", "□□平面図", "境界確定図□"],
)
def test_title_meta_keeps_digits_and_title_words(text):
    """表題欄の中では、数字を1つでも含む文字と図名の定型語を含む文字を残す。"""
    item = box(rect(900, 650, 60, 20), text)
    result = classify_boxes([item], IMAGE_SIZE, CONFIG)
    if text == "□□□":  # 数字も定型語もない（作成者名や会社名）
        assert result.title_text == [item] and result.title_meta == []
    else:
        assert [b["text"] for b in result.title_meta] == [normalize_text(text)]


def test_title_words_are_not_special_outside_the_title_block():
    item = box(rect(100, 100, 60, 20), "現況平面図")
    result = classify_boxes([item], IMAGE_SIZE, CONFIG)
    assert result.text == [item] and result.title_meta == []


def test_exclude_mode_all_ignores_the_region():
    config = {**CONFIG, "exclude_mode": "all"}
    boxes = [box(rect(900, 650, 60, 20), "2024"), box(rect(900, 620, 60, 20), "")]
    result = classify_boxes(boxes, IMAGE_SIZE, config)
    assert (result.chiban, result.uncertain, result.text, result.title_text) == ([], [], [], [])


def test_exclude_mode_rejects_unknown_value():
    with pytest.raises(ValueError):
        classify_boxes([], IMAGE_SIZE, {**CONFIG, "exclude_mode": "none"})


def test_bottom_strip_preset_replaces_default_region():
    preset = Path(__file__).parents[1] / "configs" / "bottom_strip.yaml"
    config = load_config(preset)
    assert config["exclude_regions"] == [[0.0, 0.90, 1.0, 1.0]]
    assert config["exclude_mode"] == "chiban_only"
    in_strip = box(rect(100, 680, 60, 20), "12")
    assert classify_boxes([in_strip], IMAGE_SIZE, config).chiban == []


def test_normalized_chiban_gets_lower_score():
    clean = box(rect(100, 100, 60, 20), "273-9")
    fixed = box(rect(300, 100, 60, 20), "273/9")
    wide = box(rect(500, 100, 60, 20), "２７３－９")
    result = classify_boxes([clean, fixed, wide], IMAGE_SIZE, CONFIG)
    factor = CONFIG["chiban"]["normalized_score_factor"]
    assert [b["score"] for b in result.chiban] == pytest.approx([0.9, 0.9 * factor, 0.9])
    assert [b["text"] for b in result.chiban] == ["273-9"] * 3


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
    chiban = box(rect(*center, 16, 22, angle), "5")  # 実際の OCR の枠に近い寸法
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


def test_tight_box_excludes_name_ink_at_bottom_of_ocr_box():
    """OCR の地番枠の下側に氏名の上端が入っていても、地番のインクまで詰まる（§13.1）。"""
    gray = blank_gray()
    draw_ink(gray, [[180, 140], [220, 140], [220, 156], [180, 156]])  # 地番
    draw_ink(gray, [[165, 162], [235, 162], [235, 180], [165, 180]])  # 氏名（枠に上端が入る）
    ocr_box = [[170, 130], [230, 130], [230, 168], [170, 168]]
    result = np.array(tight_box(ocr_box, quad_frame(ocr_box), gray, CONFIG["mask"]))
    assert result[:, 1].max() <= 158.5
    assert result[:, 1].min() == pytest.approx(139, abs=1.5)
    assert result[:, 0].min() == pytest.approx(179, abs=1.5)
    assert result[:, 0].max() == pytest.approx(221, abs=1.5)


def test_tight_box_cuts_at_valley_when_a_line_joins_chiban_and_name():
    gray = blank_gray()
    draw_ink(gray, [[180, 140], [220, 140], [220, 156], [180, 156]])
    draw_ink(gray, [[165, 162], [235, 162], [235, 180], [165, 180]])
    draw_ink(gray, [[198, 100], [200, 100], [200, 200], [198, 200]])  # 縦に貫く筆界線
    ocr_box = [[170, 130], [230, 130], [230, 170], [170, 170]]
    result = np.array(tight_box(ocr_box, quad_frame(ocr_box), gray, CONFIG["mask"]))
    assert result[:, 1].max() <= 163


def test_tight_box_never_grows_and_keeps_box_without_ink():
    box_quad = rect(200, 150, 80, 40)
    assert tight_box(box_quad, quad_frame(box_quad), blank_gray(), CONFIG["mask"]) == box_quad
    gray = blank_gray()
    draw_ink(gray, [[100, 100], [300, 100], [300, 200], [100, 200]])  # 枠より広いインク
    result = np.array(tight_box(box_quad, quad_frame(box_quad), gray, CONFIG["mask"]))
    assert result[:, 0].min() >= 160 and result[:, 0].max() <= 240


def test_lone_text_is_cut_below_unread_chiban_above_it():
    """読めなかった地番のインクが枠の上側にあれば、谷で切って塗らない（§15.2）。"""
    gray = blank_gray()
    draw_ink(gray, [[190, 132], [210, 132], [210, 140], [190, 140]])  # 読めなかった1桁の地番の下端
    draw_ink(gray, [[170, 146], [230, 146], [230, 170], [170, 170]])  # 氏名
    box_quad = [[160.0, 130.0], [240.0, 130.0], [240.0, 172.0], [160.0, 172.0]]
    result = np.array(refine_lone_text(box_quad, gray, CONFIG["mask"]))
    assert 140 <= result[:, 1].min() <= 146
    assert result[:, 1].max() >= 170


def test_lone_text_without_ink_above_valley_is_only_tightened():
    gray = blank_gray()
    draw_ink(gray, [[170, 146], [230, 146], [230, 170], [170, 170]])
    box_quad = [[160.0, 130.0], [240.0, 130.0], [240.0, 172.0], [160.0, 172.0]]
    result = np.array(refine_lone_text(box_quad, gray, CONFIG["mask"]))
    assert result[:, 1].min() == pytest.approx(143, abs=1.5)


def test_lone_text_keeps_top_when_a_line_fills_every_row():
    """筆界線が縦に貫いて谷がなければ切らない。"""
    gray = blank_gray()
    draw_ink(gray, [[170, 134], [230, 134], [230, 170], [170, 170]])
    box_quad = [[160.0, 130.0], [240.0, 130.0], [240.0, 172.0], [160.0, 172.0]]
    result = np.array(refine_lone_text(box_quad, gray, CONFIG["mask"]))
    assert result[:, 1].min() == pytest.approx(131, abs=1.5)


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


def test_paint_never_touches_protected_boxes_and_splits_into_pieces(tmp_path):
    """地番の枠が塗りの中央にあっても、地番は塗られず、塗りは複数の項目に割れる（§12.2）。"""
    source = tmp_path / "a.tif"
    tifffile.imwrite(source, np.full((100, 200, 3), 200, dtype=np.uint8), photometric="rgb")
    chiban = box(rect(100, 50, 20, 60), "12")
    lone = rect(100, 50, 100, 20)  # 地番の枠を横切る
    out_tif, result = write_outputs(source, tmp_path / "out", [chiban], [], CONFIG["mask"], [lone])
    painted = tifffile.imread(out_tif)
    assert len(result["masks"]) == 2
    assert {m["status"] for m in result["masks"]} == {"unpaired"}
    assert painted[50, 100].tolist() == [200, 200, 200]  # 地番の枠の中
    assert painted[50, 90].tolist() == [200, 200, 200]  # 地番の枠の端
    assert painted[50, 70].tolist() == [0, 0, 0]
    assert painted[50, 130].tolist() == [0, 0, 0]


def test_uncertain_boxes_are_protected_too(tmp_path):
    source = tmp_path / "a.tif"
    tifffile.imwrite(source, np.full((100, 200, 3), 200, dtype=np.uint8), photometric="rgb")
    uncertain = box(rect(100, 50, 20, 60), "273:9")
    _, result = write_outputs(source, tmp_path / "out", [], [], CONFIG["mask"], [rect(100, 50, 100, 20)], [uncertain])
    assert len(result["masks"]) == 2


def test_subtract_removes_hole_by_splitting_and_drops_covered_paint():
    protected = protected_area([rect(100, 50, 20, 10)], 1)
    pieces = subtract_protected(rect(100, 50, 100, 60), protected)
    assert len(pieces) == 2
    area = sum(Polygon(piece).area for piece in pieces)
    assert area == pytest.approx(100 * 60 - 22 * 12)
    assert subtract_protected(rect(100, 50, 10, 4), protected) == []
    untouched = rect(300, 50, 10, 4)
    assert subtract_protected(untouched, protected) == [untouched]


def test_masked_below_is_false_when_all_paint_is_removed(tmp_path):
    source = tmp_path / "a.tif"
    tifffile.imwrite(source, np.full((100, 200, 3), 200, dtype=np.uint8), photometric="rgb")
    chiban = box(rect(100, 40, 60, 60), "12")
    name = box(rect(100, 60, 50, 10))
    pairs = [Pair(chiban=chiban, quad=name["quad"], distance=1.0, source=name)]
    _, result = write_outputs(source, tmp_path / "out", [chiban], pairs, CONFIG["mask"])
    assert result["masks"] == []
    assert result["chiban"][0]["masked_below"] is False


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
MARGIN = CONFIG["evaluate"]["overmask_margin_px"]


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
    scores = evaluate(truth, result, gray_with_ink(chiban, name), THRESHOLD, MARGIN)
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
    scores = evaluate(truth, result, gray_with_ink(chiban, name), THRESHOLD, MARGIN)
    assert scores["name_coverage_pct"] == 50.0
    assert scores["chiban_survival_pct"] == 0.0
    assert scores["chiban_detection_pct"] == 0.0
    assert scores["false_positives"] == 1
    assert 0 < scores["overmask_pct"] < 100


def test_overmask_allows_margin_around_name_but_keeps_raw_value():
    name = [[30, 45], [70, 45], [70, 55], [30, 55]]
    with_margin = [[27, 42], [73, 42], [73, 58], [27, 58]]
    truth = {"items": [truth_item(rect(50, 30, 20, 10), name)]}
    result = {"chiban": [], "masks": [{"quad": with_margin, "chiban": "1", "status": "auto"}]}
    scores = evaluate(truth, result, gray_with_ink(name), THRESHOLD, MARGIN)
    assert scores["overmask_pct"] == 0.0
    assert scores["overmask_raw_pct"] > 40
    beyond = [[10, 42], [73, 42], [73, 58], [10, 58]]
    result["masks"][0]["quad"] = beyond
    assert evaluate(truth, result, gray_with_ink(name), THRESHOLD, MARGIN)["overmask_pct"] > 0


def test_evaluate_survival_ignores_paint_on_blank_margin_of_chiban_box():
    """正解の枠の余白に塗りが触れただけでは、地番は生きている（§11.5）。"""
    chiban_box = [[40, 20], [60, 20], [60, 40], [40, 40]]
    ink_only = [[44, 22], [56, 22], [56, 32], [44, 32]]
    touching_margin = [[30, 36], [70, 36], [70, 46], [30, 46]]
    truth = {"items": [truth_item(chiban_box, rect(50, 41, 40, 10))]}
    result = {"chiban": [], "masks": [{"quad": touching_margin, "chiban": None, "status": "unpaired"}]}
    assert evaluate(truth, result, gray_with_ink(ink_only), THRESHOLD, MARGIN)["chiban_survival_pct"] == 100.0
    touching_ink = [[30, 30], [70, 30], [70, 46], [30, 46]]
    result["masks"][0]["quad"] = touching_ink
    assert evaluate(truth, result, gray_with_ink(ink_only), THRESHOLD, MARGIN)["chiban_survival_pct"] == 0.0


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
    scores = evaluate(truth, result, gray_with_ink(chiban, name), THRESHOLD, MARGIN)
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
    scores = evaluate(truth, result, gray_with_ink(clear_chiban, hard_chiban), THRESHOLD, MARGIN)
    assert scores["name_coverage_pct"] == 50.0
    assert scores["excluding_line_overlaps"]["name_coverage_pct"] == 100.0
    assert scores["excluding_line_overlaps"]["truth_items"] == 1


def test_evaluate_reports_title_block_scores():
    name, chiban = rect(30, 40, 40, 10), rect(30, 20, 20, 10)
    author, sheet_no = rect(90, 60, 40, 10), rect(90, 80, 40, 10)
    truth = {
        "items": [truth_item(chiban, name)],
        "title": [{"kind": "name", "quad": author}, {"kind": "meta", "quad": sheet_no}],
    }
    result = {
        "chiban": [],
        "title_meta": [{"text": "J1", "quad": sheet_no, "score": 1.0, "status": "title_meta"}],
        "masks": [{"quad": name, "chiban": None, "status": "auto"}, {"quad": author, "chiban": None, "status": "in_title_block"}],
    }
    scores = evaluate(truth, result, gray_with_ink(chiban, sheet_no), THRESHOLD, MARGIN)
    assert scores["title_name_coverage_pct"] == 100.0
    assert scores["title_meta_survival_pct"] == 100.0
    assert scores["title_meta_kept"] == 1
    assert scores["overmask_pct"] == 0.0  # 表題欄の作成者名を塗っても、塗りすぎではない


def test_evaluate_reports_scores_by_placement():
    flat_chiban, flat_name = rect(30, 20, 20, 10), rect(30, 40, 40, 10)
    turned_chiban, turned_name = rect(90, 20, 20, 10), rect(90, 40, 40, 10)
    truth = {
        "items": [
            {**truth_item(flat_chiban, flat_name), "placement": "horizontal"},
            {**truth_item(turned_chiban, turned_name), "placement": "rotated"},
        ]
    }
    result = {"chiban": [], "masks": [{"quad": flat_name, "chiban": None, "status": "unpaired"}]}
    scores = evaluate(truth, result, gray_with_ink(flat_chiban), THRESHOLD, MARGIN)
    assert scores["by_placement"]["horizontal"]["name_coverage_pct"] == 100.0
    assert scores["by_placement"]["rotated"]["name_coverage_pct"] == 0.0
    assert scores["by_placement"]["rotated"]["truth_items"] == 1
    assert "leader" not in scores["by_placement"]


def test_average_scores_skips_rows_without_the_key():
    rows = [{"by": {"rotated": {"a": 10.0}}}, {"by": {"leader": {"a": 2.0}}}, {"by": {"rotated": {"a": 20.0}}}]
    assert average_scores(rows) == {"by": {"rotated": {"a": 15.0}, "leader": {"a": 2.0}}}


def test_average_scores_handles_nested_values():
    rows = [{"a": 1.0, "b": {"c": 2.0}}, {"a": 3.0, "b": {"c": 4.0}}]
    assert average_scores(rows) == {"a": 2.0, "b": {"c": 3.0}}


# ocr（エンジンなし）


def test_scrub_drops_names_but_keeps_numbers_and_title_words():
    keep = CONFIG["ocr"]["keep_char_class"], CONFIG["title_keep_words"]
    assert ocr_module.scrub_text("架空 太郎", *keep) == ""
    assert ocr_module.scrub_text("273-4", *keep) == "273-4"
    assert ocr_module.scrub_text("S=1:500", *keep) == "S=1:500"
    assert ocr_module.scrub_text("273ノ9", *keep) == "273ノ9"
    # 数字を含めば、数字の位置が分かる形で残し、許可外の文字は保存しない（§18.1）
    assert ocr_module.scrub_text("273田", *keep) == "273□"
    assert ocr_module.scrub_text("3田中", *keep) == "3□□"
    assert ocr_module.scrub_text("作成 R6.4.1", *keep) == "□□ R6.4.1"
    # 図名の定型語だけは文字のまま残す。ほかの文字（氏名など）は残さない
    assert ocr_module.scrub_text("現況平面図", *keep) == "現況平面図"
    assert ocr_module.scrub_text("架空町現況平面図", *keep) == "□□□現況平面図"
    assert ocr_module.scrub_text("作成者 架空 太郎", *keep) == ""


def test_cache_name_changes_when_scrub_rule_changes(tmp_path):
    image = tmp_path / "a.tif"
    tifffile.imwrite(image, np.zeros((4, 4, 3), dtype=np.uint8), photometric="rgb")
    base = {**CONFIG["ocr"], "title_keep_words": ["平面図"]}
    other = {**base, "title_keep_words": ["平面図", "台帳"]}
    assert ocr_module.cache_file(image, base, tmp_path) != ocr_module.cache_file(image, other, tmp_path)
    assert ocr_module.cache_file(image, base, tmp_path) == ocr_module.cache_file(image, dict(base), tmp_path)


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


def steep_image():
    """縦に読む文字の代わりに、上下で明るさの違う帯を描く。向きが分かるようにする。"""
    image = np.full((300, 300, 3), 240, dtype=np.uint8)
    image[100:150, 145:155] = 20
    image[150:200, 145:155] = 120
    return image


def test_rerecognize_expands_tries_both_directions_and_takes_better_score(monkeypatch):
    item = {"text": "63-3", "score": 0.99, "quad": rect(150, 150, 100, 20, 90)}
    flat = {"text": "12", "score": 0.8, "quad": rect(50, 50, 100, 20, 0)}
    seen = []

    def fake_recognize(crops, _cfg):
        seen.extend(crops)
        return [("□", 0.2), ("263-3", 0.5)]

    monkeypatch.setattr(ocr_module, "recognize_crops", fake_recognize)
    result = ocr_module.rerecognize_steep([item, flat], steep_image(), CONFIG["ocr"])
    assert len(seen) == 2  # 急な枠だけ2通り
    assert seen[0].shape[:2] == (20, 120)  # 長辺方向に両側10%広げる
    assert result[0]["text"] == "263-3"
    assert result[0]["score"] == pytest.approx(0.5 * CONFIG["ocr"]["rerecognize"]["score_factor"])
    assert result[0]["quad"] == rect(150, 150, 100, 20, 90)  # 四角形は変えない
    assert result[1] == flat


def test_rerecognize_crops_are_opposite_directions():
    frame = quad_frame(rect(150, 150, 100, 20, 90))
    forward = ocr_module.crop_along_text(steep_image(), frame, 0.0, False)
    backward = ocr_module.crop_along_text(steep_image(), frame, 0.0, True)
    assert forward[10, 25].mean() != pytest.approx(forward[10, 75].mean(), abs=30)
    assert forward[10, 25].mean() == pytest.approx(backward[10, 75].mean(), abs=5)


def test_rerecognize_never_keeps_name_text(monkeypatch):
    item = {"text": "", "score": 0.9, "quad": rect(150, 150, 100, 20, 80)}
    monkeypatch.setattr(ocr_module, "recognize_crops", lambda _crops, _cfg: [("架空 太郎", 0.9), ("架空", 0.1)])
    result = ocr_module.rerecognize_steep([item], steep_image(), CONFIG["ocr"])
    assert result[0]["text"] == ""


def test_rerecognize_skips_square_boxes(monkeypatch):
    """1桁の地番の縦長の枠は、水平でも 90 度に見えるので対象にしない。"""
    item = {"text": "5", "score": 0.9, "quad": rect(150, 150, 16, 21)}
    monkeypatch.setattr(ocr_module, "recognize_crops", lambda *_args: pytest.fail("呼ばれないはず"))
    assert ocr_module.rerecognize_steep([item], steep_image(), CONFIG["ocr"]) == [item]


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


# pipeline（OCR エンジンなし）


def test_mask_file_labels_each_kind_of_paint(tmp_path, monkeypatch):
    from kurokofude import pipeline

    source = tmp_path / "a.tif"
    tifffile.imwrite(source, np.full((700, 1000, 3), 240, dtype=np.uint8), photometric="rgb")
    boxes = [
        box(rect(300, 300, 60, 20), "12-3"),
        box(rect(300, 326, 90, 18)),  # 地番の下の氏名
        box(rect(600, 200, 90, 18)),  # 地番と離れた文字
        box(rect(900, 620, 90, 18)),  # 表題欄の文字
        box(rect(900, 650, 60, 18), "2024"),  # 表題欄の数字
    ]
    monkeypatch.setattr(pipeline, "run_ocr", lambda *_args: boxes)
    summary = pipeline.mask_file(source, CONFIG, tmp_path / "out", tmp_path / "cache")
    result = json.loads((tmp_path / "out" / "a.chiban.json").read_text(encoding="utf-8"))
    assert sorted(m["status"] for m in result["masks"]) == ["auto", "in_title_block", "unpaired"]
    assert [u["text"] for u in result["title_meta"]] == ["2024"]
    assert result["title_meta"][0]["status"] == "title_meta"
    assert result["uncertain"] == []
    assert summary.masks == 3
    assert summary.chiban == 1


# mixed


def mixed_gray():
    """左に氏名（幅80）、広い隙間、右に地番（幅80）のインク。"""
    gray = np.full((300, 500), 240, dtype=np.uint8)
    cv2.fillPoly(gray, [np.array([[110, 150], [190, 150], [190, 170], [110, 170]], dtype=np.int32)], 20)
    cv2.fillPoly(gray, [np.array([[215, 150], [295, 150], [295, 170], [215, 170]], dtype=np.int32)], 20)
    return gray


def mixed_box(text, in_region=False):
    return {
        "text": text,
        "score": 0.9,
        "quad": [[100.0, 145.0], [305.0, 145.0], [305.0, 175.0], [100.0, 175.0]],
        "in_region": in_region,
    }


def test_mixed_is_recognized_when_box_has_placeholder_and_digits():
    both = box(rect(100, 100, 60, 20), "□□□273-10")
    only_marks = box(rect(100, 200, 60, 20), "□□")
    result = classify_boxes([both, only_marks], IMAGE_SIZE, CONFIG)
    assert [b["text"] for b in result.mixed] == ["□□□273-10"]
    assert only_marks not in result.mixed
    assert only_marks["text"] in [b["text"] for b in result.text + result.uncertain]


def test_mixed_splits_name_side_from_chiban_at_ink_valley():
    frame_of = quad_frame
    result = resolve_mixed([mixed_box("□□□273-10")], mixed_gray(), CONFIG, frame_of)
    assert result.masked == []
    chiban = result.chiban[0]
    assert chiban["text"] == "273-10"
    assert chiban["status"] == "split"
    assert chiban["score"] == pytest.approx(0.9 * CONFIG["chiban"]["normalized_score_factor"])
    assert np.array(chiban["quad"])[:, 0].min() == pytest.approx(202.5, abs=8)
    assert np.array(result.split_masks[0])[:, 0].max() == pytest.approx(202.5, abs=8)


def test_mixed_split_does_not_fall_into_gap_between_family_and_given_name():
    """姓と名の隙間ではなく、氏名と地番の隙間で分ける。名の字が地番側に残らない。"""
    gray = np.full((300, 500), 240, dtype=np.uint8)
    for x0, x1 in ((110, 145), (165, 185), (210, 300)):  # 姓、名、地番
        cv2.fillPoly(gray, [np.array([[x0, 150], [x1, 150], [x1, 170], [x0, 170]], dtype=np.int32)], 20)
    item = mixed_box("□□□273-10")
    item["quad"] = [[100.0, 145.0], [310.0, 145.0], [310.0, 175.0], [100.0, 175.0]]
    result = resolve_mixed([item], gray, CONFIG, quad_frame)
    assert 185 <= np.array(result.chiban[0]["quad"])[:, 0].min() <= 210


def test_mixed_ignores_narrow_gaps_between_digits():
    gray = np.full((300, 500), 240, dtype=np.uint8)
    for x0 in range(110, 300, 22):  # 文字の間が2pxしかない
        cv2.fillPoly(gray, [np.array([[x0, 150], [x0 + 20, 150], [x0 + 20, 170], [x0, 170]], dtype=np.int32)], 20)
    item = mixed_box("□□□273-10")
    result = resolve_mixed([item], gray, CONFIG, quad_frame)
    assert result.masked == [item["quad"]]


def test_mixed_with_digits_first_keeps_left_side():
    result = resolve_mixed([mixed_box("273-10□□□")], mixed_gray(), CONFIG, quad_frame)
    assert np.array(result.chiban[0]["quad"])[:, 0].max() < 215
    assert np.array(result.split_masks[0])[:, 0].min() > 190


@pytest.mark.parametrize(
    "text",
    ["□□2024-3-15", "□12□34", "□□□123456"],
)
def test_mixed_is_masked_whole_when_digit_side_is_not_chiban(text):
    item = mixed_box(text)
    result = resolve_mixed([item], mixed_gray(), CONFIG, quad_frame)
    assert result.chiban == [] and result.split_masks == []
    assert result.masked == [item["quad"]]


def test_mixed_is_masked_whole_without_valley():
    gray = np.full((300, 500), 240, dtype=np.uint8)
    cv2.fillPoly(gray, [np.array([[100, 150], [305, 150], [305, 170], [100, 170]], dtype=np.int32)], 20)
    item = mixed_box("□□□273-10")
    result = resolve_mixed([item], gray, CONFIG, quad_frame)
    assert result.masked == [item["quad"]]


def test_mixed_in_exclude_region_is_masked_whole():
    item = mixed_box("□□□273-10", in_region=True)
    result = resolve_mixed([item], mixed_gray(), CONFIG, quad_frame)
    assert result.masked == [item["quad"]] and result.chiban == []


def test_mask_file_splits_mixed_box_and_paints_only_name_side(tmp_path, monkeypatch):
    from kurokofude import pipeline

    source = tmp_path / "a.tif"
    gray = mixed_gray()
    tifffile.imwrite(source, np.stack([gray] * 3, axis=-1), photometric="rgb")
    item = {k: v for k, v in mixed_box("□□□273-10").items() if k != "in_region"}
    monkeypatch.setattr(pipeline, "run_ocr", lambda *_args: [item])
    pipeline.mask_file(source, CONFIG, tmp_path / "out", tmp_path / "cache")
    result = json.loads((tmp_path / "out" / "a.chiban.json").read_text(encoding="utf-8"))
    assert [c["status"] for c in result["chiban"]] == ["split"]
    assert [c["text"] for c in result["chiban"]] == ["273-10"]
    assert [m["status"] for m in result["masks"]] == ["split"]
    painted = tifffile.imread(tmp_path / "out" / "a.tif")
    assert painted[160, 150].tolist() == [0, 0, 0]  # 氏名側
    assert painted[160, 250].tolist() == [20, 20, 20]  # 地番側


# batch


def confirmed_project(tmp_path, monkeypatch):
    """確定済みの結果が out にある状態を作る。OCR は呼ばない。"""
    from kurokofude import pipeline

    source = tmp_path / "a.tif"
    tifffile.imwrite(source, np.full((700, 1000, 3), 240, dtype=np.uint8), photometric="rgb")
    monkeypatch.setattr(pipeline, "run_ocr", lambda *_args: [box(rect(300, 300, 60, 20), "12-3"), box(rect(300, 326, 90, 18))])
    out = tmp_path / "out"
    pipeline.mask_file(source, CONFIG, out, tmp_path / "cache")
    result_path = out / "a.chiban.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    result["confirmed"] = True
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return source, out, result_path


def test_mask_refuses_confirmed_drawing_without_force(tmp_path, monkeypatch):
    from kurokofude import pipeline

    source, out, result_path = confirmed_project(tmp_path, monkeypatch)
    monkeypatch.setattr(pipeline, "run_ocr", lambda *_args: pytest.fail("確定済みなら OCR の前に止まるはず"))
    with pytest.raises(pipeline.ConfirmedError):
        pipeline.mask_file(source, CONFIG, out, tmp_path / "cache")
    assert json.loads(result_path.read_text(encoding="utf-8"))["confirmed"] is True


def test_mask_overwrites_confirmed_drawing_with_force(tmp_path, monkeypatch):
    from kurokofude import pipeline

    source, out, result_path = confirmed_project(tmp_path, monkeypatch)
    pipeline.mask_file(source, CONFIG, out, tmp_path / "cache", force=True)
    assert "confirmed" not in json.loads(result_path.read_text(encoding="utf-8"))


def test_cli_mask_stops_with_message_and_force_goes_through(tmp_path, monkeypatch, capsys):
    from kurokofude import cli

    source, out, result_path = confirmed_project(tmp_path, monkeypatch)
    arguments = ["mask", str(source), "--out", str(out), "--cache", str(tmp_path / "cache")]
    assert cli.main(arguments) == 1
    assert "--force" in capsys.readouterr().err
    assert cli.main([*arguments, "--force"]) == 0
    assert "confirmed" not in json.loads(result_path.read_text(encoding="utf-8"))


def test_batch_skips_confirmed_drawing_and_succeeds(tmp_path, monkeypatch, capsys):
    from kurokofude import cli

    source, out, result_path = confirmed_project(tmp_path, monkeypatch)
    assert cli.main(["batch", str(tmp_path), "--out", str(out), "--cache", str(tmp_path / "cache")]) == 0
    assert "飛ばした 1 枚" in capsys.readouterr().out
    assert json.loads(result_path.read_text(encoding="utf-8"))["confirmed"] is True


def test_batch_continues_after_failure_and_reports_it(tmp_path, monkeypatch, capsys):
    from kurokofude import cli
    from kurokofude.pipeline import MaskSummary

    for name in ("a.tif", "b.tif", "c.tiff", "note.txt"):
        (tmp_path / name).write_bytes(b"")
    seen = []

    def fake_mask_file(path, _config, out_dir, _cache, force=False):
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
    assert evaluate(truth, result, gray, THRESHOLD, MARGIN)["chiban_survival_pct"] >= FIXTURE_SURVIVAL_FLOOR
