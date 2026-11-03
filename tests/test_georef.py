"""④ 照合（tools/georef）の単体テスト。画像も OCR も使わない。"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "georef"))

import georef  # noqa: E402

CONFIG = {**georef.load_config(), "max_iterations": 0}
PIXEL_M = 0.0635
ROTATION_DEG = 1.5
ORIGIN = (-61480.0, -143640.0)  # 左上の画素の公共座標（東, 北）
IMAGE_SIZE = [1000, 800]


def truth_tfw():
    theta = math.radians(ROTATION_DEG)
    a, b = PIXEL_M * math.cos(theta), PIXEL_M * math.sin(theta)
    return [a, b, b, -a, ORIGIN[0], ORIGIN[1]]


def to_world(tfw, col, row):
    a, d, b, e, c, f = tfw
    return a * col + b * row + c, d * col + e * row + f


def to_pixel(tfw, east, north):
    a, d, b, e, c, f = tfw
    matrix = np.array([[a, b], [d, e]])
    return np.linalg.solve(matrix, np.array([east - c, north - f]))


def square_feature(text, east, north, half):
    ring = [[east - half, north - half], [east + half, north - half], [east + half, north + half], [east - half, north + half], [east - half, north - half]]
    return {"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {"chiban": text}}


def scene(labels=None, extra_features=()):
    """格子状の筆と、筆の重心に書かれた地番。labels で読みを差し替えられる。"""
    tfw = truth_tfw()
    features, chiban = [], []
    labels = labels or {}
    index = 0
    for col in range(100, 1000, 200):
        for row in range(100, 800, 200):
            text = f"{100 + index}-{index % 3 + 1}"
            east, north = to_world(tfw, col, row)
            features.append(square_feature(text, east, north, 5.0))
            chiban.append({"text": labels.get(text, text), "center": [float(col), float(row)], "score": 0.99})
            index += 1
    features.extend(extra_features)
    return {"chiban": chiban, "image_size": IMAGE_SIZE}, {"type": "FeatureCollection", "features": features}


def test_recovers_world_file_from_chiban_alone():
    chiban_json, geojson = scene()
    report = georef.georeference(chiban_json, geojson, CONFIG)
    assert report["status"] == "ok"
    assert report["world_file"] == pytest.approx(truth_tfw(), abs=1e-6)
    assert report["residual_rms_m"] == pytest.approx(0, abs=1e-6)
    assert report["pixel_size_m"] == pytest.approx(PIXEL_M)
    assert report["rotation_deg"] == pytest.approx(ROTATION_DEG)
    assert report["pairs_used"] == report["pairs_found"] == 20


def test_world_file_maps_pixels_like_gis_does():
    """.tfw の並びは A, D, B, E, C, F。画像の行が下へ増えると北が減る。"""
    chiban_json, geojson = scene()
    tfw = georef.georeference(chiban_json, geojson, CONFIG)["world_file"]
    top = to_world(tfw, 500, 0)
    bottom = to_world(tfw, 500, 799)
    assert bottom[1] < top[1]


def test_mirrored_fit_is_much_worse():
    """向きの確認：行を反転しない（鏡映）変換では、ほとんどの組が一致しない。"""
    chiban_json, geojson = scene()
    report = georef.georeference(chiban_json, geojson, CONFIG)
    assert report["mirrored_pairs_used"] < report["pairs_used"] / 2


def test_hyphen_dropped_chiban_is_reread_with_lower_score():
    chiban_json, geojson = scene(labels={"103-1": "1031"})
    report = georef.georeference(chiban_json, geojson, CONFIG)
    assert report["rereadings"] == [{"read_as": "1031", "matched": "103-1"}]
    reread = next(r for r in report["residuals"] if r["read_as"] == "1031")
    assert reread["score"] == pytest.approx(0.99 * CONFIG["reread_score_factor"])


def test_hyphen_candidates():
    assert georef.hyphen_candidates("2739") == ["2-739", "27-39", "273-9"]
    assert georef.hyphen_candidates("273-9") == []
    assert georef.hyphen_candidates("7") == []


def test_duplicates_are_skipped_first_then_chosen_by_predicted_position():
    """重複した地番は1回目の解では使わず、2回目以降は予測した位置に近い筆を採る（§18.3）。"""
    tfw = truth_tfw()
    far_east, far_north = to_world(tfw, 900, 700)
    duplicate = square_feature("101-2", far_east + 30, far_north - 30, 5.0)  # 公図の別の場所にも 101-2
    chiban_json, geojson = scene(extra_features=[duplicate])
    first = georef.georeference(chiban_json, geojson, CONFIG)
    assert "101-2" not in {r["chiban"] for r in first["residuals"]}
    assert {"text": "101-2", "reason": "重複（1回目の解では使わない）"} in first["unmatched"]

    second = georef.georeference(chiban_json, geojson, {**CONFIG, "max_iterations": 1})
    assert "101-2" in second["chosen_by_prediction"]
    assert next(r for r in second["residuals"] if r["chiban"] == "101-2")["residual_m"] == pytest.approx(0, abs=1e-6)


def test_wrong_pair_is_excluded_as_outlier():
    chiban_json, geojson = scene()
    chiban_json["chiban"][0]["center"] = [900.0, 700.0]  # 読み違いで別の場所の地番になった
    report = georef.georeference(chiban_json, geojson, CONFIG)
    assert report["excluded"] == [chiban_json["chiban"][0]["text"]]
    assert report["world_file"] == pytest.approx(truth_tfw(), abs=1e-6)


def test_too_few_pairs_fail():
    chiban_json, geojson = scene()
    chiban_json["chiban"] = chiban_json["chiban"][:2]
    assert georef.georeference(chiban_json, geojson, CONFIG)["status"] == "failed"


def test_footprint_refinement_uses_visible_part_of_edge_parcels():
    """図郭の端で切れた筆は、見えている部分に地番が書かれる。見えている部分の重心で解き直すと合う。"""
    tfw = truth_tfw()
    chiban_json, geojson = scene()
    for row in (100, 300, 500, 700):  # 左端の外まで続く細長い筆と、その見えている部分の中心に書いた地番
        text = f"9{row}"
        east, north = to_world(tfw, 20, row)
        long_ring = [[east - 60, north - 3], [east + 1.27, north - 3], [east + 1.27, north + 3], [east - 60, north + 3], [east - 60, north - 3]]
        geojson["features"].append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [long_ring]}, "properties": {"chiban": text}})
        chiban_json["chiban"].append({"text": text, "center": [10.0, float(row)], "score": 0.99})
    loose = {**CONFIG, "ransac_tolerance_m": 50.0}
    plain = georef.georeference(chiban_json, geojson, loose)
    refined = georef.georeference(chiban_json, geojson, {**loose, "max_iterations": 3})
    assert refined["residual_rms_m"] < plain["residual_rms_m"] / 3
    assert [step["clip"] for step in refined["iterations"]][:2] == ["none", "image"]


def test_frame_quad_is_used_after_the_image_clip():
    """図郭線が画像より内側なら、2回目以降は図郭線の範囲で切る。図郭線の外まで続く筆で効く。"""
    tfw = truth_tfw()
    chiban_json, geojson = scene()
    frame = [[50, 50], [950, 50], [950, 750], [50, 750]]
    for row in (150, 350, 550):  # 図郭線の左の外まで続く筆。地番は図郭線の内側の見えている部分の中心
        text = f"8{row}"
        east, north = to_world(tfw, 60, row)
        ring = [[east - 40, north - 3], [east + 1.27, north - 3], [east + 1.27, north + 3], [east - 40, north + 3], [east - 40, north - 3]]
        geojson["features"].append({"type": "Feature", "geometry": {"type": "Polygon", "coordinates": [ring]}, "properties": {"chiban": text}})
        chiban_json["chiban"].append({"text": text, "center": [65.0, float(row)], "score": 0.99})  # 列 50〜80 の中心
    loose = {**CONFIG, "ransac_tolerance_m": 50.0, "max_iterations": 4}
    image_only = georef.georeference(chiban_json, geojson, loose)
    with_frame = georef.georeference(chiban_json, geojson, loose, frame)
    assert with_frame["used_frame"] is True
    assert "frame" in [step["clip"] for step in with_frame["iterations"]]
    # 図郭線で切ると、筆全体の重心で解いた1回目より合う（反復は図郭線を今の解で写すので、真の縮尺に
    # 必ず一致するわけではない。実データでの収束は accuracy_log の「照合」）
    assert with_frame["residual_rms_m"] < with_frame["iterations"][0]["residual_rms_m"] / 5
    assert image_only["used_frame"] is False
