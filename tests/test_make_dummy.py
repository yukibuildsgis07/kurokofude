"""ダミー図面生成の単体テスト。元データ（zip）は使わず、小さな XML で確かめる。"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "make_dummy"))

import moj_xml  # noqa: E402
import render  # noqa: E402

XML = """<?xml version="1.0" encoding="UTF-8"?>
<地図 xmlns="http://www.moj.go.jp/MINJI/tizuxml" xmlns:zmn="http://www.moj.go.jp/MINJI/tizuzumen">
 <地図名>テスト</地図名>
 <座標系>公共座標8系</座標系>
 <空間属性>
  <zmn:GM_Point id="P1"><zmn:GM_Point.position><zmn:DirectPosition><zmn:X>100.0</zmn:X><zmn:Y>10.0</zmn:Y></zmn:DirectPosition></zmn:GM_Point.position></zmn:GM_Point>
  <zmn:GM_Curve id="C1"><zmn:GM_Curve.segment><zmn:GM_LineString><zmn:GM_LineString.controlPoint>
    <zmn:GM_PointArray.column><zmn:GM_Position.indirect><zmn:GM_PointRef.point idref="P1"/></zmn:GM_Position.indirect></zmn:GM_PointArray.column>
    <zmn:GM_PointArray.column><zmn:GM_Position.direct><zmn:X>100.0</zmn:X><zmn:Y>20.0</zmn:Y></zmn:GM_Position.direct></zmn:GM_PointArray.column>
    <zmn:GM_PointArray.column><zmn:GM_Position.direct><zmn:X>90.0</zmn:X><zmn:Y>20.0</zmn:Y></zmn:GM_Position.direct></zmn:GM_PointArray.column>
  </zmn:GM_LineString.controlPoint></zmn:GM_LineString></zmn:GM_Curve.segment></zmn:GM_Curve>
  <zmn:GM_Curve id="C2"><zmn:GM_Curve.segment><zmn:GM_LineString><zmn:GM_LineString.controlPoint>
    <zmn:GM_PointArray.column><zmn:GM_Position.direct><zmn:X>90.0</zmn:X><zmn:Y>10.0</zmn:Y></zmn:GM_Position.direct></zmn:GM_PointArray.column>
    <zmn:GM_PointArray.column><zmn:GM_Position.direct><zmn:X>90.0</zmn:X><zmn:Y>20.0</zmn:Y></zmn:GM_Position.direct></zmn:GM_PointArray.column>
  </zmn:GM_LineString.controlPoint></zmn:GM_LineString></zmn:GM_Curve.segment></zmn:GM_Curve>
  <zmn:GM_Curve id="C3"><zmn:GM_Curve.segment><zmn:GM_LineString><zmn:GM_LineString.controlPoint>
    <zmn:GM_PointArray.column><zmn:GM_Position.direct><zmn:X>90.0</zmn:X><zmn:Y>10.0</zmn:Y></zmn:GM_Position.direct></zmn:GM_PointArray.column>
    <zmn:GM_PointArray.column><zmn:GM_Position.indirect><zmn:GM_PointRef.point idref="P1"/></zmn:GM_Position.indirect></zmn:GM_PointArray.column>
  </zmn:GM_LineString.controlPoint></zmn:GM_LineString></zmn:GM_Curve.segment></zmn:GM_Curve>
  <zmn:GM_Surface id="F1"><zmn:GM_Surface.patch><zmn:GM_Polygon><zmn:GM_Polygon.boundary><zmn:GM_SurfaceBoundary><zmn:GM_SurfaceBoundary.exterior><zmn:GM_Ring>
    <zmn:GM_CompositeCurve.generator idref="C1"/><zmn:GM_CompositeCurve.generator idref="C2"/><zmn:GM_CompositeCurve.generator idref="C3"/>
  </zmn:GM_Ring></zmn:GM_SurfaceBoundary.exterior></zmn:GM_SurfaceBoundary></zmn:GM_Polygon.boundary></zmn:GM_Polygon></zmn:GM_Surface.patch></zmn:GM_Surface>
 </空間属性>
 <主題属性>
  <筆 id="H1"><大字名>架空町</大字名><地番>12-3</地番><形状 idref="F1"/></筆>
 </主題属性>
 <図郭>
  <地図番号>K   24-3</地図番号><縮尺分母>500</縮尺分母>
  <左下座標><zmn:X>0.0</zmn:X><zmn:Y>0.0</zmn:Y></左下座標>
  <左上座標><zmn:X>150.0</zmn:X><zmn:Y>0.0</zmn:Y></左上座標>
  <右下座標><zmn:X>0.0</zmn:X><zmn:Y>200.0</zmn:Y></右下座標>
  <右上座標><zmn:X>150.0</zmn:X><zmn:Y>200.0</zmn:Y></右上座標>
  <筆参照 idref="H1"/>
 </図郭>
</地図>
""".encode("utf-8")


def test_coordinate_system_is_read_exactly():
    assert moj_xml.read_coordinate_system(XML) == "公共座標8系"
    assert moj_xml.read_coordinate_system("<座標系>任意座標系</座標系>".encode("utf-8")) == "任意座標系"


def test_parse_map_builds_polygon_with_east_north_order():
    data = moj_xml.parse_map(XML)
    parcel = data.parcels["H1"]
    assert parcel.chiban == "12-3"
    assert parcel.polygon.area == pytest.approx(100.0)
    assert parcel.polygon.bounds == pytest.approx((10.0, 90.0, 20.0, 100.0))
    sheet = data.sheets[0]
    assert sheet.number == "K 24-3"
    assert sheet.corners["ul"] == (0.0, 150.0)


def test_geojson_has_no_place_names():
    data = moj_xml.parse_map(XML)
    geojson = moj_xml.sheet_to_geojson(data.sheets[0], data)
    assert geojson["features"][0]["properties"] == {"chiban": "12-3"}
    assert "架空町" not in json.dumps(geojson, ensure_ascii=False)


def test_transform_size_follows_scale():
    sheet = moj_xml.parse_map(XML).sheets[0]
    matrix, (width, height) = render.sheet_transform(sheet, render.pixels_per_meter(200, 500), margin_px=0)
    assert width == 3150 and height == 2362
    upper_left = render.apply_affine(matrix, np.array([[0.0, 150.0]]))
    assert upper_left[0] == pytest.approx([0, 0], abs=1e-6)


def test_world_file_inverts_transform_with_public_coordinates():
    """公共座標の大きな値でも、全体の傾きを含めて画素 ↔ 座標が往復できる。"""
    src = np.array([[-61082.0, -141095.0], [-60882.0, -141095.0], [-61082.0, -141245.0]])
    dst = np.array([[10.0, 10.0], [3160.0, 10.0], [10.0, 2372.0]])
    world_to_px = render.solve_affine(src, dst)
    skew = np.array([[0.9994, 0.0349, -30.0], [-0.0349, 0.9994, 50.0]])
    full = render.compose(skew, world_to_px)
    a, d, b, e, c, f = render.world_file_from(full)
    px = render.apply_affine(full, src)
    east = a * px[:, 0] + b * px[:, 1] + c
    north = d * px[:, 0] + e * px[:, 1] + f
    assert east == pytest.approx(src[:, 0], abs=1e-6)
    assert north == pytest.approx(src[:, 1], abs=1e-6)


def test_quad_order_and_down_direction():
    quad = render.text_quad(np.array([100.0, 100.0]), 40, 10, 30)
    assert Polygon(quad).area == pytest.approx(400)
    _, down = render.text_vectors(30)
    assert down[1] > 0
    assert ((quad[3] - quad[0]) / 10) == pytest.approx(down)


def test_fixture_truth_has_slanted_labels_and_no_place_names():
    truth = json.loads((ROOT / "tests" / "fixtures" / "sample_small.truth.json").read_text(encoding="utf-8"))
    assert truth["image_size"] == [1000, 700]
    assert any(item["slanted"] for item in truth["items"])
    assert all(len(item["name"]["quad"]) == 4 for item in truth["items"])
    assert all(set(item["name"]) == {"quad"} for item in truth["items"])  # 氏名の文字列を持たない
    assert {"source", "image_size", "sheet", "scale_denominator", "items"} == set(truth)


def test_config_uses_only_plane_coordinate_system_8():
    config = yaml.safe_load((ROOT / "tools" / "make_dummy" / "config.yaml").read_text(encoding="utf-8"))
    assert config["source"]["coordinate_system"] == "公共座標8系"
