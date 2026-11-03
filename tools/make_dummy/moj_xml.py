"""法務省 登記所備付地図XML を読み、筆と図郭を取り出して GeoJSON にする。

座標の扱い：XML の zmn:X は北方向、zmn:Y は東方向（測量の座標系）。
ここから先は GIS の慣習に合わせて (east, north) = (Y, X) の順で持つ。

自治体名・大字名は読み捨てる（図面と正解データに地名を残さないため）。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from shapely.geometry import LineString, Polygon, mapping
from shapely.ops import polygonize

HEADER_BYTES = 4096
COORD_SYSTEM_RE = re.compile("<座標系>([^<]*)</座標系>")

Point = tuple[float, float]


@dataclass
class Parcel:
    chiban: str
    polygon: Polygon


@dataclass
class Sheet:
    number: str
    scale_denominator: int
    corners: dict[str, Point]
    parcel_ids: list[str] = field(default_factory=list)


@dataclass
class MapData:
    coordinate_system: str
    parcels: dict[str, Parcel]
    sheets: list[Sheet]


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _read_position(element: ET.Element) -> Point | None:
    values = {_local(child.tag): child.text for child in element}
    if "X" in values and "Y" in values:
        return float(values["Y"]), float(values["X"])
    return None


def read_xml_bytes(zip_path: Path) -> bytes:
    with zipfile.ZipFile(zip_path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".xml")]
        if len(names) != 1:
            raise ValueError(f"XMLが1つではありません: {zip_path.name}")
        return archive.read(names[0])


def read_coordinate_system(xml_bytes: bytes) -> str | None:
    """全体をパースせず、先頭だけ見て座標系を返す。"""
    match = COORD_SYSTEM_RE.search(xml_bytes[:HEADER_BYTES].decode("utf-8", errors="ignore"))
    return match.group(1) if match else None


def parse_map(xml_bytes: bytes) -> MapData:
    root = ET.fromstring(xml_bytes)
    points: dict[str, Point] = {}
    curves: dict[str, list[Point]] = {}
    surface_rings: dict[str, list[str]] = {}
    coordinate_system = ""
    raw_parcels: list[tuple[str, str, str]] = []
    sheets: list[Sheet] = []

    for element in root:
        name = _local(element.tag)
        if name == "座標系":
            coordinate_system = element.text or ""
        elif name == "空間属性":
            for shape in element:
                kind = _local(shape.tag)
                if kind == "GM_Point":
                    position = next(
                        (_read_position(e) for e in shape.iter() if _local(e.tag) == "DirectPosition"),
                        None,
                    )
                    if position:
                        points[shape.attrib["id"]] = position
                elif kind == "GM_Curve":
                    curves[shape.attrib["id"]] = _read_curve(shape, points)
                elif kind == "GM_Surface":
                    exterior = next(
                        (e for e in shape.iter() if _local(e.tag) == "GM_SurfaceBoundary.exterior"),
                        None,
                    )
                    if exterior is not None:
                        surface_rings[shape.attrib["id"]] = [
                            e.attrib["idref"]
                            for e in exterior.iter()
                            if _local(e.tag) == "GM_CompositeCurve.generator"
                        ]
        elif name == "主題属性":
            for item in element:
                if _local(item.tag) != "筆":
                    continue
                fields = {_local(c.tag): c for c in item}
                if "地番" in fields and "形状" in fields:
                    raw_parcels.append(
                        (item.attrib["id"], fields["地番"].text or "", fields["形状"].attrib["idref"])
                    )
        elif name == "図郭":
            sheets.append(_parse_sheet(element))

    parcels: dict[str, Parcel] = {}
    for parcel_id, chiban, surface_id in raw_parcels:
        polygon = _build_polygon(surface_rings.get(surface_id, []), curves)
        if polygon is not None:
            parcels[parcel_id] = Parcel(chiban=chiban, polygon=polygon)
    return MapData(coordinate_system=coordinate_system, parcels=parcels, sheets=sheets)


def _read_curve(curve: ET.Element, points: dict[str, Point]) -> list[Point]:
    """曲線の頂点は、座標の直書き（direct）と点の参照（indirect）が混在する。"""
    result: list[Point] = []
    for element in curve.iter():
        name = _local(element.tag)
        if name == "GM_Position.direct":
            position = _read_position(element)
            if position:
                result.append(position)
        elif name == "GM_PointRef.point" and element.attrib.get("idref") in points:
            result.append(points[element.attrib["idref"]])
    return result


def _parse_sheet(element: ET.Element) -> Sheet:
    fields = {_local(c.tag): c for c in element}
    corners = {}
    for key, label in (("ll", "左下座標"), ("ul", "左上座標"), ("lr", "右下座標"), ("ur", "右上座標")):
        position = _read_position(fields[label])
        if position is None:
            raise ValueError(f"図郭の{label}が読めません")
        corners[key] = position
    return Sheet(
        number=" ".join((fields["地図番号"].text or "").split()),
        scale_denominator=int(fields["縮尺分母"].text or 0),
        corners=corners,
        parcel_ids=[c.attrib["idref"] for c in element if _local(c.tag) == "筆参照"],
    )


def _build_polygon(curve_ids: list[str], curves: dict[str, list[Point]]) -> Polygon | None:
    lines = [LineString(curves[c]) for c in curve_ids if len(curves.get(c, [])) >= 2]
    polygons = list(polygonize(lines))
    if not polygons:
        return None
    return max(polygons, key=lambda p: p.area)


def sheet_to_geojson(sheet: Sheet, data: MapData, chiban_filter=None) -> dict:
    """図郭に属する筆を GeoJSON にする。座標は (east, north)。"""
    features = []
    for parcel_id in sheet.parcel_ids:
        parcel = data.parcels.get(parcel_id)
        if parcel is None or (chiban_filter and not chiban_filter(parcel.chiban)):
            continue
        features.append(
            {"type": "Feature", "geometry": mapping(parcel.polygon), "properties": {"chiban": parcel.chiban}}
        )
    return {
        "type": "FeatureCollection",
        "properties": {
            "coordinate_system": data.coordinate_system,
            "sheet": sheet.number,
            "scale_denominator": sheet.scale_denominator,
        },
        "features": features,
    }
