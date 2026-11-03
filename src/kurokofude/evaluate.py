"""正解 JSON との採点（設計書 §2.5、§11.5、§11.10）。氏名の文字列は使わない。"""

from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from .mask import STATUS_UNPAIRED, fill_polygons

STATUS_REMOVED = "removed"
PERCENT = 100.0
MIN_POLYGON_VERTICES = 3
PAINTED = [255]


def _polygon(quad) -> Polygon:
    return Polygon(quad).buffer(0)


def _ratio(numerator: float, denominator: float) -> float:
    return PERCENT * numerator / denominator if denominator else 0.0


def _ink_is_painted(quad, ink: np.ndarray, painted: np.ndarray) -> bool:
    """正解の地番枠の中のインク画素が、1つでも塗られていれば True。"""
    region = np.zeros(ink.shape, dtype=np.uint8)
    cv2.fillPoly(region, [np.round(np.asarray(quad, dtype=float)).astype(np.int32)], 255)
    return bool(np.any((region > 0) & ink & (painted > 0)))


def _item_scores(items: list[dict], masks_geometry, detected: list[dict], ink: np.ndarray, painted: np.ndarray) -> dict:
    name_polygons = [_polygon(item["name"]["quad"]) for item in items]
    name_area = sum(p.area for p in name_polygons)
    covered = sum(p.intersection(masks_geometry).area for p in name_polygons)
    survived = sum(1 for item in items if not _ink_is_painted(item["chiban"]["quad"], ink, painted))
    matched = 0
    for item in items:
        polygon = _polygon(item["chiban"]["quad"])
        if any(d["text"] == item["chiban"]["text"] and polygon.contains(Point(d["center"])) for d in detected):
            matched += 1
    return {
        "name_coverage_pct": round(_ratio(covered, name_area), 2),
        "chiban_survival_pct": round(_ratio(survived, len(items)), 2),
        "chiban_detection_pct": round(_ratio(matched, len(items)), 2),
        "truth_items": len(items),
    }


def evaluate(truth: dict, result: dict, gray: np.ndarray, ink_threshold: int) -> dict:
    """gray は塗る前の図面。地番のインクが塗られていないかを画素で確かめるのに使う。"""
    active = [
        m
        for m in result["masks"]
        if m.get("status") != STATUS_REMOVED and len(m["quad"]) >= MIN_POLYGON_VERTICES
    ]
    masks_geometry = unary_union([_polygon(m["quad"]) for m in active])
    painted = fill_polygons(np.zeros(gray.shape, dtype=np.uint8), [m["quad"] for m in active], PAINTED)
    ink = gray < ink_threshold

    items = truth["items"]
    detected = result["chiban"]
    names_union = unary_union([_polygon(item["name"]["quad"]) for item in items])
    chiban_polygons = [_polygon(item["chiban"]["quad"]) for item in items]
    clear_items = [item for item in items if not item.get("overlaps_line")]

    scores = _item_scores(items, masks_geometry, detected, ink, painted)
    return {
        "name_coverage_pct": scores["name_coverage_pct"],
        "overmask_pct": round(_ratio(masks_geometry.difference(names_union).area, masks_geometry.area), 2),
        "chiban_survival_pct": scores["chiban_survival_pct"],
        "chiban_detection_pct": scores["chiban_detection_pct"],
        "false_positives": sum(
            1 for d in detected if not any(p.contains(Point(d["center"])) for p in chiban_polygons)
        ),
        "truth_items": scores["truth_items"],
        "masks": len(result["masks"]),
        "unpaired_masks": sum(1 for m in result["masks"] if m.get("status") == STATUS_UNPAIRED),
        "uncertain_chiban": len(result.get("uncertain", [])),
        "excluding_line_overlaps": _item_scores(clear_items, masks_geometry, detected, ink, painted),
    }


def average_scores(rows: list[dict]) -> dict:
    """数値の項目を平均する。入れ子の項目も同じように平均する。"""
    averaged = {}
    for key in rows[0]:
        values = [row[key] for row in rows]
        averaged[key] = average_scores(values) if isinstance(values[0], dict) else round(sum(values) / len(values), 2)
    return averaged
