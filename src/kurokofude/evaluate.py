"""正解 JSON との採点（設計書 §2.5）。氏名の文字列は使わず、面積だけで測る。"""

from __future__ import annotations

from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

STATUS_REMOVED = "removed"
PERCENT = 100.0


def _polygon(quad) -> Polygon:
    return Polygon(quad).buffer(0)


def _ratio(numerator: float, denominator: float) -> float:
    return PERCENT * numerator / denominator if denominator else 0.0


def evaluate(truth: dict, result: dict) -> dict:
    masks = unary_union(
        [_polygon(m["quad"]) for m in result["masks"] if m.get("status") != STATUS_REMOVED and len(m["quad"]) >= 3]
    )
    name_polygons = [_polygon(item["name"]["quad"]) for item in truth["items"]]
    chiban_polygons = [_polygon(item["chiban"]["quad"]) for item in truth["items"]]
    names_union = unary_union(name_polygons)

    name_area = sum(p.area for p in name_polygons)
    covered = sum(p.intersection(masks).area for p in name_polygons)
    outside = masks.difference(names_union).area
    survived = sum(1 for p in chiban_polygons if p.intersection(masks).area == 0)

    detected = result["chiban"]
    matched_truth = 0
    for item, polygon in zip(truth["items"], chiban_polygons):
        if any(d["text"] == item["chiban"]["text"] and polygon.contains(Point(d["center"])) for d in detected):
            matched_truth += 1
    false_positives = sum(
        1 for d in detected if not any(p.contains(Point(d["center"])) for p in chiban_polygons)
    )
    return {
        "name_coverage_pct": round(_ratio(covered, name_area), 2),
        "overmask_pct": round(_ratio(outside, masks.area), 2),
        "chiban_survival_pct": round(_ratio(survived, len(chiban_polygons)), 2),
        "chiban_detection_pct": round(_ratio(matched_truth, len(chiban_polygons)), 2),
        "false_positives": false_positives,
        "truth_items": len(chiban_polygons),
        "masks": len(result["masks"]),
    }
