"""地番ごとに、真下の氏名ボックスを選ぶ（設計書 §2.3）。

氏名は読まない。候補の判定に使うのは四角形の座標だけ。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import Frame, angle_difference_deg, clip_below_line, intervals_overlap, project_interval, quad_frame


@dataclass
class Pair:
    chiban: dict
    quad: list[list[float]]  # 塗る多角形（切った後）
    distance: float


def gap_below(chiban: Frame, candidate_center: np.ndarray) -> float:
    """地番の下辺の中点から、候補の中心まで「下」方向に測った距離。"""
    bottom_mid = chiban.center + chiban.v * chiban.height / 2
    return float((candidate_center - bottom_mid) @ chiban.v)


def is_candidate(chiban: Frame, quad, pair_cfg: dict) -> tuple[bool, float]:
    frame = quad_frame(quad)
    if angle_difference_deg(frame.angle_deg, chiban.angle_deg) > pair_cfg["max_angle_deg"]:
        return False, 0.0
    distance = gap_below(chiban, frame.center)
    if not 0 < distance <= pair_cfg["max_gap"] * chiban.height:
        return False, 0.0
    half_width = chiban.width / 2
    overlap = intervals_overlap((-half_width, half_width), project_interval(quad, chiban.center, chiban.u))
    lateral = abs(float((frame.center - chiban.center) @ chiban.u))
    if not overlap and lateral > pair_cfg["max_lateral"] * chiban.width:
        return False, 0.0
    return True, distance


def cut_at_chiban_bottom(quad, chiban: Frame, margin_px: float) -> list[list[float]]:
    """候補の上側を、地番の下辺を延ばした直線（＋余白）で切る。"""
    line_point = chiban.center + chiban.v * (chiban.height / 2 + margin_px)
    return clip_below_line(quad, line_point, chiban.v)


def pair_names(boxes: list[dict], chiban_boxes: list[dict], pair_cfg: dict) -> list[Pair]:
    """地番 → 氏名ボックスを選ぶ。1つの候補を複数の地番が選んだら、近い方に渡す。"""
    chiban_ids = {id(box) for box in chiban_boxes}
    chiban_quads = {tuple(map(tuple, box["quad"])) for box in chiban_boxes}
    candidates = [
        box
        for box in boxes
        if id(box) not in chiban_ids and tuple(map(tuple, box["quad"])) not in chiban_quads
    ]
    best_for_candidate: dict[int, tuple[float, dict, dict]] = {}
    for chiban in chiban_boxes:
        frame = quad_frame(chiban["quad"])
        best: tuple[float, dict] | None = None
        for candidate in candidates:
            ok, distance = is_candidate(frame, candidate["quad"], pair_cfg)
            if ok and (best is None or distance < best[0]):
                best = (distance, candidate)
        if best is None:
            continue
        distance, candidate = best
        current = best_for_candidate.get(id(candidate))
        if current is None or distance < current[0]:
            best_for_candidate[id(candidate)] = (distance, chiban, candidate)

    pairs = []
    for distance, chiban, candidate in best_for_candidate.values():
        frame = quad_frame(chiban["quad"])
        quad = cut_at_chiban_bottom(candidate["quad"], frame, pair_cfg["cut_margin_px"])
        if quad:
            pairs.append(Pair(chiban=chiban, quad=quad, distance=distance))
    return pairs
