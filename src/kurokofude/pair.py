"""地番ごとに、真下の氏名ボックスを選ぶ（設計書 §2.3）。

氏名は読まない。候補の判定に使うのは四角形の座標だけ。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .geometry import Frame, angle_difference_deg, intervals_overlap, project_interval, quad_frame


@dataclass
class Pair:
    chiban: dict
    quad: list[list[float]]  # 塗る多角形。refine.py が切って詰める
    distance: float
    source: dict  # 元の OCR ボックス


def gap_below(chiban: Frame, candidate_center: np.ndarray) -> float:
    """地番の下辺の中点から、候補の中心まで「下」方向に測った距離。"""
    bottom_mid = chiban.center + chiban.v * chiban.height / 2
    return float((candidate_center - bottom_mid) @ chiban.v)


def is_candidate(chiban: Frame, quad, frame: Frame, pair_cfg: dict) -> tuple[bool, float]:
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


def pair_names(
    boxes: list[dict], chiban_boxes: list[dict], pair_cfg: dict, frame_of: Callable = quad_frame
) -> list[Pair]:
    """条件を満たす枠を全部、その地番の氏名群とする（§11.2）。1つの枠は最も近い地番に渡す。"""
    chiban_ids = {id(box) for box in chiban_boxes}
    chiban_quads = {tuple(map(tuple, box["quad"])) for box in chiban_boxes}
    candidates = [
        box
        for box in boxes
        if id(box) not in chiban_ids and tuple(map(tuple, box["quad"])) not in chiban_quads
    ]
    nearest: dict[int, tuple[float, dict, dict]] = {}
    for chiban in chiban_boxes:
        frame = frame_of(chiban["quad"])
        for candidate in candidates:
            ok, distance = is_candidate(frame, candidate["quad"], frame_of(candidate["quad"]), pair_cfg)
            current = nearest.get(id(candidate))
            if ok and (current is None or distance < current[0]):
                nearest[id(candidate)] = (distance, chiban, candidate)

    return [
        Pair(chiban=chiban, quad=candidate["quad"], distance=distance, source=candidate)
        for distance, chiban, candidate in nearest.values()
    ]
