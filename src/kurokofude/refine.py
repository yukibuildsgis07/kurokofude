"""画素を見て、塗る多角形を整える（設計書 §11.3）。文字は読まず、インクの有無だけを見る。"""

from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np

from .geometry import Frame, clip_below_line, intervals_overlap, project_interval, quad_frame
from .pair import Pair

MIN_POLYGON_VERTICES = 3


def sample_ink(
    gray: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    threshold: int,
) -> np.ndarray:
    """u, v 軸に沿って切り出したインクの有無。行が v、列が u。画像の外は紙として扱う。"""
    cols = max(1, int(np.ceil(u_range[1] - u_range[0])))
    rows = max(1, int(np.ceil(v_range[1] - v_range[0])))
    start = origin + u * u_range[0] + v * v_range[0]
    patch_to_image = np.array([[u[0], v[0], start[0]], [u[1], v[1], start[1]]])
    patch = cv2.warpAffine(
        gray,
        patch_to_image,
        (cols, rows),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,
    )
    return patch < threshold


def _middle_of_longest_minimum_run(counts: np.ndarray) -> int:
    is_minimum = counts == counts.min()
    best_start, best_length, start = 0, 0, None
    for index, flag in enumerate([*is_minimum, False]):
        if flag and start is None:
            start = index
        elif not flag and start is not None:
            if index - start > best_length:
                best_start, best_length = start, index - start
            start = None
    return best_start + (best_length - 1) // 2


def cut_offset(chiban: Frame, quad, gray: np.ndarray, mask_cfg: dict) -> float:
    """切る線の位置を、地番の中心から「下」方向の距離で返す。インクの谷を探す。"""
    chiban_bottom = chiban.height / 2
    name_top = project_interval(quad, chiban.center, chiban.v)[0]
    low, high = sorted((chiban_bottom, name_top))
    midpoint = (low + high) / 2
    if high - low < mask_cfg["valley_min_band_px"]:
        return midpoint

    half = chiban.width / 2
    name_span = project_interval(quad, chiban.center, chiban.u)
    if intervals_overlap((-half, half), name_span):
        span = (max(-half, name_span[0]), min(half, name_span[1]))
    else:
        span = name_span
    ink = sample_ink(gray, chiban.center, chiban.u, chiban.v, span, (low, high), mask_cfg["ink_threshold"])
    counts = ink.sum(axis=1)
    if counts.min() == counts.max():
        return midpoint
    return low + _middle_of_longest_minimum_run(counts) + 0.5


def tighten(polygon, frame: Frame, gray: np.ndarray, mask_cfg: dict) -> list[list[float]]:
    """インクのない余白を詰める。詰めるだけで広げない。インクが全面にある向きは詰まらない。"""
    u_range = project_interval(polygon, frame.center, frame.u)
    v_range = project_interval(polygon, frame.center, frame.v)
    ink = sample_ink(gray, frame.center, frame.u, frame.v, u_range, v_range, mask_cfg["ink_threshold"])
    if not ink.any():
        return [list(map(float, p)) for p in polygon]

    padding = mask_cfg["tighten_padding_px"]
    result = polygon
    for axis, extent, line_counts in ((frame.u, u_range, ink.sum(axis=0)), (frame.v, v_range, ink.sum(axis=1))):
        inked = np.flatnonzero(line_counts)
        low_cut = extent[0] + inked[0] - padding
        high_cut = extent[0] + inked[-1] + 1 + padding
        if low_cut > extent[0]:
            result = clip_below_line(result, frame.center + axis * low_cut, axis)
        if high_cut < extent[1]:
            result = clip_below_line(result, frame.center + axis * high_cut, -axis)
    return [list(map(float, p)) for p in result]


def refine_pair(pair: Pair, gray: np.ndarray, mask_cfg: dict, frame_of: Callable = quad_frame) -> None:
    """ペアの塗る範囲を、地番との境目で切ってから詰める。"""
    chiban = frame_of(pair.chiban["quad"])
    source = pair.source["quad"]
    offset = cut_offset(chiban, source, gray, mask_cfg)
    cut = clip_below_line(source, chiban.center + chiban.v * offset, chiban.v)
    if len(cut) < MIN_POLYGON_VERTICES:
        cut = source
    pair.quad = tighten(cut, frame_of(source), gray, mask_cfg)


def refine_unpaired(quad, gray: np.ndarray, mask_cfg: dict, frame_of: Callable = quad_frame) -> list[list[float]]:
    return tighten(quad, frame_of(quad), gray, mask_cfg)
