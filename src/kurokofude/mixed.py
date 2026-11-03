"""OCR が氏名と地番を1つにまとめた枠を分ける（設計書 §13.2）。文字列は □ と数字だけを見る。"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .detect import EDGE_NOISE, is_chiban_text
from .geometry import Frame, clip_below_line, project_interval
from .mask import STATUS_SPLIT
from .ocr import PLACEHOLDER
from .refine import sample_ink

NAME_FIRST = re.compile(f"({PLACEHOLDER}+)([^{PLACEHOLDER}]+)")
DIGITS_FIRST = re.compile(f"([^{PLACEHOLDER}]+)({PLACEHOLDER}+)")


@dataclass
class MixedResult:
    chiban: list[dict] = field(default_factory=list)  # 数字側。地番として残す
    split_masks: list[list] = field(default_factory=list)  # □ 側。塗る
    masked: list[list] = field(default_factory=list)  # 分けられなかった枠全体。塗る


def parse_mixed(text: str) -> tuple[bool, int, str] | None:
    """(□ が先か, □ の数, 数字側の文字列)。入り組んでいれば None。"""
    match = NAME_FIRST.fullmatch(text)
    if match:
        return True, len(match.group(1)), match.group(2)
    match = DIGITS_FIRST.fullmatch(text)
    if match:
        return False, len(match.group(2)), match.group(1)
    return None


def find_split_offset(
    quad, frame: Frame, gray: np.ndarray, boundary_ratio: float, mask_cfg: dict, mixed_cfg: dict
) -> float | None:
    """仮の分割位置の近傍でインクの谷を探す。返すのは frame.center から u 方向の距離。"""
    u_range = project_interval(quad, frame.center, frame.u)
    v_range = project_interval(quad, frame.center, frame.v)
    ink = sample_ink(gray, frame.center, frame.u, frame.v, u_range, v_range, mask_cfg["ink_threshold"])
    counts = ink.sum(axis=0)
    if not counts.any():
        return None
    length = len(counts)
    guess = boundary_ratio * length
    reach = mixed_cfg["search_ratio"] * length
    start, end = max(0, int(guess - reach)), min(length, int(guess + reach) + 1)
    low = counts[start:end] <= counts.max() * mixed_cfg["valley_max_ink_ratio"]
    min_gap = mixed_cfg["min_gap_ratio"] * (v_range[1] - v_range[0])

    best: float | None = None
    run_start = None
    for index, flag in enumerate([*low.tolist(), False]):
        if flag and run_start is None:
            run_start = index
        elif not flag and run_start is not None:
            if index - run_start >= min_gap:
                middle = start + (run_start + index) / 2
                if best is None or abs(middle - guess) < abs(best - guess):
                    best = middle
            run_start = None
    return None if best is None else u_range[0] + best


def split_one(box: dict, frame: Frame, gray: np.ndarray, config: dict) -> tuple[dict, list] | None:
    """分けられたら (数字側の地番, □ 側の多角形) を返す。"""
    parsed = parse_mixed(box["text"])
    if parsed is None:
        return None
    name_first, name_count, digits_raw = parsed
    digits = EDGE_NOISE.sub("", digits_raw)
    if not is_chiban_text(digits, config["chiban"]):
        return None
    name_width = name_count * config["mixed"]["name_char_width"]
    total = name_width + len(digits_raw)
    boundary_ratio = name_width / total if name_first else len(digits_raw) / total
    offset = find_split_offset(box["quad"], frame, gray, boundary_ratio, config["mask"], config["mixed"])
    if offset is None:
        return None
    line = frame.center + frame.u * offset
    left = clip_below_line(box["quad"], line, -frame.u)
    right = clip_below_line(box["quad"], line, frame.u)
    digit_side, name_side = (right, left) if name_first else (left, right)
    chiban = {
        **box,
        "text": digits,
        "quad": digit_side,
        "score": box["score"] * config["chiban"]["normalized_score_factor"],
        "status": STATUS_SPLIT,
    }
    return chiban, name_side


def resolve_mixed(boxes: list[dict], gray: np.ndarray, config: dict, frame_of: Callable) -> MixedResult:
    """分割を試み、できなければ枠全体を塗る側に倒す。除外領域の中は地番にしないので分割しない。"""
    result = MixedResult()
    for box in boxes:
        split = None if box["in_region"] else split_one(box, frame_of(box["quad"]), gray, config)
        if split is None:
            result.masked.append(box["quad"])
        else:
            result.chiban.append(split[0])
            result.split_masks.append(split[1])
    return result
