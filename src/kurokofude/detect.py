"""地番ボックスの判定（設計書 §2.2）。"""

from __future__ import annotations

import re
import unicodedata

import numpy as np

from .geometry import quad_frame

# OCR の読み違いのうち、数字に戻してよいもの
CHAR_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "|": "1", "ー": "-", "−": "-", "–": "-", "—": "-"})


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).translate(CHAR_FIXES).replace(" ", "")


def is_chiban_text(text: str, chiban_cfg: dict) -> bool:
    if not re.fullmatch(chiban_cfg["pattern"], text):
        return False
    digits = len(text.split("-")[0])
    return chiban_cfg["min_digits"] <= digits <= chiban_cfg["max_digits"]


def in_exclude_region(center: np.ndarray, image_size: tuple[int, int], regions: list) -> bool:
    x, y = center[0] / image_size[0], center[1] / image_size[1]
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in regions)


def has_decoy_neighbor(box: dict, boxes: list[dict], chiban_cfg: dict) -> bool:
    """縮尺の `1:500` のように、近くに記号があれば地番ではない。"""
    frame = quad_frame(box["quad"])
    reach = chiban_cfg["decoy_neighbor_distance"] * frame.height
    for other in boxes:
        if other is box or not re.search(chiban_cfg["decoy_pattern"], other["text"]):
            continue
        if np.linalg.norm(np.mean(other["quad"], axis=0) - frame.center) <= reach:
            return True
    return False


def detect_chiban(boxes: list[dict], image_size: tuple[int, int], config: dict) -> list[dict]:
    """地番と判定したボックスを、正規化した text で返す。image_size は (幅, 高さ)。"""
    chiban_cfg = config["chiban"]
    found = []
    for box in boxes:
        text = normalize_text(box["text"])
        if not is_chiban_text(text, chiban_cfg):
            continue
        if in_exclude_region(np.mean(box["quad"], axis=0), image_size, config["exclude_regions"]):
            continue
        if has_decoy_neighbor(box, boxes, chiban_cfg):
            continue
        found.append({**box, "text": text})
    return found
