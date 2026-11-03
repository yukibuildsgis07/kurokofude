"""OCR ボックスを文字の種類だけで3つに分ける（設計書 §2.2、§11.1）。

氏名の文字列は使わない。OCR 直後に空文字か □ になっている。
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

import numpy as np

from .geometry import quad_frame
from .ocr import digit_ratio

# OCR の読み違いのうち、数字に戻してよいもの
CHAR_FIXES = str.maketrans({"O": "0", "o": "0", "l": "1", "I": "1", "ー": "-", "−": "-", "–": "-", "—": "-"})


@dataclass
class Classified:
    chiban: list[dict]  # (a) 地番。残す
    uncertain: list[dict]  # (b) 数字が主体だが地番の形でない。残して人が見る
    text: list[dict]  # (c) それ以外。塗る


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


def classify_boxes(boxes: list[dict], image_size: tuple[int, int], config: dict) -> Classified:
    """image_size は (幅, 高さ)。除外領域の中のボックスはどれにも入れない。"""
    chiban_cfg = config["chiban"]
    result = Classified([], [], [])
    for box in boxes:
        if in_exclude_region(np.mean(box["quad"], axis=0), image_size, config["exclude_regions"]):
            continue
        text = normalize_text(box["text"])
        if is_chiban_text(text, chiban_cfg) and not has_decoy_neighbor(box, boxes, chiban_cfg):
            result.chiban.append({**box, "text": text})
        elif digit_ratio(text) >= config["ocr"]["digit_ratio_min"]:
            result.uncertain.append({**box, "text": text})
        else:
            result.text.append(box)
    return result
