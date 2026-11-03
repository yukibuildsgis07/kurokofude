"""図面1枚を OCR → 判定 → 塗りまで通す。CLI と MCP サーバーの共通の入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import tifffile

from .detect import classify_boxes
from .mask import write_outputs
from .ocr import read_image_for_ocr, run_ocr
from .pair import pair_names
from .refine import refine_pair, refine_unpaired


@dataclass
class MaskSummary:
    out_path: Path
    boxes: int
    chiban: int
    masks: int


def image_size_of(path: Path) -> tuple[int, int]:
    with tifffile.TiffFile(path) as tif:
        shape = tif.pages[0].shape
    return shape[1], shape[0]


def mask_file(in_path: Path, config: dict, out_dir: Path, cache_dir: Path) -> MaskSummary:
    boxes = run_ocr(in_path, config["ocr"], cache_dir)
    classified = classify_boxes(boxes, image_size_of(in_path), config)
    pairs = pair_names(classified.text, classified.chiban, config["pair"])
    gray = cv2.cvtColor(read_image_for_ocr(in_path), cv2.COLOR_BGR2GRAY)
    for pair in pairs:
        refine_pair(pair, gray, config["mask"])
    paired_ids = {id(pair.source) for pair in pairs}
    unpaired = (
        [refine_unpaired(box["quad"], gray, config["mask"]) for box in classified.text if id(box) not in paired_ids]
        if config["mask"]["unpaired_text"]
        else []
    )
    out_path, _ = write_outputs(
        in_path, out_dir, classified.chiban, pairs, config["mask"], unpaired, classified.uncertain
    )
    return MaskSummary(out_path, len(boxes), len(classified.chiban), len(pairs) + len(unpaired))
