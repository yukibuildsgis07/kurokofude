"""図面1枚を OCR → 判定 → 塗りまで通す。CLI と MCP サーバーの共通の入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tifffile

from .detect import detect_chiban
from .mask import write_outputs
from .ocr import run_ocr
from .pair import pair_names


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
    chiban_boxes = detect_chiban(boxes, image_size_of(in_path), config)
    pairs = pair_names(boxes, chiban_boxes, config["pair"])
    out_path, _ = write_outputs(in_path, out_dir, chiban_boxes, pairs, config["mask"])
    return MaskSummary(out_path, len(boxes), len(chiban_boxes), len(pairs))
