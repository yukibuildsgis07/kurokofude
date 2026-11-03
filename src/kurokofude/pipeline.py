"""図面1枚を OCR → 判定 → 塗りまで通す。CLI と MCP サーバーの共通の入口。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import tifffile

from .detect import classify_boxes
from .geometry import FrameResolver
from .mask import write_outputs
from .mixed import resolve_mixed
from .ocr import read_image_for_ocr, run_ocr
from .pair import pair_names
from .refine import refine_lone_text, refine_pair, refine_unpaired, tight_box


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
    ocr_cfg = {**config["ocr"], "title_keep_words": config.get("title_keep_words", [])}
    boxes = run_ocr(in_path, ocr_cfg, cache_dir)
    classified = classify_boxes(boxes, image_size_of(in_path), config)
    frame_of = FrameResolver(boxes, config["angle"]).frame
    gray = cv2.cvtColor(read_image_for_ocr(in_path), cv2.COLOR_BGR2GRAY)
    mixed = resolve_mixed(classified.mixed, gray, config, frame_of)
    chiban_boxes = classified.chiban + mixed.chiban
    pairs = pair_names(classified.text, chiban_boxes, config["pair"], frame_of)
    for pair in pairs:
        refine_pair(pair, gray, config["mask"], frame_of)
    paired_ids = {id(pair.source) for pair in pairs}
    unpaired = (
        [
            refine_lone_text(box["quad"], gray, config["mask"], frame_of)
            for box in classified.text
            if id(box) not in paired_ids
        ]
        if config["mask"]["unpaired_text"]
        else []
    )
    title = [refine_unpaired(box["quad"], gray, config["mask"], frame_of) for box in classified.title_text]
    protected_quads = [
        tight_box(box["quad"], frame_of(box["quad"]), gray, config["mask"])
        for box in [*chiban_boxes, *classified.uncertain, *classified.title_meta]
    ]
    split = [refine_unpaired(quad, gray, config["mask"], frame_of) for quad in mixed.split_masks]
    mixed_masked = [refine_unpaired(quad, gray, config["mask"], frame_of) for quad in mixed.masked]
    out_path, result = write_outputs(
        in_path,
        out_dir,
        chiban_boxes,
        pairs,
        config["mask"],
        unpaired,
        classified.uncertain,
        title,
        protected_quads,
        split,
        mixed_masked,
        classified.title_meta,
    )
    return MaskSummary(out_path, len(boxes), len(chiban_boxes), len(result["masks"]))
