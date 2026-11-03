"""PaddleOCR の呼び出しと結果のキャッシュ（設計書 §2.1）。

返す形：[{"text": str, "score": float, "quad": [[x, y] × 4]}]
quad は OCR が返した傾いた四角形のまま。氏名の文字列はここで捨てる（§10）。
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import tifffile
from shapely.geometry import Polygon

HASH_CHUNK_BYTES = 1024 * 1024
GRAY_NDIM = 2


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def scrub_text(text: str, keep_pattern: str) -> str:
    """地番になりうる文字列だけ残す。氏名はキャッシュにもログにも残さない。"""
    return text if re.fullmatch(keep_pattern, text) else ""


def tile_origins(length: int, max_side: int, overlap: int) -> list[int]:
    """1辺を、重なりありで max_side 以下に分割したときの開始位置。"""
    if length <= max_side:
        return [0]
    stride = max_side - overlap
    origins = list(range(0, length - max_side, stride))
    origins.append(length - max_side)
    return origins


def merge_duplicates(boxes: list[dict], iou_threshold: float) -> list[dict]:
    """タイル境界で二重に検出したボックスを、スコアの高い方に統合する。"""
    kept: list[tuple[dict, Polygon]] = []
    for box in sorted(boxes, key=lambda b: b["score"], reverse=True):
        polygon = Polygon(box["quad"]).buffer(0)
        duplicate = False
        for _, other in kept:
            union = polygon.union(other).area
            if union > 0 and polygon.intersection(other).area / union > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append((box, polygon))
    return [box for box, _ in kept]


def _create_engine(ocr_cfg: dict):
    from paddleocr import PaddleOCR  # 重いので必要なときだけ読む

    return PaddleOCR(
        lang=ocr_cfg["lang"],
        enable_mkldnn=ocr_cfg["enable_mkldnn"],
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _recognize(engine, image_bgr: np.ndarray, origin: tuple[int, int], keep_pattern: str) -> list[dict]:
    boxes = []
    for page in engine.predict(image_bgr):
        for text, score, poly in zip(page["rec_texts"], page["rec_scores"], page["rec_polys"]):
            quad = (np.asarray(poly, dtype=float) + origin).tolist()
            boxes.append({"text": scrub_text(text, keep_pattern), "score": float(score), "quad": quad})
    return boxes


def read_image_for_ocr(image_path: Path) -> np.ndarray:
    image = tifffile.imread(image_path)
    if image.ndim == GRAY_NDIM:
        image = np.stack([image] * 3, axis=-1)
    return image[..., :3][..., ::-1].copy()  # RGB → BGR


def run_engine(image_bgr: np.ndarray, ocr_cfg: dict) -> list[dict]:
    engine = _create_engine(ocr_cfg)
    keep_pattern = ocr_cfg["keep_text_pattern"]
    height, width = image_bgr.shape[:2]
    max_side, overlap = ocr_cfg["max_side"], ocr_cfg["tile_overlap"]
    boxes: list[dict] = []
    for y in tile_origins(height, max_side, overlap):
        for x in tile_origins(width, max_side, overlap):
            tile = image_bgr[y : y + max_side, x : x + max_side]
            boxes.extend(_recognize(engine, tile, (x, y), keep_pattern))
    if max(height, width) > max_side:
        boxes = merge_duplicates(boxes, ocr_cfg["tile_merge_iou"])
    return boxes


def run_ocr(image_path: Path, ocr_cfg: dict, cache_dir: Path) -> list[dict]:
    """同じ画像では OCR を再実行しない。キャッシュの鍵は画像の SHA-256。"""
    cache_path = Path(cache_dir) / f"{file_sha256(image_path)}.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    boxes = run_engine(read_image_for_ocr(image_path), ocr_cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(boxes, ensure_ascii=False), encoding="utf-8")
    return boxes
