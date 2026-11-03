"""PaddleOCR の呼び出しと結果のキャッシュ（設計書 §2.1）。

返す形：[{"text": str, "score": float, "quad": [[x, y] × 4]}]
quad は OCR が返した傾いた四角形のまま。氏名の文字列はここで捨てる（§10）。
"""

from __future__ import annotations

import hashlib
import json
import re
import string
import unicodedata
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import tifffile
from shapely.geometry import Polygon

from .geometry import Frame, quad_frame

HASH_CHUNK_BYTES = 1024 * 1024
GRAY_NDIM = 2
PLACEHOLDER = "□"
RULE_TAG_LENGTH = 8
ASCII_DIGITS = frozenset(string.digits)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def digit_ratio(text: str) -> float:
    """空白を除いた文字のうち、数字の割合。"""
    chars = [c for c in unicodedata.normalize("NFKC", text) if not c.isspace()]
    if not chars:
        return 0.0
    return sum(1 for c in chars if c in ASCII_DIGITS) / len(chars)


def scrub_text(text: str, keep_char_class: str, keep_words: list[str] = ()) -> str:
    """氏名をキャッシュにもログにも残さない（§11.1、§18.1）。

    数字を含む文字列と、図名の定型語を含む文字列は、許可外の文字を □ にして残す。
    定型語に一致した部分だけは文字のまま残す。それ以外は空文字にする。
    """
    allowed = [re.fullmatch(keep_char_class, c) is not None for c in text]
    if all(allowed):
        return text
    found_word = False
    for word in keep_words:
        start = text.find(word)
        while word and start >= 0:
            found_word = True
            allowed[start : start + len(word)] = [True] * len(word)
            start = text.find(word, start + len(word))
    has_digit = any(c in ASCII_DIGITS for c in unicodedata.normalize("NFKC", text))
    if not has_digit and not found_word:
        return ""
    return "".join(c if ok else PLACEHOLDER for c, ok in zip(text, allowed))


def scrub_rule_tag(ocr_cfg: dict) -> str:
    """保存の規則が変わったら、キャッシュを使い直さないための目印。"""
    rule = json.dumps([ocr_cfg["keep_char_class"], list(ocr_cfg.get("title_keep_words", []))], ensure_ascii=False)
    return hashlib.sha256(rule.encode("utf-8")).hexdigest()[:RULE_TAG_LENGTH]


def cache_file(image_path: Path, ocr_cfg: dict, cache_dir: Path) -> Path:
    return Path(cache_dir) / f"{file_sha256(image_path)}-{scrub_rule_tag(ocr_cfg)}.json"


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


@lru_cache(maxsize=1)
def _create_engine(lang: str, enable_mkldnn: bool):
    """モデルの読み込みは重いので、batch では使い回す。"""
    from paddleocr import PaddleOCR

    return PaddleOCR(
        lang=lang,
        enable_mkldnn=enable_mkldnn,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


def _recognize(engine, image_bgr: np.ndarray, origin: tuple[int, int], ocr_cfg: dict) -> list[dict]:
    boxes = []
    for page in engine.predict(image_bgr):
        for text, score, poly in zip(page["rec_texts"], page["rec_scores"], page["rec_polys"]):
            quad = (np.asarray(poly, dtype=float) + origin).tolist()
            scrubbed = scrub_text(text, ocr_cfg["keep_char_class"], ocr_cfg.get("title_keep_words", []))
            boxes.append({"text": scrubbed, "score": float(score), "quad": quad})
    return boxes


@lru_cache(maxsize=1)
def _create_recognizer(model_name: str, enable_mkldnn: bool):
    from paddleocr import TextRecognition

    return TextRecognition(model_name=model_name, enable_mkldnn=enable_mkldnn)


def recognize_crops(crops: list[np.ndarray], ocr_cfg: dict) -> list[tuple[str, float]]:
    """検出なしで認識だけを行う。"""
    recognizer = _create_recognizer(ocr_cfg["rerecognize"]["rec_model_name"], ocr_cfg["enable_mkldnn"])
    return [(page["rec_text"], float(page["rec_score"])) for page in recognizer.predict(crops)]


def crop_along_text(image_bgr: np.ndarray, frame: Frame, expand_ratio: float, reverse: bool) -> np.ndarray:
    """枠を長辺方向に広げて切り出し、読み方向が右になるように回す。"""
    margin = frame.width * expand_ratio
    width = max(1, int(np.ceil(frame.width + 2 * margin)))
    height = max(1, int(np.ceil(frame.height)))
    sign = -1.0 if reverse else 1.0
    u, v = frame.u * sign, frame.v * sign
    start = frame.center - u * width / 2 - v * height / 2
    patch_to_image = np.array([[u[0], v[0], start[0]], [u[1], v[1], start[1]]])
    return cv2.warpAffine(
        image_bgr, patch_to_image, (width, height), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE
    )


def rerecognize_steep(boxes: list[dict], image_bgr: np.ndarray, ocr_cfg: dict) -> list[dict]:
    """急な角度の枠の認識をやり直す（§15.1）。四角形は変えない。"""
    cfg = ocr_cfg["rerecognize"]
    targets = []
    for box in boxes:
        frame = quad_frame(box["quad"])
        if abs(frame.angle_deg) > cfg["min_angle_deg"] and frame.width >= cfg["min_aspect"] * frame.height:
            targets.append((box, frame))
    if not targets:
        return boxes
    crops = [
        crop_along_text(image_bgr, frame, cfg["expand_ratio"], reverse)
        for _, frame in targets
        for reverse in (False, True)
    ]
    results = recognize_crops(crops, ocr_cfg)
    for index, (box, _) in enumerate(targets):
        text, score = max(results[2 * index : 2 * index + 2], key=lambda result: result[1])
        box["text"] = scrub_text(text, ocr_cfg["keep_char_class"], ocr_cfg.get("title_keep_words", []))
        box["score"] = score * cfg["score_factor"]
    return boxes


def read_image_for_ocr(image_path: Path) -> np.ndarray:
    image = tifffile.imread(image_path)
    if image.ndim == GRAY_NDIM:
        image = np.stack([image] * 3, axis=-1)
    return image[..., :3][..., ::-1].copy()  # RGB → BGR


def run_engine(image_bgr: np.ndarray, ocr_cfg: dict) -> list[dict]:
    engine = _create_engine(ocr_cfg["lang"], ocr_cfg["enable_mkldnn"])
    height, width = image_bgr.shape[:2]
    max_side, overlap = ocr_cfg["max_side"], ocr_cfg["tile_overlap"]
    boxes: list[dict] = []
    for y in tile_origins(height, max_side, overlap):
        for x in tile_origins(width, max_side, overlap):
            tile = image_bgr[y : y + max_side, x : x + max_side]
            boxes.extend(_recognize(engine, tile, (x, y), ocr_cfg))
    if max(height, width) > max_side:
        boxes = merge_duplicates(boxes, ocr_cfg["tile_merge_iou"])
    return rerecognize_steep(boxes, image_bgr, ocr_cfg)


def run_ocr(image_path: Path, ocr_cfg: dict, cache_dir: Path) -> list[dict]:
    """同じ画像では OCR を再実行しない。キャッシュの鍵は画像の SHA-256。"""
    cache_path = cache_file(image_path, ocr_cfg, cache_dir)
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    boxes = run_engine(read_image_for_ocr(image_path), ocr_cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(boxes, ensure_ascii=False), encoding="utf-8")
    return boxes
