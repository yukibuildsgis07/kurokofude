"""塗りつぶしと出力（設計書 §2.4）。画像のサイズと画素の位置は変えない。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import tifffile

from .pair import Pair

GRAY_NDIM = 2
WORLD_FILE_SUFFIX = ".tfw"
STATUS_AUTO = "auto"
STATUS_UNPAIRED = "unpaired"
STATUS_IN_TITLE_BLOCK = "in_title_block"
STATUS_UNCERTAIN_CHIBAN = "uncertain_chiban"
# GeoTIFF のタグ：ModelPixelScale, ModelTiepoint, ModelTransformation, GeoKey×3, GDAL_METADATA, GDAL_NODATA
GEO_TAG_CODES = (33550, 33922, 34264, 34735, 34736, 34737, 42112, 42113)


def fill_polygons(image: np.ndarray, polygons: list, color) -> np.ndarray:
    """傾いた四角形を塗る。元の配列は変更しない。"""
    painted = image.copy()
    if image.ndim == GRAY_NDIM:
        fill_value = int(color[0])
    else:
        fill_value = tuple(int(c) for c in color) + (0,) * (image.shape[2] - len(color))
    for polygon in polygons:
        points = np.round(np.asarray(polygon, dtype=float)).astype(np.int32)
        cv2.fillPoly(painted, [points], fill_value)
    return painted


def _read_tiff(path: Path) -> tuple[np.ndarray, dict]:
    """画素と、書き戻すときに保つ属性（解像度・GeoTIFF タグ）を読む。"""
    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        image = page.asarray()
        extratags = []
        for code in GEO_TAG_CODES:
            tag = page.tags.get(code)
            if tag is not None:
                extratags.append((tag.code, tag.dtype, tag.count, tag.value, True))
        attributes = {"photometric": page.photometric, "extratags": extratags}
        x_res = page.tags.get("XResolution")
        y_res = page.tags.get("YResolution")
        unit = page.tags.get("ResolutionUnit")
        if x_res is not None and y_res is not None:
            attributes["resolution"] = (x_res.value, y_res.value)
            if unit is not None:
                attributes["resolutionunit"] = unit.value
    return image, attributes


def write_masked_tiff(in_path: Path, out_path: Path, polygons: list, color) -> tuple[int, int]:
    """塗った TIFF を書き、(幅, 高さ) を返す。"""
    image, attributes = _read_tiff(in_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_path, fill_polygons(image, polygons, color), **attributes)
    return image.shape[1], image.shape[0]


def build_result(
    source_name: str,
    image_size: tuple[int, int],
    chiban_boxes: list[dict],
    pairs: list[Pair],
    unpaired: list[list],
    uncertain: list[dict],
    title: list[list] = (),
) -> dict:
    """地番一覧 JSON（§2.4、§11.1）。氏名の文字列は持たない。"""
    masked_ids = {id(pair.chiban) for pair in pairs}
    return {
        "source": source_name,
        "image_size": list(image_size),
        "chiban": [
            {
                "text": box["text"],
                "center": np.mean(box["quad"], axis=0).tolist(),
                "quad": box["quad"],
                "score": box["score"],
                "masked_below": id(box) in masked_ids,
            }
            for box in chiban_boxes
        ],
        "uncertain": [
            {"text": box["text"], "quad": box["quad"], "score": box["score"], "status": STATUS_UNCERTAIN_CHIBAN}
            for box in uncertain
        ],
        "masks": [{"quad": pair.quad, "chiban": pair.chiban["text"], "status": STATUS_AUTO} for pair in pairs]
        + [{"quad": quad, "chiban": None, "status": STATUS_UNPAIRED} for quad in unpaired]
        + [{"quad": quad, "chiban": None, "status": STATUS_IN_TITLE_BLOCK} for quad in title],
    }


def write_outputs(
    in_path: Path,
    out_dir: Path,
    chiban_boxes: list[dict],
    pairs: list[Pair],
    mask_cfg: dict,
    unpaired: list[list] = (),
    uncertain: list[dict] = (),
    title: list[list] = (),
) -> tuple[Path, dict]:
    in_path, out_dir = Path(in_path), Path(out_dir)
    out_tif = out_dir / in_path.name
    if out_tif.resolve() == in_path.resolve():
        raise ValueError("出力先が入力と同じです。元の図面は上書きしません")
    polygons = [pair.quad for pair in pairs] + list(unpaired) + list(title)
    size = write_masked_tiff(in_path, out_tif, polygons, mask_cfg["color"])
    world_file = in_path.with_suffix(WORLD_FILE_SUFFIX)
    if world_file.is_file():
        shutil.copyfile(world_file, out_dir / world_file.name)
    result = build_result(in_path.name, size, chiban_boxes, pairs, list(unpaired), list(uncertain), list(title))
    (out_dir / f"{in_path.stem}.chiban.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return out_tif, result
