"""塗りつぶしと出力（設計書 §2.4）。画像のサイズと画素の位置は変えない。"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import tifffile
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from .pair import Pair

GRAY_NDIM = 2
WORLD_FILE_SUFFIX = ".tfw"
JOIN_STYLE_MITRE = 2  # 四角形を角を保ったまま膨らませる
STATUS_AUTO = "auto"
STATUS_UNPAIRED = "unpaired"
STATUS_IN_TITLE_BLOCK = "in_title_block"
STATUS_SPLIT = "split"
STATUS_MIXED_MASKED = "mixed_masked"
STATUS_UNCERTAIN_CHIBAN = "uncertain_chiban"
STATUS_TITLE_META = "title_meta"
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


def protected_area(quads: list[list], margin_px: float):
    """塗ってはいけない範囲：地番と数字主体の枠を少し膨らませたもの（§12.2）。"""
    return unary_union(
        [Polygon(quad).buffer(0).buffer(margin_px, join_style=JOIN_STYLE_MITRE) for quad in quads]
    )


def _simple_pieces(geometry) -> list[list[list[float]]]:
    """穴のない単純なポリゴンの頂点列に分解する。穴は、その位置で左右に割って消す。"""
    if geometry.is_empty:
        return []
    if geometry.geom_type != "Polygon":
        return [piece for part in getattr(geometry, "geoms", []) for piece in _simple_pieces(part)]
    if geometry.area <= 0:
        return []
    if geometry.interiors:
        min_x, min_y, max_x, max_y = geometry.bounds
        split_x = Polygon(geometry.interiors[0]).centroid.x
        return _simple_pieces(geometry.intersection(box(min_x, min_y, split_x, max_y))) + _simple_pieces(
            geometry.intersection(box(split_x, min_y, max_x, max_y))
        )
    return [[list(point) for point in geometry.exterior.coords[:-1]]]


def subtract_protected(polygon, protected) -> list[list[list[float]]]:
    """塗る範囲から保護範囲を引く。重ならなければ元の頂点のまま返す。"""
    shape = Polygon(polygon).buffer(0)
    if not shape.intersects(protected):
        return [polygon]
    return _simple_pieces(shape.difference(protected))


def build_masks(pairs: list[Pair], others: list[tuple[list, str]], protected) -> tuple[list[dict], set[int]]:
    """塗りの項目と、塗りが残った地番の id を返す。ポリゴン1つが1項目。"""
    masks: list[dict] = []
    masked_ids: set[int] = set()
    for pair in pairs:
        pieces = subtract_protected(pair.quad, protected)
        if pieces:
            masked_ids.add(id(pair.chiban))
        masks.extend({"quad": piece, "chiban": pair.chiban["text"], "status": STATUS_AUTO} for piece in pieces)
    for polygons, status in others:
        for polygon in polygons:
            masks.extend(
                {"quad": piece, "chiban": None, "status": status} for piece in subtract_protected(polygon, protected)
            )
    return masks, masked_ids


def build_result(
    source_name: str,
    image_size: tuple[int, int],
    chiban_boxes: list[dict],
    masks: list[dict],
    masked_ids: set[int],
    uncertain: list[dict],
    title_meta: list[dict] = (),
    frame_quad: list | None = None,
) -> dict:
    """地番一覧 JSON（§2.4、§11.1）。氏名の文字列は持たない。"""
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
                **({"status": box["status"]} if "status" in box else {}),
            }
            for box in chiban_boxes
        ],
        "uncertain": [
            {"text": box["text"], "quad": box["quad"], "score": box["score"], "status": STATUS_UNCERTAIN_CHIBAN}
            for box in uncertain
        ],
        "title_meta": [
            {"text": box["text"], "quad": box["quad"], "score": box["score"], "status": STATUS_TITLE_META}
            for box in title_meta
        ],
        "masks": masks,
        "frame_quad": frame_quad,  # 図郭線（画素座標4点）。安全なデータで、④の照合で使う（§5）
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
    protected_quads: list[list] | None = None,
    split: list[list] = (),
    mixed_masked: list[list] = (),
    title_meta: list[dict] = (),
    frame_quad: list | None = None,
) -> tuple[Path, dict]:
    in_path, out_dir = Path(in_path), Path(out_dir)
    out_tif = out_dir / in_path.name
    if out_tif.resolve() == in_path.resolve():
        raise ValueError("出力先が入力と同じです。元の図面は上書きしません")
    if protected_quads is None:
        protected_quads = [box["quad"] for box in [*chiban_boxes, *uncertain, *title_meta]]
    protected = protected_area(protected_quads, mask_cfg["protect_margin_px"])
    others = [
        (list(unpaired), STATUS_UNPAIRED),
        (list(title), STATUS_IN_TITLE_BLOCK),
        (list(split), STATUS_SPLIT),
        (list(mixed_masked), STATUS_MIXED_MASKED),
    ]
    masks, masked_ids = build_masks(pairs, others, protected)
    size = write_masked_tiff(in_path, out_tif, [mask["quad"] for mask in masks], mask_cfg["color"])
    world_file = in_path.with_suffix(WORLD_FILE_SUFFIX)
    if world_file.is_file():
        shutil.copyfile(world_file, out_dir / world_file.name)
    result = build_result(
        in_path.name, size, chiban_boxes, masks, masked_ids, list(uncertain), list(title_meta), frame_quad
    )
    (out_dir / f"{in_path.stem}.chiban.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return out_tif, result
