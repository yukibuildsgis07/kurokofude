"""④ デモ用の比較画像：照合の前後で、塗った図面が公図にどう重なるかを同じ範囲・同じ縮尺で描く。

python tools/georef/compare.py dummy_001

使うのは塗った図面（data/masked/）だけ。生の図面は読まない。
公共座標は (東, 北)。北が上のキャンバスに描くので、キャンバスの行は北が増えると減る。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import tifffile
import yaml
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent))
from experiment import position_error, read_world_file, shifted, to_world  # noqa: E402
from georef import load_config  # noqa: E402

MASKED = Path("data") / "masked"
TRUTH = Path("data") / "truth"
REF = Path("data") / "ref"
DOCS = Path("docs")
FONT_CONFIG = Path(__file__).resolve().parents[1] / "make_dummy" / "config.yaml"


def image_corners(size: tuple[int, int]) -> np.ndarray:
    width, height = size
    return np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)


def canvas_extent(tfws: list[np.ndarray], size: tuple[int, int], margin_m: float) -> tuple[float, float, float, float]:
    """すべての .tfw で図面が覆う範囲に余白を足す。返すのは (西, 南, 東, 北)。"""
    points = np.vstack([to_world(tfw, image_corners(size)) for tfw in tfws])
    west, south = points.min(axis=0) - margin_m
    east, north = points.max(axis=0) + margin_m
    return float(west), float(south), float(east), float(north)


def world_to_canvas(world: np.ndarray, extent: tuple[float, float, float, float], m_per_px: float) -> np.ndarray:
    """(東, 北) → キャンバスの (列, 行)。行は下向きなので、北から引く。"""
    west, _, _, north = extent
    return np.column_stack([(world[:, 0] - west) / m_per_px, (north - world[:, 1]) / m_per_px])


def pixel_to_canvas_matrix(tfw: np.ndarray, extent: tuple, m_per_px: float) -> np.ndarray:
    """図面の画素 → キャンバスの 2×3 行列。3点の対応から解く。"""
    source = np.array([[0.0, 0.0], [1000.0, 0.0], [0.0, 1000.0]])
    target = world_to_canvas(to_world(tfw, source), extent, m_per_px)
    return np.linalg.solve(np.hstack([source, np.ones((3, 1))]), target).T


def draw_parcels(canvas: np.ndarray, geojson: dict, extent: tuple, m_per_px: float, cfg: dict, scale: int) -> None:
    color = tuple(cfg["parcel_color_rgb"])
    for feature in geojson["features"]:
        geometry = feature["geometry"]
        polygons = geometry["coordinates"] if geometry["type"] == "MultiPolygon" else [geometry["coordinates"]]
        for polygon in polygons:
            for ring in polygon:
                points = world_to_canvas(np.array(ring)[:, :2], extent, m_per_px)
                shifted_points = np.round(points * (1 << cfg["line_shift_bits"])).astype(np.int32)
                cv2.polylines(
                    canvas, [shifted_points], True, color, cfg["parcel_width_px"] * scale, cv2.LINE_AA, cfg["line_shift_bits"]
                )


def render(image: np.ndarray, tfw: np.ndarray, geojson: dict, extent: tuple, cfg: dict) -> np.ndarray:
    """公図の筆界線を描き、その上に塗った図面を半透明で重ねる。大きく描いてから縮めて、線を滑らかにする。"""
    scale = cfg["supersample"]
    west, south, east, north = extent
    final_width = cfg["width_px"]
    final_height = int(round(final_width * (north - south) / (east - west)))
    m_per_px = (east - west) / (final_width * scale)
    size = (final_width * scale, final_height * scale)

    canvas = np.full((size[1], size[0], 3), cfg["background_rgb"], dtype=np.uint8)
    draw_parcels(canvas, geojson, extent, m_per_px, cfg, scale)
    matrix = pixel_to_canvas_matrix(tfw, extent, m_per_px)
    warped = cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR)
    inside = cv2.warpAffine(np.full(image.shape[:2], 255, dtype=np.uint8), matrix, size, flags=cv2.INTER_NEAREST) > 0
    alpha = cfg["drawing_alpha"]
    blended = canvas.astype(np.float32)
    blended[inside] = blended[inside] * (1 - alpha) + warped[inside].astype(np.float32) * alpha
    return cv2.resize(blended.astype(np.uint8), (final_width, final_height), interpolation=cv2.INTER_AREA)


def find_font() -> str:
    candidates = yaml.safe_load(FONT_CONFIG.read_text(encoding="utf-8"))["fonts"]
    path = next((c for c in candidates if Path(c).is_file()), None)
    if path is None:
        raise FileNotFoundError("日本語フォントが見つかりません")
    return path


def add_label(picture: np.ndarray, text: str, cfg: dict) -> Image.Image:
    """右下に文字を入れる。"""
    image = Image.fromarray(picture)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(find_font(), cfg["label_font_px"])
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    pad = cfg["label_padding_px"]
    x1, y1 = image.width - pad, image.height - pad
    x0, y0 = x1 - (right - left) - 2 * pad, y1 - (bottom - top) - 2 * pad
    draw.rectangle((x0, y0, x1, y1), fill=tuple(cfg["label_box_rgb"]))
    draw.text((x0 + pad - left, y0 + pad - top), text, font=font, fill=tuple(cfg["label_text_rgb"]))
    return image


def side_by_side(left: Image.Image, right: Image.Image, gap_px: int, background: list[int]) -> Image.Image:
    joined = Image.new("RGB", (left.width + gap_px + right.width, left.height), tuple(background))
    joined.paste(left, (0, 0))
    joined.paste(right, (left.width + gap_px, 0))
    return joined


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="照合の前後の比較画像を作る")
    parser.add_argument("name", help="図面名（例 dummy_001）")
    parser.add_argument("--before-tfw", type=Path, help="ずらした .tfw。省略すると実験と同じずらし方で作る")
    parser.add_argument("--after-tfw", type=Path, help="照合後の .tfw。省略すると data/masked/<図面>.tfw")
    parser.add_argument("--out", type=Path, default=DOCS)
    args = parser.parse_args(argv)

    cfg = load_config()["compare"]
    tif = MASKED / f"{args.name}.tif"
    image = tifffile.imread(tif)[..., :3]
    size = (image.shape[1], image.shape[0])
    truth = read_world_file(TRUTH / f"{args.name}.tfw")
    before = read_world_file(args.before_tfw) if args.before_tfw else shifted(truth, list(size))
    after = read_world_file(args.after_tfw or MASKED / f"{args.name}.tfw")
    geojson = json.loads((REF / f"{args.name}.geojson").read_text(encoding="utf-8"))

    extent = canvas_extent([before, after], size, cfg["margin_m"])
    before_error = position_error(before, truth, list(size))[0]
    after_error = position_error(after, truth, list(size))[0]
    before_image = add_label(render(image, before, geojson, extent, cfg), f"照合前：ずれ {before_error:.1f}m", cfg)
    after_image = add_label(render(image, after, geojson, extent, cfg), f"照合後：ずれ {after_error:.1f}m", cfg)

    args.out.mkdir(parents=True, exist_ok=True)
    before_image.save(args.out / "georef_before.png", optimize=True)
    after_image.save(args.out / "georef_after.png", optimize=True)
    side_by_side(before_image, after_image, cfg["gap_px"], cfg["background_rgb"]).save(
        args.out / "georef_before_after.png", optimize=True
    )
    before_image.save(
        args.out / "georef_before_after.gif",
        save_all=True,
        append_images=[after_image],
        duration=cfg["gif_frame_ms"],
        loop=0,
        optimize=True,
    )
    print(f"照合前 {before_error:.2f} m、照合後 {after_error:.2f} m → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
