"""筆と架空氏名を描いて、スキャン風に加工する。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon, box

from moj_xml import Parcel, Sheet

MM_PER_INCH = 25.4
MM_PER_METER_AT_SCALE_1 = 1000.0
FULL_INK = 255
TEXT_PATCH_PADDING_PX = 3
MAX_MIN_RECT_SIDES = 4
HALF_TURN_DEG = 180
QUARTER_TURN_DEG = 90


@dataclass
class Label:
    chiban: str
    name: str
    angle_deg: float
    slanted: bool
    chiban_quad: np.ndarray
    name_quad: np.ndarray


@dataclass
class Rendered:
    image: np.ndarray  # RGB
    labels: list[Label]
    world_file: list[float]  # A, D, B, E, C, F


def pixels_per_meter(dpi: float, scale_denominator: int) -> float:
    return dpi / MM_PER_INCH * MM_PER_METER_AT_SCALE_1 / scale_denominator


def mm_to_px(mm: float, dpi: float) -> float:
    return mm * dpi / MM_PER_INCH


def find_font_path(candidates: list[str], override: str | None) -> Path:
    for candidate in ([override] if override else candidates):
        path = Path(candidate)
        if path.is_file():
            return path
    raise FileNotFoundError("日本語フォントが見つかりません。--font で指定してください")


def sheet_transform(sheet: Sheet, px_per_m: float, margin_px: float) -> tuple[np.ndarray, tuple[int, int]]:
    """図郭の四隅から world (east, north) → pixel の 2×3 行列と画像サイズを求める。"""
    c = sheet.corners
    width_m = math.dist(c["ul"], c["ur"])
    height_m = math.dist(c["ul"], c["ll"])
    width = int(round(width_m * px_per_m + 2 * margin_px))
    height = int(round(height_m * px_per_m + 2 * margin_px))
    src = np.array([c["ul"], c["ur"], c["ll"]], dtype=np.float64)
    dst = np.array(
        [
            [margin_px, margin_px],
            [width - margin_px, margin_px],
            [margin_px, height - margin_px],
        ],
        dtype=np.float64,
    )
    return solve_affine(src, dst), (width, height)


def solve_affine(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """3点対応の厳密解。float32 に落とす cv2.getAffineTransform は公共座標だと桁落ちする。"""
    design = np.hstack([src, np.ones((3, 1))])
    return np.linalg.solve(design, dst).T


def apply_affine(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ matrix[:, :2].T + matrix[:, 2]


def compose(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    full = lambda m: np.vstack([m, [0, 0, 1]])  # noqa: E731
    return (full(outer) @ full(inner))[:2]


def world_file_from(world_to_pixel: np.ndarray) -> list[float]:
    """pixel → world の係数を .tfw の並び（A, D, B, E, C, F）で返す。(0,0) は左上画素の中心。"""
    inverse = np.linalg.inv(np.vstack([world_to_pixel, [0, 0, 1]]))
    return [
        float(inverse[0, 0]),
        float(inverse[1, 0]),
        float(inverse[0, 1]),
        float(inverse[1, 1]),
        float(inverse[0, 2]),
        float(inverse[1, 2]),
    ]


def text_vectors(angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """画像座標での読み方向 u と、文字の下方向 v。angle は見た目で反時計回りが正。"""
    theta = math.radians(angle_deg)
    return np.array([math.cos(theta), -math.sin(theta)]), np.array([math.sin(theta), math.cos(theta)])


def text_extent(font: ImageFont.FreeTypeFont, text: str) -> tuple[float, float]:
    left, top, right, bottom = font.getbbox(text)
    return float(right - left), float(bottom - top)


def text_quad(center: np.ndarray, width: float, height: float, angle_deg: float) -> np.ndarray:
    """文字の外接の傾いた四角形。読み順で 左上・右上・右下・左下。"""
    u, v = text_vectors(angle_deg)
    half_u, half_v = u * width / 2, v * height / 2
    return np.array(
        [center - half_u - half_v, center + half_u - half_v, center + half_u + half_v, center - half_u + half_v]
    )


def stamp_text(
    ink: np.ndarray, font: ImageFont.FreeTypeFont, text: str, center: np.ndarray, angle_deg: float
) -> None:
    left, top, right, bottom = font.getbbox(text)
    width, height = right - left, bottom - top
    pad = TEXT_PATCH_PADDING_PX
    patch = Image.new("L", (width + 2 * pad, height + 2 * pad), 0)
    ImageDraw.Draw(patch).text((pad - left, pad - top), text, fill=FULL_INK, font=font)
    patch_array = np.asarray(patch)

    u, v = text_vectors(angle_deg)
    patch_center = np.array([(width + 2 * pad - 1) / 2, (height + 2 * pad - 1) / 2])
    matrix = np.array(
        [
            [u[0], v[0], center[0] - u[0] * patch_center[0] - v[0] * patch_center[1]],
            [u[1], v[1], center[1] - u[1] * patch_center[0] - v[1] * patch_center[1]],
        ]
    )
    warped = cv2.warpAffine(patch_array, matrix, (ink.shape[1], ink.shape[0]), flags=cv2.INTER_LINEAR)
    np.maximum(ink, warped, out=ink)


def parcel_angle(polygon_px: Polygon) -> float:
    """筆の最小外接矩形の長辺の向き（-90〜90度に畳む）。"""
    coords = np.array(polygon_px.minimum_rotated_rectangle.exterior.coords)[: MAX_MIN_RECT_SIDES + 1]
    edges = np.diff(coords, axis=0)
    longest = edges[np.argmax(np.hypot(edges[:, 0], edges[:, 1]))]
    angle = -math.degrees(math.atan2(longest[1], longest[0]))
    while angle > QUARTER_TURN_DEG:
        angle -= HALF_TURN_DEG
    while angle <= -QUARTER_TURN_DEG:
        angle += HALF_TURN_DEG
    return angle


def choose_angle(polygon_px: Polygon, rng: random.Random, cfg: dict) -> tuple[float, bool]:
    if rng.random() >= cfg["slanted_ratio"]:
        return 0.0, False
    angle = parcel_angle(polygon_px)
    magnitude = abs(angle)
    if magnitude > cfg["slant_max_deg"] or magnitude < cfg["slant_min_deg"]:
        # 筆の向きが読める範囲にないときは、範囲内の角度を振る（文字を倒しすぎない）
        magnitude = rng.uniform(cfg["slant_min_deg"], cfg["slant_max_deg"])
        angle = magnitude if rng.random() < 0.5 else -magnitude
    return angle, True


def fake_name(rng: random.Random, names: dict) -> str:
    return rng.choice(names["family"]) + "　" + rng.choice(names["given"])


def search_offsets(step: int, rings: int):
    yield np.zeros(2)
    for ring in range(1, rings + 1):
        for k in range(8 * ring):
            theta = 2 * math.pi * k / (8 * ring)
            yield np.array([math.cos(theta), math.sin(theta)]) * step * ring


def layout_labels(
    parcels: list[Parcel],
    world_to_px: np.ndarray,
    size: tuple[int, int],
    keep_out: Polygon,
    chiban_font: ImageFont.FreeTypeFont,
    name_font: ImageFont.FreeTypeFont,
    gap_px: float,
    rng: random.Random,
    render_cfg: dict,
    names: dict,
) -> list[Label]:
    width, height = size
    canvas = box(0, 0, width, height)
    taken: list[Polygon] = [keep_out]
    labels: list[Label] = []
    clearance = render_cfg["label_clearance_px"]

    for parcel in parcels:
        polygon_px = Polygon(apply_affine(world_to_px, np.array(parcel.polygon.exterior.coords)))
        if not polygon_px.is_valid:
            polygon_px = polygon_px.buffer(0)
        visible = polygon_px.intersection(canvas)
        if visible.is_empty or visible.geom_type != "Polygon":
            continue
        point = visible.representative_point()
        anchor = np.array([point.x, point.y])

        name = fake_name(rng, names)
        angle, slanted = choose_angle(polygon_px, rng, render_cfg)
        _, down = text_vectors(angle)
        chiban_w, chiban_h = text_extent(chiban_font, parcel.chiban)
        name_w, name_h = text_extent(name_font, name)
        to_name = down * (chiban_h / 2 + gap_px + name_h / 2)

        for offset in search_offsets(render_cfg["placement_search_step_px"], render_cfg["placement_search_rings"]):
            chiban_center = anchor + offset - to_name / 2
            name_center = chiban_center + to_name
            chiban_quad = text_quad(chiban_center, chiban_w, chiban_h, angle)
            name_quad = text_quad(name_center, name_w, name_h, angle)
            block = Polygon(chiban_quad).union(Polygon(name_quad)).buffer(clearance)
            inside = canvas.buffer(-clearance).contains(block)
            if inside and all(not block.intersects(t) for t in taken):
                taken.append(block)
                labels.append(Label(parcel.chiban, name, angle, slanted, chiban_quad, name_quad))
                break
    return labels


def scanify(ink: np.ndarray, rng: np.random.Generator, cfg: dict) -> np.ndarray:
    """インク層と紙からスキャン風の RGB を合成する。傾きは呼び出し側で適用済み。"""
    height, width = ink.shape
    grid = cfg["fade_grid"]
    fade = rng.uniform(cfg["fade_min"], 1.0, size=(grid, grid)).astype(np.float32)
    fade = cv2.resize(fade, (width, height), interpolation=cv2.INTER_CUBIC)
    alpha = np.clip(ink.astype(np.float32) / FULL_INK * fade, 0, 1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), cfg["blur_sigma"])[..., None]

    paper = np.array(cfg["paper_color_rgb"], dtype=np.float32)
    paper = paper + rng.uniform(-cfg["paper_jitter"], cfg["paper_jitter"], size=3)
    ink_color = np.array(cfg["ink_color_rgb"], dtype=np.float32)
    image = paper * (1 - alpha) + ink_color * alpha
    image += rng.normal(0, cfg["noise_sigma"], size=(height, width, 1)).astype(np.float32)
    return np.clip(image, 0, FULL_INK).astype(np.uint8)


def draw_title_block(
    ink: np.ndarray, font: ImageFont.FreeTypeFont, sheet: Sheet, date: str, cfg: dict, frame_px: int
) -> Polygon:
    """右下に表題欄を描く。傾ける前の座標で描くので、全体の傾きと一緒に回る。"""
    height, width = ink.shape
    x1 = width - cfg["padding_px"]
    y1 = height - cfg["padding_px"]
    x0 = x1 - cfg["width_px"]
    y0 = y1 - cfg["height_px"]
    cv2.rectangle(ink, (x0, y0), (x1, y1), 0, cv2.FILLED)  # 表題欄の下の筆界線は隠す
    cv2.rectangle(ink, (x0, y0), (x1, y1), FULL_INK, frame_px)
    lines = [f"S = 1:{sheet.scale_denominator}", f"図郭番号 {sheet.number}", f"作成 {date}"]
    for index, line in enumerate(lines):
        x = x0 + cfg["font_px"] // 2
        y = y0 + cfg["font_px"] // 2 + index * cfg["line_px"]
        text_w, text_h = text_extent(font, line)
        center = np.array([x + text_w / 2, y + text_h / 2])
        stamp_text(ink, font, line, center, 0.0)
    return box(x0, y0, x1, y1)


def densest_crop_origin(
    parcels: list[Parcel], world_to_px: np.ndarray, size: tuple[int, int], crop: tuple[int, int], stride: int
) -> tuple[int, int]:
    """筆の代表点が最も多く入る切り出し位置を探す。"""
    points = np.array(
        [[p.polygon.representative_point().x, p.polygon.representative_point().y] for p in parcels]
    )
    pixels = apply_affine(world_to_px, points)
    best, best_count = (0, 0), -1
    for y in range(0, max(1, size[1] - crop[1] + 1), stride):
        for x in range(0, max(1, size[0] - crop[0] + 1), stride):
            inside = (
                (pixels[:, 0] >= x)
                & (pixels[:, 0] < x + crop[0])
                & (pixels[:, 1] >= y)
                & (pixels[:, 1] < y + crop[1])
            )
            count = int(inside.sum())
            if count > best_count:
                best, best_count = (x, y), count
    return best


def render_sheet(
    sheet: Sheet,
    parcels: list[Parcel],
    drawable: list[Parcel],
    cfg: dict,
    font_path: Path,
    seed: int,
    crop: tuple[int, int] | None = None,
) -> Rendered:
    render_cfg, scan_cfg, title_cfg = cfg["render"], cfg["scan"], cfg["title_block"]
    dpi = render_cfg["dpi"]
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    px_per_m = pixels_per_meter(dpi, sheet.scale_denominator)
    margin = mm_to_px(render_cfg["margin_mm"], dpi)
    world_to_px, size = sheet_transform(sheet, px_per_m, margin)
    if crop is not None:
        x, y = densest_crop_origin(drawable, world_to_px, size, crop, render_cfg["crop_stride_px"])
        world_to_px = world_to_px.copy()
        world_to_px[:, 2] -= (x, y)
        size = crop
    width, height = size

    ink = np.zeros((height, width), dtype=np.uint8)
    for parcel in parcels:
        ring = apply_affine(world_to_px, np.array(parcel.polygon.exterior.coords))
        cv2.polylines(
            ink, [np.round(ring).astype(np.int32)], True, FULL_INK, render_cfg["line_width_px"], cv2.LINE_AA
        )
    frame = apply_affine(world_to_px, np.array([sheet.corners[k] for k in ("ul", "ur", "lr", "ll")]))
    cv2.polylines(ink, [np.round(frame).astype(np.int32)], True, FULL_INK, render_cfg["frame_width_px"], cv2.LINE_AA)

    chiban_font = ImageFont.truetype(str(font_path), int(round(mm_to_px(render_cfg["chiban_height_mm"], dpi))))
    name_font = ImageFont.truetype(str(font_path), int(round(mm_to_px(render_cfg["name_height_mm"], dpi))))
    title_font = ImageFont.truetype(str(font_path), title_cfg["font_px"])

    keep_out = draw_title_block(ink, title_font, sheet, rng.choice(title_cfg["dates"]), title_cfg, render_cfg["frame_width_px"])
    shuffled = list(drawable)
    rng.shuffle(shuffled)
    labels = layout_labels(
        shuffled,
        world_to_px,
        size,
        keep_out,
        chiban_font,
        name_font,
        mm_to_px(render_cfg["name_gap_mm"], dpi),
        rng,
        render_cfg,
        cfg["names"],
    )
    for label in labels:
        stamp_text(ink, chiban_font, label.chiban, label.chiban_quad.mean(axis=0), label.angle_deg)
        stamp_text(ink, name_font, label.name, label.name_quad.mean(axis=0), label.angle_deg)

    skew_deg = rng.uniform(-scan_cfg["skew_max_deg"], scan_cfg["skew_max_deg"])
    skew = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), skew_deg, 1.0)
    ink = cv2.warpAffine(ink, skew, (width, height), flags=cv2.INTER_LINEAR)
    for label in labels:
        label.chiban_quad = apply_affine(skew, label.chiban_quad)
        label.name_quad = apply_affine(skew, label.name_quad)
        label.angle_deg += skew_deg

    return Rendered(
        image=scanify(ink, np_rng, scan_cfg),
        labels=labels,
        world_file=world_file_from(compose(skew, world_to_px)),
    )
