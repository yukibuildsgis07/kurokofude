"""筆と架空氏名を描いて、スキャン風に加工する。"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import nearest_points, unary_union
from shapely.prepared import prep

from moj_xml import Parcel, Sheet

MM_PER_INCH = 25.4
MM_PER_METER_AT_SCALE_1 = 1000.0
FULL_INK = 255
TEXT_PATCH_PADDING_PX = 3
MAX_MIN_RECT_SIDES = 4
HALF_TURN_DEG = 180
QUARTER_TURN_DEG = 90
SHORT_CHIBAN_MAX_LEN = 2
HORIZONTAL_DEG = 0.0
PLACEMENT_HORIZONTAL = "horizontal"
PLACEMENT_ROTATED = "rotated"
PLACEMENT_LEADER = "leader"


@dataclass
class Label:
    chiban: str
    name: str
    angle_deg: float
    placement: str  # horizontal / rotated / leader
    chiban_quad: np.ndarray
    name_quad: np.ndarray
    overlaps_line: bool = False
    leader: np.ndarray | None = None  # 引き出し線の両端（筆の内側の点、文字枠の端）


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


def fake_name(rng: random.Random, names: dict) -> str:
    return rng.choice(names["family"]) + "　" + rng.choice(names["given"])


def search_offsets(step: int, rings: int):
    yield np.zeros(2)
    for ring in range(1, rings + 1):
        for k in range(8 * ring):
            theta = 2 * math.pi * k / (8 * ring)
            yield np.array([math.cos(theta), math.sin(theta)]) * step * ring


@dataclass
class _Plan:
    """1つの注記の中身。位置はまだ決まっていない。"""

    parcel: Parcel
    polygon: Polygon  # 画素座標の筆（画像内に切り抜いたもの）
    anchor: np.ndarray  # 筆の内側の点
    name: str
    long_axis_deg: float  # 筆の長軸の角度
    chiban_size: tuple[float, float]
    name_size: tuple[float, float]


class _Layout:
    """置いた注記と引き出し線を覚えて、空いているかを判定する。"""

    def __init__(self, size: tuple[int, int], keep_out: Polygon, lines, clearance: float, gap_px: float):
        self.canvas = box(0, 0, size[0], size[1]).buffer(-clearance)
        self.taken: list = [keep_out]
        self.lines = prep(lines)
        self.clearance = clearance
        self.gap_px = gap_px

    def quads(self, plan: _Plan, center: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
        _, down = text_vectors(angle)
        to_name = down * (plan.chiban_size[1] / 2 + self.gap_px + plan.name_size[1] / 2)
        chiban_center = center - to_name / 2
        return (
            text_quad(chiban_center, *plan.chiban_size, angle),
            text_quad(chiban_center + to_name, *plan.name_size, angle),
        )

    def block(self, chiban_quad: np.ndarray, name_quad: np.ndarray):
        return Polygon(chiban_quad).union(Polygon(name_quad)).buffer(self.clearance)

    def is_free(self, block) -> bool:
        return self.canvas.contains(block) and all(not block.intersects(t) for t in self.taken)

    def crosses_line(self, block) -> bool:
        return self.lines.intersects(block)


def _make_plan(parcel, world_to_px, canvas, chiban_font, name_font, rng, names) -> _Plan | None:
    polygon_px = Polygon(apply_affine(world_to_px, np.array(parcel.polygon.exterior.coords)))
    if not polygon_px.is_valid:
        polygon_px = polygon_px.buffer(0)
    visible = polygon_px.intersection(canvas)
    if visible.is_empty or visible.geom_type != "Polygon":
        return None
    point = visible.representative_point()
    name = fake_name(rng, names)
    return _Plan(
        parcel,
        visible,
        np.array([point.x, point.y]),
        name,
        parcel_angle(polygon_px),
        text_extent(chiban_font, parcel.chiban),
        text_extent(name_font, name),
    )


def _place_inside(plan: _Plan, layout: _Layout, step: int, angle: float, placement: str) -> Label | None:
    """筆の内側で、線にも他の注記にも重ならない位置を代表点に近い順に探す。"""
    min_x, min_y, max_x, max_y = plan.polygon.bounds
    grid = np.array(
        [(x, y) for y in np.arange(min_y, max_y, step) for x in np.arange(min_x, max_x, step)], dtype=float
    ).reshape(-1, 2)
    points = np.vstack([plan.anchor, grid[np.argsort(np.linalg.norm(grid - plan.anchor, axis=1))]])
    inside = prep(plan.polygon)
    for center in points:
        chiban_quad, name_quad = layout.quads(plan, center, angle)
        block = layout.block(chiban_quad, name_quad)
        if inside.contains(block) and not layout.crosses_line(block) and layout.is_free(block):
            layout.taken.append(block)
            return Label(plan.parcel.chiban, plan.name, angle, placement, chiban_quad, name_quad)
    return None


def _place_with_leader(plan: _Plan, layout: _Layout, step: int, rings: int) -> Label | None:
    """筆の外の空いた場所に水平で置き、筆の内側の点から文字枠の端まで引き出し線を引く。"""
    for offset in search_offsets(step, rings):
        chiban_quad, name_quad = layout.quads(plan, plan.anchor + offset, HORIZONTAL_DEG)
        block = layout.block(chiban_quad, name_quad)
        if layout.crosses_line(block) or not layout.is_free(block):
            continue
        text = Polygon(chiban_quad).union(Polygon(name_quad))
        end = nearest_points(Point(plan.anchor), text)[1]
        leader = LineString([plan.anchor, (end.x, end.y)])
        if any(leader.intersects(t) for t in layout.taken):
            continue
        layout.taken.extend([block, leader.buffer(layout.clearance)])
        return Label(
            plan.parcel.chiban,
            plan.name,
            HORIZONTAL_DEG,
            PLACEMENT_LEADER,
            chiban_quad,
            name_quad,
            leader=np.array(leader.coords),
        )
    return None


def _place_over_line(plan: _Plan, layout: _Layout, step: int, rings: int) -> Label | None:
    """どうしても置けないときだけ、水平で線と重ねて置く。"""
    for offset in search_offsets(step, rings):
        chiban_quad, name_quad = layout.quads(plan, plan.anchor + offset, HORIZONTAL_DEG)
        block = layout.block(chiban_quad, name_quad)
        if layout.is_free(block):
            layout.taken.append(block)
            return Label(
                plan.parcel.chiban,
                plan.name,
                HORIZONTAL_DEG,
                PLACEMENT_HORIZONTAL,
                chiban_quad,
                name_quad,
                overlaps_line=layout.crosses_line(block),
            )
    return None


def layout_labels(
    parcels: list[Parcel],
    world_to_px: np.ndarray,
    size: tuple[int, int],
    keep_out: Polygon,
    lines,
    chiban_font: ImageFont.FreeTypeFont,
    name_font: ImageFont.FreeTypeFont,
    gap_px: float,
    rng: random.Random,
    render_cfg: dict,
    names: dict,
) -> list[Label]:
    """全筆で水平を試し、余った筆を筆の長軸に回し、次に引き出し線、最後に線との重なりを許す（§14）。"""
    canvas = box(0, 0, size[0], size[1])
    layout = _Layout(size, keep_out, lines, render_cfg["label_clearance_px"], gap_px)
    plans = [
        plan
        for plan in (_make_plan(p, world_to_px, canvas, chiban_font, name_font, rng, names) for p in parcels)
        if plan is not None
    ]
    inside_step = render_cfg["inside_search_step_px"]
    leader_step = render_cfg["leader_search_step_px"]
    leader_rings = render_cfg["leader_max_distance_px"] // leader_step
    stages = [
        lambda plan: _place_inside(plan, layout, inside_step, HORIZONTAL_DEG, PLACEMENT_HORIZONTAL),
        lambda plan: _place_inside(plan, layout, inside_step, plan.long_axis_deg, PLACEMENT_ROTATED),
        lambda plan: _place_with_leader(plan, layout, leader_step, leader_rings),
        lambda plan: _place_over_line(
            plan, layout, render_cfg["placement_search_step_px"], render_cfg["placement_search_rings"]
        ),
    ]
    labels: list[Label] = []
    for place in stages:
        remaining = []
        for plan in plans:
            label = place(plan)
            if label is None:
                remaining.append(plan)
            else:
                labels.append(label)
        plans = remaining
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


def draw_title_block(ink: np.ndarray, font_of, sheet: Sheet, date: str, cfg: dict, frame_px: int) -> Polygon:
    """右下に表題欄を描く。大きさは除外領域に揃え、文字は枠に入るように小さくする。"""
    height, width = ink.shape
    padding = cfg["padding_px"]
    x0 = int(width * (1 - cfg["width_ratio"])) + padding
    y0 = int(height * (1 - cfg["height_ratio"])) + padding
    x1, y1 = width - padding, height - padding
    cv2.rectangle(ink, (x0, y0), (x1, y1), 0, cv2.FILLED)  # 表題欄の下の筆界線は隠す
    cv2.rectangle(ink, (x0, y0), (x1, y1), FULL_INK, frame_px)

    lines = [f"S = 1:{sheet.scale_denominator}", f"図郭番号 {sheet.number}", f"作成 {date}"]
    inner_width, inner_height = x1 - x0 - 2 * padding, y1 - y0 - 2 * padding
    size = cfg["font_px"]
    while size > 1:
        font = font_of(size)
        if (
            max(text_extent(font, line)[0] for line in lines) <= inner_width
            and size * cfg["line_ratio"] * len(lines) <= inner_height
        ):
            break
        size -= 1
    line_height = size * cfg["line_ratio"]
    center_x, center_y = (x0 + x1) / 2, (y0 + y1) / 2
    for index, line in enumerate(lines):
        offset = (index - (len(lines) - 1) / 2) * line_height
        stamp_text(ink, font, line, np.array([center_x, center_y + offset]), 0.0)
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


def view_transform(
    sheet: Sheet, drawable: list[Parcel], cfg: dict, crop: tuple[int, int] | None
) -> tuple[np.ndarray, tuple[int, int]]:
    """描く範囲の world → pixel の行列と画像サイズ。crop があれば筆の多い場所を切り出す。"""
    render_cfg = cfg["render"]
    dpi = render_cfg["dpi"]
    px_per_m = pixels_per_meter(dpi, sheet.scale_denominator)
    margin = mm_to_px(render_cfg["margin_mm"], dpi)
    world_to_px, size = sheet_transform(sheet, px_per_m, margin)
    if crop is None:
        return world_to_px, size
    x, y = densest_crop_origin(drawable, world_to_px, size, crop, render_cfg["crop_stride_px"])
    world_to_px = world_to_px.copy()
    world_to_px[:, 2] -= (x, y)
    return world_to_px, crop


def parcels_in_view(parcels: list[Parcel], world_to_px: np.ndarray, size: tuple[int, int]) -> list[Parcel]:
    result = []
    for parcel in parcels:
        point = parcel.polygon.representative_point()
        x, y = apply_affine(world_to_px, np.array([[point.x, point.y]]))[0]
        if 0 <= x < size[0] and 0 <= y < size[1]:
            result.append(parcel)
    return result


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

    world_to_px, size = view_transform(sheet, drawable, cfg, crop)
    width, height = size

    ink = np.zeros((height, width), dtype=np.uint8)
    rings = []
    for parcel in parcels:
        ring = apply_affine(world_to_px, np.array(parcel.polygon.exterior.coords))
        rings.append(LineString(ring))
        cv2.polylines(
            ink, [np.round(ring).astype(np.int32)], True, FULL_INK, render_cfg["line_width_px"], cv2.LINE_AA
        )
    frame = apply_affine(world_to_px, np.array([sheet.corners[k] for k in ("ul", "ur", "lr", "ll")]))
    rings.append(LineString(np.vstack([frame, frame[:1]])))
    cv2.polylines(ink, [np.round(frame).astype(np.int32)], True, FULL_INK, render_cfg["frame_width_px"], cv2.LINE_AA)

    chiban_font = ImageFont.truetype(str(font_path), int(round(mm_to_px(render_cfg["chiban_height_mm"], dpi))))
    name_font = ImageFont.truetype(str(font_path), int(round(mm_to_px(render_cfg["name_height_mm"], dpi))))
    title_font_of = lambda px: ImageFont.truetype(str(font_path), px)  # noqa: E731

    keep_out = draw_title_block(ink, title_font_of, sheet, rng.choice(title_cfg["dates"]), title_cfg, render_cfg["frame_width_px"])
    shuffled = list(drawable)
    rng.shuffle(shuffled)
    shuffled.sort(key=lambda p: len(p.chiban) > SHORT_CHIBAN_MAX_LEN)  # 短い地番を先に置いて確実に残す
    labels = layout_labels(
        shuffled,
        world_to_px,
        size,
        keep_out,
        unary_union(rings),
        chiban_font,
        name_font,
        mm_to_px(render_cfg["name_gap_mm"], dpi),
        rng,
        render_cfg,
        cfg["names"],
    )
    for label in labels:
        if label.leader is not None:
            start, end = np.round(label.leader).astype(int).tolist()
            cv2.line(ink, tuple(start), tuple(end), FULL_INK, render_cfg["leader_width_px"], cv2.LINE_AA)
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
