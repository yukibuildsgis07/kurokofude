"""傾いた四角形の共通計算。OCR のボックスは水平化せず、向きと大きさだけを取り出す。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

HALF_TURN_DEG = 180.0
QUARTER_TURN_DEG = 90.0
DEFAULT_ANGLE_DEG = 0.0


@dataclass(frozen=True)
class Frame:
    """四角形に沿った座標軸。u は読み方向、v は図面の傾きに沿った「下」（画像の y が増える側）。"""

    center: np.ndarray
    u: np.ndarray
    v: np.ndarray
    width: float
    height: float
    angle_deg: float


def _extent(points: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    projected = points @ axis
    return float(projected.min()), float(projected.max())


def quad_frame(quad) -> Frame:
    """長辺の向きを角度とする（設計書 §2.3）。"""
    points = np.asarray(quad, dtype=float)
    edges = np.roll(points, -1, axis=0) - points
    longest = edges[int(np.argmax(np.hypot(edges[:, 0], edges[:, 1])))]
    u = longest / np.linalg.norm(longest)
    if u[0] < 0 or (u[0] == 0 and u[1] > 0):
        u = -u
    v = np.array([-u[1], u[0]])
    if v[1] < 0:
        v = -v
    u_min, u_max = _extent(points, u)
    v_min, v_max = _extent(points, v)
    return Frame(
        center=points.mean(axis=0),
        u=u,
        v=v,
        width=u_max - u_min,
        height=v_max - v_min,
        angle_deg=-math.degrees(math.atan2(u[1], u[0])),
    )


def frame_with_angle(quad, angle_deg: float) -> Frame:
    """角度を外から与えた Frame。u, v の向きと幅・高さはその角度で測り直す。"""
    points = np.asarray(quad, dtype=float)
    theta = math.radians(angle_deg)
    u = np.array([math.cos(theta), -math.sin(theta)])
    v = np.array([math.sin(theta), math.cos(theta)])
    u_min, u_max = _extent(points, u)
    v_min, v_max = _extent(points, v)
    return Frame(points.mean(axis=0), u, v, u_max - u_min, v_max - v_min, angle_deg)


class FrameResolver:
    """正方形に近い枠は自分の長辺を信じず、近傍の細長い枠の角度を使う（§11.6）。"""

    def __init__(self, boxes: list[dict], angle_cfg: dict):
        self._square_max = angle_cfg["square_aspect_max"]
        self._neighbors = angle_cfg["neighbors"]
        references = [
            frame for frame in (quad_frame(box["quad"]) for box in boxes)
            if frame.width >= angle_cfg["reference_aspect_min"] * frame.height
        ]
        self._centers = np.array([f.center for f in references]).reshape(-1, 2)
        self._angles = np.array([f.angle_deg for f in references])
        self._overall = float(np.median(self._angles)) if len(references) else DEFAULT_ANGLE_DEG

    def frame(self, quad) -> Frame:
        own = quad_frame(quad)
        if own.width > self._square_max * own.height:
            return own
        if not len(self._angles):
            return frame_with_angle(quad, self._overall)
        distances = np.linalg.norm(self._centers - own.center, axis=1)
        nearest = np.argsort(distances)[: self._neighbors]
        return frame_with_angle(quad, float(np.median(self._angles[nearest])))


def angle_difference_deg(a: float, b: float) -> float:
    """向きの差。文字の長辺は 180 度回しても同じ向き。"""
    diff = abs(a - b) % HALF_TURN_DEG
    return HALF_TURN_DEG - diff if diff > QUARTER_TURN_DEG else diff


def intervals_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] <= b[1] and b[0] <= a[1]


def project_interval(quad, origin: np.ndarray, axis: np.ndarray) -> tuple[float, float]:
    return _extent(np.asarray(quad, dtype=float) - origin, axis)


def clip_below_line(quad, point: np.ndarray, normal: np.ndarray) -> list[list[float]]:
    """(p - point)・normal >= 0 の側だけを残す（Sutherland-Hodgman の半平面クリップ）。"""
    points = np.asarray(quad, dtype=float)
    distances = (points - point) @ normal
    clipped: list[list[float]] = []
    count = len(points)
    for i in range(count):
        current, following = points[i], points[(i + 1) % count]
        d_current, d_following = distances[i], distances[(i + 1) % count]
        if d_current >= 0:
            clipped.append(current.tolist())
        if (d_current >= 0) != (d_following >= 0):
            ratio = d_current / (d_current - d_following)
            clipped.append((current + ratio * (following - current)).tolist())
    return clipped
