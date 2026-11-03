"""④ 地番の照合でワールドファイルを作る（設計書 §5）。生の図面は使わない。

入力は塗った図面の地番一覧（data/masked/<図面>.chiban.json）と公図（data/ref/<図面>.geojson）だけ。

python tools/georef/georef.py data/masked/<図面>.chiban.json data/ref/<図面>.geojson
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml
from shapely.geometry import Polygon, shape

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.yaml"
HYPHEN = "-"
SIMILARITY_UNKNOWNS = 4
TFW_DECIMALS = 10
REPORT_DECIMALS = 3


@dataclass
class Pair:
    text: str  # 公図で引き当てた地番
    read_as: str  # OCR が読んだ地番
    pixel: tuple[float, float]  # 画素座標（列, 行）
    world: tuple[float, float]  # 筆の重心（東, 北）
    score: float
    reread: bool  # ハイフンを挿して読み替えたか


@dataclass
class Fit:
    params: np.ndarray  # a, b, c, d（東 = a·列 + b·行 + c、北 = b·列 − a·行 + d）
    inliers: list[Pair]
    residuals_m: np.ndarray
    outliers: list[Pair] = field(default_factory=list)


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or DEFAULT_CONFIG).read_text(encoding="utf-8"))


def parcel_shapes(geojson: dict) -> dict:
    """地番 → 筆のポリゴン。同じ地番の筆が複数あれば、どれか決められないので使わない。"""
    counts = Counter(f["properties"]["chiban"] for f in geojson["features"])
    return {
        f["properties"]["chiban"]: shape(f["geometry"]).buffer(0)
        for f in geojson["features"]
        if counts[f["properties"]["chiban"]] == 1
    }


def parcel_centroids(shapes: dict, footprint=None) -> dict[str, tuple[float, float]]:
    """地番 → 筆の重心。footprint（画像が覆う範囲）があれば、筆の見えている部分の重心にする。"""
    centroids = {}
    for text, polygon in shapes.items():
        visible = polygon.intersection(footprint) if footprint is not None else polygon
        if visible.is_empty:
            continue
        point = visible.centroid
        centroids[text] = (point.x, point.y)
    return centroids


def footprint_of(params: np.ndarray, image_size: list[int]) -> Polygon:
    width, height = image_size
    corners = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=float)
    return Polygon(apply(params, corners))


def hyphen_candidates(text: str) -> list[str]:
    """ハイフンが落ちた地番の候補（2739 → 2-739、27-39、273-9）。"""
    if HYPHEN in text or len(text) < 2:
        return []
    return [f"{text[:i]}{HYPHEN}{text[i:]}" for i in range(1, len(text))]


def make_pairs(chiban: list[dict], centroids: dict, config: dict) -> tuple[list[Pair], list[dict]]:
    """地番の文字列で筆を引き当てる。公図にない地番はハイフンを挿した候補で読み替える（§5）。"""
    pairs, unmatched = [], []
    seen: Counter = Counter(item["text"] for item in chiban)
    for item in chiban:
        text = item["text"]
        if seen[text] > 1:  # 図面に同じ地番が2つあれば、どちらか決められない
            unmatched.append({"text": text, "reason": "図面に同じ地番が複数"})
            continue
        if text in centroids:
            pairs.append(Pair(text, text, tuple(item["center"]), centroids[text], item["score"], False))
            continue
        hits = [c for c in hyphen_candidates(text) if c in centroids and seen[c] == 0]
        if len(hits) == 1:
            score = item["score"] * config["reread_score_factor"]
            pairs.append(Pair(hits[0], text, tuple(item["center"]), centroids[hits[0]], score, True))
        else:
            reason = "読み替えの候補が複数" if hits else "公図にない"
            unmatched.append({"text": text, "reason": reason})
    return pairs, unmatched


def design_rows(pixel: np.ndarray, flip_rows: bool) -> tuple[np.ndarray, np.ndarray]:
    """相似変換の最小二乗の係数行列。flip_rows=False は鏡映を許した変換（向きの確認用）。"""
    col, row = pixel[:, 0], pixel[:, 1]
    ones, zeros = np.ones_like(col), np.zeros_like(col)
    if flip_rows:  # 東 = a·列 + b·行 + c、北 = b·列 − a·行 + d
        return np.column_stack([col, row, ones, zeros]), np.column_stack([-row, col, zeros, ones])
    # 行を反転しない（鏡映）：東 = a·列 − b·行 + c、北 = b·列 + a·行 + d
    return np.column_stack([col, -row, ones, zeros]), np.column_stack([row, col, zeros, ones])


def solve(pixel: np.ndarray, world: np.ndarray, flip_rows: bool = True) -> np.ndarray:
    east, north = design_rows(pixel, flip_rows)
    matrix = np.vstack([east, north])
    target = np.concatenate([world[:, 0], world[:, 1]])
    params, *_ = np.linalg.lstsq(matrix, target, rcond=None)
    return params


def apply(params: np.ndarray, pixel: np.ndarray, flip_rows: bool = True) -> np.ndarray:
    east, north = design_rows(pixel, flip_rows)
    return np.column_stack([east @ params, north @ params])


def fit_pairs(pairs: list[Pair], config: dict, flip_rows: bool = True) -> Fit | None:
    """2組から変換を作って一致する組を数え（RANSAC 風）、一致した組で最小二乗をやり直す。"""
    if len(pairs) < config["min_pairs"]:
        return None
    pixel = np.array([p.pixel for p in pairs], dtype=float)
    world = np.array([p.world for p in pairs], dtype=float)
    candidates = list(itertools.combinations(range(len(pairs)), 2))
    rng = random.Random(config["seed"])
    if len(candidates) > config["max_pair_candidates"]:
        candidates = rng.sample(candidates, config["max_pair_candidates"])
    tolerance = config["ransac_tolerance_m"]
    best: np.ndarray | None = None
    best_count = -1
    for i, j in candidates:
        params = solve(pixel[[i, j]], world[[i, j]], flip_rows)
        errors = np.linalg.norm(apply(params, pixel, flip_rows) - world, axis=1)
        count = int((errors <= tolerance).sum())
        if count > best_count:
            best, best_count = errors <= tolerance, count
    if best is None or best_count < config["min_pairs"]:
        return None
    for _ in range(len(pairs)):  # 一致した組で解き直し、一致の範囲が変わらなくなるまで繰り返す
        params = solve(pixel[best], world[best], flip_rows)
        errors = np.linalg.norm(apply(params, pixel, flip_rows) - world, axis=1)
        updated = errors <= tolerance
        if updated.sum() < config["min_pairs"] or (updated == best).all():
            break
        best = updated
    params = solve(pixel[best], world[best], flip_rows)
    errors = np.linalg.norm(apply(params, pixel, flip_rows) - world, axis=1)
    inliers = [p for p, ok in zip(pairs, best) if ok]
    outliers = [p for p, ok in zip(pairs, best) if not ok]
    return Fit(params, inliers, errors[best], outliers)


def world_file(params: np.ndarray) -> list[float]:
    """.tfw の並び（A, D, B, E, C, F）。東 = A·列 + B·行 + C、北 = D·列 + E·行 + F。"""
    a, b, c, d = params
    return [a, b, b, -a, c, d]


def georeference(chiban_json: dict, geojson: dict, config: dict) -> dict:
    shapes = parcel_shapes(geojson)
    centroids = parcel_centroids(shapes)
    pairs, unmatched = make_pairs(chiban_json["chiban"], centroids, config)
    fit = fit_pairs(pairs, config)
    # 図郭の端の筆は、重心が画像の外にあるのに地番は見えている部分に書かれる。解けた変換で
    # 画像が覆う範囲を出し、筆をその範囲で切った重心で解き直す
    for _ in range(config["footprint_iterations"] if fit else 0):
        visible = parcel_centroids(shapes, footprint_of(fit.params, chiban_json["image_size"]))
        pairs = [
            Pair(p.text, p.read_as, p.pixel, visible[p.text], p.score, p.reread) for p in pairs if p.text in visible
        ]
        refit = fit_pairs(pairs, config)
        if refit is None:
            break
        fit = refit
    mirrored = fit_pairs(pairs, config, flip_rows=False)
    report: dict = {
        "pairs_found": len(pairs),
        "unmatched": unmatched,
        "rereadings": [{"read_as": p.read_as, "matched": p.text} for p in pairs if p.reread],
    }
    if fit is None:
        report["status"] = "failed"
        report["message"] = f"一致するペアが {config['min_pairs']} 組に足りません"
        return report
    a, b = fit.params[0], fit.params[1]
    report.update(
        {
            "status": "ok",
            "world_file": world_file(fit.params),
            "pairs_used": len(fit.inliers),
            "pixel_size_m": round(float(np.hypot(a, b)), 6),
            "rotation_deg": round(float(np.degrees(np.arctan2(b, a))), REPORT_DECIMALS),
            "residual_rms_m": round(float(np.sqrt(np.mean(fit.residuals_m**2))), REPORT_DECIMALS),
            "residual_max_m": round(float(fit.residuals_m.max()), REPORT_DECIMALS),
            "residuals": [
                {"chiban": p.text, "read_as": p.read_as, "residual_m": round(float(r), REPORT_DECIMALS), "score": round(p.score, 3)}
                for p, r in sorted(zip(fit.inliers, fit.residuals_m), key=lambda x: -x[1])
            ],
            "excluded": [p.text for p in fit.outliers],
            # 向きの確認：鏡映を許した変換のほうが合うなら、行の向きを取り違えている
            "mirrored_residual_rms_m": (
                round(float(np.sqrt(np.mean(mirrored.residuals_m**2))), REPORT_DECIMALS) if mirrored else None
            ),
            "mirrored_pairs_used": len(mirrored.inliers) if mirrored else 0,
        }
    )
    return report


def write_world_file(path: Path, values: list[float]) -> None:
    path.write_text("\n".join(f"{v:.{TFW_DECIMALS}f}" for v in values) + "\n", encoding="ascii")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="地番の照合でワールドファイルを作る")
    parser.add_argument("chiban_json", type=Path, help="data/masked/<図面>.chiban.json")
    parser.add_argument("geojson", type=Path, help="data/ref/<図面>.geojson")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--no-write", action="store_true", help=".tfw を書かずに結果だけ出す")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    chiban_json = json.loads(args.chiban_json.read_text(encoding="utf-8"))
    report = georeference(chiban_json, json.loads(args.geojson.read_text(encoding="utf-8")), config)
    stem = args.chiban_json.name.removesuffix(".chiban.json")
    if report["status"] == "ok" and not args.no_write:
        tfw = args.chiban_json.with_name(f"{stem}.tfw")
        write_world_file(tfw, report["world_file"])
        report["tfw"] = tfw.as_posix()
    report_path = args.chiban_json.with_name(f"{stem}.georef.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k not in {"residuals", "world_file"}}, ensure_ascii=False, indent=1))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
