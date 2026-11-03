"""④ の検証：わざとずらした .tfw から始め、地番の照合で正しい位置に戻るかを m 単位で測る。

python tools/georef/experiment.py dummy_001 dummy_002 ...
正解は data/truth/<図面>.tfw。ずらした .tfw を data/masked/<図面>.tfw に書いてから照合で上書きする。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from georef import georeference, load_config, write_world_file  # noqa: E402

MASKED = Path("data") / "masked"
TRUTH = Path("data") / "truth"
REF = Path("data") / "ref"
GRID = 5  # 画像を GRID×GRID の点で比べる
SHIFT_EAST_M = 35.0
SHIFT_NORTH_M = -25.0
SHIFT_ROTATION_DEG = 2.0
SHIFT_SCALE = 1.03


def read_world_file(path: Path) -> np.ndarray:
    return np.array([float(v) for v in path.read_text(encoding="ascii").split()])


def to_world(tfw: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    a, d, b, e, c, f = tfw
    return np.column_stack([a * pixel[:, 0] + b * pixel[:, 1] + c, d * pixel[:, 0] + e * pixel[:, 1] + f])


def grid(size: list[int]) -> np.ndarray:
    xs = np.linspace(0, size[0] - 1, GRID)
    ys = np.linspace(0, size[1] - 1, GRID)
    return np.array([(x, y) for y in ys for x in xs])


def position_error(tfw: np.ndarray, truth: np.ndarray, size: list[int]) -> tuple[float, float]:
    """画像全体の格子点で、正解の位置とのずれ（RMS、最大）。m 単位。"""
    points = grid(size)
    errors = np.linalg.norm(to_world(tfw, points) - to_world(truth, points), axis=1)
    return float(np.sqrt(np.mean(errors**2))), float(errors.max())


def shifted(truth: np.ndarray, size: list[int]) -> np.ndarray:
    """正解を、画像の中心まわりに回して拡大し、平行移動した .tfw（わざとずらした位置）。"""
    a, d, b, e, c, f = truth
    center = np.array([(size[0] - 1) / 2, (size[1] - 1) / 2])
    world_center = to_world(truth, center[None])[0]
    theta = math.radians(SHIFT_ROTATION_DEG)
    rotation = SHIFT_SCALE * np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    linear = rotation @ np.array([[a, b], [d, e]])
    new_center = world_center + np.array([SHIFT_EAST_M, SHIFT_NORTH_M])
    offset = new_center - linear @ center
    return np.array([linear[0, 0], linear[1, 0], linear[0, 1], linear[1, 1], offset[0], offset[1]])


def run(name: str, config: dict, no_frame: bool = False) -> dict:
    chiban_path = MASKED / f"{name}.chiban.json"
    chiban_json = json.loads(chiban_path.read_text(encoding="utf-8"))
    size = chiban_json["image_size"]
    truth = read_world_file(TRUTH / f"{name}.tfw")
    start = shifted(truth, size)
    write_world_file(MASKED / f"{name}.tfw", list(start))
    before = position_error(start, truth, size)

    truth_json = json.loads((TRUTH / f"{name}.truth.json").read_text(encoding="utf-8"))
    frame = None if no_frame else truth_json.get("frame_quad")
    report = georeference(chiban_json, json.loads((REF / f"{name}.geojson").read_text(encoding="utf-8")), config, frame)
    row = {"name": name, "before_rms_m": round(before[0], 2), "before_max_m": round(before[1], 2)}
    if report["status"] != "ok":
        return {**row, "status": report["status"]}
    result = np.array(report["world_file"])
    write_world_file(MASKED / f"{name}.tfw", list(result))
    after = position_error(result, truth, size)
    truth_rotation = math.degrees(math.atan2(truth[1], truth[0]))
    return {
        **row,
        "status": "ok",
        "after_rms_m": round(after[0], 2),
        "after_max_m": round(after[1], 2),
        "pairs_found": report["pairs_found"],
        "pairs_used": report["pairs_used"],
        "residual_rms_m": report["residual_rms_m"],
        "mirrored_pairs_used": report["mirrored_pairs_used"],
        "scale_error_pct": round(100 * (report["pixel_size_m"] / math.hypot(truth[0], truth[1]) - 1), 2),
        "rotation_error_deg": round(report["rotation_deg"] - truth_rotation, 3),
        "rereadings": report["rereadings"],
        "unmatched": report["unmatched"],
        "excluded": report["excluded"],
        "chosen_by_prediction": report["chosen_by_prediction"],
        "iterations": len(report["iterations"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="照合の検証")
    parser.add_argument("names", nargs="+")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--max-iterations", type=int, help="0 で筆全体の重心だけ（§5 のまま）")
    parser.add_argument("--no-frame", action="store_true", help="図郭線を使わず、画像の範囲だけで切る")
    parser.add_argument("--irls", choices=["huber", "none"], help="残差で重みを決める反復の方法")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.max_iterations is not None:
        config["max_iterations"] = args.max_iterations
    if args.irls:
        config["irls"] = {**config["irls"], "method": args.irls}
    for name in args.names:
        print(json.dumps(run(name, config, args.no_frame), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
