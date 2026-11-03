"""コマンドライン入口（設計書 §2.6）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import tifffile

from .config import load_config
from .detect import detect_chiban
from .evaluate import evaluate
from .mask import write_outputs
from .ocr import run_ocr
from .pair import pair_names

DEFAULT_OUT_DIR = Path("data") / "masked"
DEFAULT_CACHE_DIR = Path("data") / "cache"


def image_size_of(path: Path) -> tuple[int, int]:
    with tifffile.TiffFile(path) as tif:
        shape = tif.pages[0].shape
    return shape[1], shape[0]


def cmd_mask(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    config = load_config(args.config)
    boxes = run_ocr(args.input, config["ocr"], args.cache)
    chiban_boxes = detect_chiban(boxes, image_size_of(args.input), config)
    pairs = pair_names(boxes, chiban_boxes, config["pair"])
    out_tif, _ = write_outputs(args.input, args.out, chiban_boxes, pairs, config["mask"])
    elapsed = time.perf_counter() - started
    print(f"{out_tif}: 文字ボックス {len(boxes)}、地番 {len(chiban_boxes)}、塗り {len(pairs)}（{elapsed:.1f} 秒）")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(truth, result), ensure_ascii=False, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kurokofude", description="図面の氏名だけを手元で塗りつぶす")
    commands = parser.add_subparsers(dest="command", required=True)

    mask = commands.add_parser("mask", help="図面を1枚塗る")
    mask.add_argument("input", type=Path)
    mask.add_argument("--config", type=Path)
    mask.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    mask.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)
    mask.set_defaults(func=cmd_mask)

    score = commands.add_parser("eval", help="正解と比べて3指標を出す")
    score.add_argument("--truth", type=Path, required=True)
    score.add_argument("--result", type=Path, required=True)
    score.set_defaults(func=cmd_eval)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
