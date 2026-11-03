"""コマンドライン入口（設計書 §2.6）。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .config import load_config
from .evaluate import evaluate
from .pipeline import MaskSummary, mask_file

DEFAULT_OUT_DIR = Path("data") / "masked"
DEFAULT_CACHE_DIR = Path("data") / "cache"
TIFF_SUFFIXES = {".tif", ".tiff"}


def _print_summary(summary: MaskSummary, elapsed: float) -> None:
    print(
        f"{summary.out_path}: 文字ボックス {summary.boxes}、地番 {summary.chiban}、"
        f"塗り {summary.masks}（{elapsed:.1f} 秒）"
    )


def cmd_mask(args: argparse.Namespace) -> int:
    started = time.perf_counter()
    summary = mask_file(args.input, load_config(args.config), args.out, args.cache)
    _print_summary(summary, time.perf_counter() - started)
    return 0


def cmd_batch(args: argparse.Namespace) -> int:
    """フォルダ直下の TIFF を順に塗る。1枚失敗しても残りは続ける。"""
    config = load_config(args.config)
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in TIFF_SUFFIXES)
    failed = 0
    for path in paths:
        started = time.perf_counter()
        try:
            summary = mask_file(path, config, args.out, args.cache)
        except Exception as exc:  # 1枚の失敗で全体を止めない
            failed += 1
            print(f"{path.name}: 失敗（{type(exc).__name__}）", file=sys.stderr)
            continue
        _print_summary(summary, time.perf_counter() - started)
    print(f"{len(paths) - failed}/{len(paths)} 枚を処理しました")
    return 1 if failed else 0


def cmd_eval(args: argparse.Namespace) -> int:
    truth = json.loads(args.truth.read_text(encoding="utf-8"))
    result = json.loads(args.result.read_text(encoding="utf-8"))
    print(json.dumps(evaluate(truth, result), ensure_ascii=False, indent=1))
    return 0


def _add_mask_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_DIR)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kurokofude", description="図面の氏名だけを手元で塗りつぶす")
    commands = parser.add_subparsers(dest="command", required=True)

    mask = commands.add_parser("mask", help="図面を1枚塗る")
    mask.add_argument("input", type=Path)
    _add_mask_options(mask)
    mask.set_defaults(func=cmd_mask)

    batch = commands.add_parser("batch", help="フォルダ内の図面をまとめて塗る")
    batch.add_argument("input", type=Path)
    _add_mask_options(batch)
    batch.set_defaults(func=cmd_batch)

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
