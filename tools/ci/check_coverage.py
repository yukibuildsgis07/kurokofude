"""CI 用：kurokofude eval を実行し、氏名カバー率がしきい値未満なら終了コード 1 にする。

python tools/ci/check_coverage.py --min 90 --truth <truth.json> --result <chiban.json> --image <塗る前.tif>
シェルのリダイレクトを使わないので、Windows の PowerShell でも同じように動く。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

REPORTED_KEYS = ("name_coverage_pct", "overmask_pct", "chiban_survival_pct", "chiban_detection_pct", "masks")


def main() -> int:
    parser = argparse.ArgumentParser(description="氏名カバー率のしきい値を確かめる")
    parser.add_argument("--min", type=float, required=True, help="氏名カバー率の下限（%）")
    parser.add_argument("--truth", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    completed = subprocess.run(
        [sys.executable, "-m", "kurokofude.cli", "eval", "--truth", args.truth, "--result", args.result, "--image", args.image],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    scores = json.loads(completed.stdout)
    for key in REPORTED_KEYS:
        print(f"{key}: {scores[key]}")
    coverage = scores["name_coverage_pct"]
    if coverage < args.min:
        print(f"氏名カバー率 {coverage}% がしきい値 {args.min}% を下回りました", file=sys.stderr)
        return 1
    print(f"氏名カバー率 {coverage}% はしきい値 {args.min}% 以上です")
    return 0


if __name__ == "__main__":
    sys.exit(main())
