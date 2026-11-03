"""登記所備付地図XMLからダミー図面（.tif + .tfw + .truth.json + .geojson）を作る。

python tools/make_dummy/make.py --source <地図XMLのzipがあるフォルダ> --out <出力先> --count 5
小さな1枚（CIのフィクスチャ用）は --crop 1000 700 --name sample_small --count 1

正解データは --truth-out に分けて書く。氏名の文字列は入れない（設計書 §6）。
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import tifffile
import yaml
from shapely.geometry import Polygon
from shapely.ops import unary_union

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from moj_xml import MapData, Parcel, Sheet, parse_map, read_coordinate_system, read_xml_bytes, sheet_to_geojson  # noqa: E402
from render import Label, find_font_path, parcels_in_view, render_sheet, view_transform  # noqa: E402

DEFAULT_CONFIG = SCRIPT_DIR / "config.yaml"
QUAD_DECIMALS = 2


def iter_map_sources(source_dir: Path, coordinate_system: str, seed: int):
    """座標系が完全に一致する地図XMLだけを、シード固定の順で返す。"""
    zips = sorted(source_dir.rglob("*.zip"))
    random.Random(seed).shuffle(zips)
    for zip_path in zips:
        xml_bytes = read_xml_bytes(zip_path)
        if read_coordinate_system(xml_bytes) == coordinate_system:
            yield xml_bytes


def parcel_area_ratio(sheet: Sheet, data: MapData, chiban_re: re.Pattern) -> float:
    """数字の地番を持つ筆が、図郭（紙）の面積に占める割合。地区外などは数えない（§17）。"""
    frame = Polygon([sheet.corners[key] for key in ("ul", "ur", "lr", "ll")])
    parcels = unary_union(
        [
            data.parcels[pid].polygon.buffer(0)
            for pid in sheet.parcel_ids
            if pid in data.parcels and chiban_re.fullmatch(data.parcels[pid].chiban)
        ]
    )
    return parcels.intersection(frame).area / frame.area if frame.area else 0.0


def pick_sheet(data: MapData, rng: random.Random, source_cfg: dict, chiban_re: re.Pattern) -> Sheet | None:
    candidates = [
        sheet
        for sheet in data.sheets
        if sheet.scale_denominator == source_cfg["scale_denominator"]
        and sum(
            1
            for pid in sheet.parcel_ids
            if pid in data.parcels and chiban_re.fullmatch(data.parcels[pid].chiban)
        )
        >= source_cfg["min_parcels_in_sheet"]
        and parcel_area_ratio(sheet, data, chiban_re) >= source_cfg["min_parcel_area_ratio"]
    ]
    return rng.choice(candidates) if candidates else None


def ensure_short_chiban(parcels: list[Parcel], rng: random.Random, source_cfg: dict, taken: set[str]) -> None:
    """1桁・2桁の地番が足りなければ、他の筆の地番を置き換える（§11.7）。

    taken（図郭にある地番）と重複させない（§18.3）。置き換えた地番も taken に足す。
    """
    for digits, minimum, low, high in (
        (1, source_cfg["min_one_digit"], 1, 9),
        (2, source_cfg["min_two_digit"], 10, 99),
    ):
        shortage = minimum - sum(1 for p in parcels if len(p.chiban) == digits)
        longer = [p for p in parcels if len(p.chiban) > 2]
        for parcel in rng.sample(longer, min(max(shortage, 0), len(longer))):
            free = [str(n) for n in range(low, high + 1) if str(n) not in taken]
            if not free:
                break
            taken.discard(parcel.chiban)
            parcel.chiban = rng.choice(free)
            taken.add(parcel.chiban)


def quad_to_list(quad: np.ndarray) -> list[list[float]]:
    return [[round(float(x), QUAD_DECIMALS), round(float(y), QUAD_DECIMALS)] for x, y in quad]


def truth_json(name: str, size: tuple[int, int], sheet: Sheet, labels: list[Label], rendered=None) -> dict:
    extra = {}
    if rendered is not None:  # 表題欄の各行（作成者名の文字列は書かない）と図郭線（照合の検証用。§18.3、§18.4）
        extra = {
            "title": [{"kind": kind, "quad": quad_to_list(quad)} for kind, quad in rendered.title],
            "frame_quad": quad_to_list(rendered.frame_quad),
        }
    return {
        **extra,
        "source": f"{name}.tif",
        "image_size": list(size),
        "sheet": sheet.number,
        "scale_denominator": sheet.scale_denominator,
        "items": [
            {
                "chiban": {"text": label.chiban, "quad": quad_to_list(label.chiban_quad)},
                "name": {"quad": quad_to_list(label.name_quad)},
                "angle_deg": round(label.angle_deg, QUAD_DECIMALS),
                "placement": label.placement,
                "overlaps_line": label.overlaps_line,
            }
            for label in labels
        ],
    }


def write_outputs(
    out_dir: Path, truth_dir: Path, ref_dir: Path, name: str, rendered, sheet: Sheet, data: MapData, chiban_re: re.Pattern
) -> None:
    for directory in (out_dir, truth_dir, ref_dir):
        directory.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(out_dir / f"{name}.tif", rendered.image, photometric="rgb")
    world_file_text = "\n".join(f"{value:.10f}" for value in rendered.world_file) + "\n"
    (out_dir / f"{name}.tfw").write_text(world_file_text, encoding="ascii")
    (truth_dir / f"{name}.tfw").write_text(world_file_text, encoding="ascii")  # 照合の検証用の正解（§5）
    height, width = rendered.image.shape[:2]
    (truth_dir / f"{name}.truth.json").write_text(
        json.dumps(truth_json(name, (width, height), sheet, rendered.labels, rendered), ensure_ascii=False, indent=1),
        encoding="utf-8",
    )
    geojson = sheet_to_geojson(sheet, data, lambda c: chiban_re.fullmatch(c) is not None)
    (ref_dir / f"{name}.geojson").write_text(json.dumps(geojson, ensure_ascii=False), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ダミー図面を生成する")
    parser.add_argument("--source", type=Path, default=Path("data") / "source")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--truth-out", type=Path, default=Path("data") / "truth")
    parser.add_argument("--ref-out", type=Path, default=Path("data") / "ref")
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--name", default="dummy", help="ファイル名の接頭辞。--count 1 ならそのまま使う")
    parser.add_argument("--crop", type=int, nargs=2, metavar=("W", "H"), help="図郭の一部だけを描く")
    parser.add_argument("--font", help="日本語フォントのパス")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    source_cfg = cfg["source"]
    chiban_re = re.compile(source_cfg["chiban_pattern"])
    font_path = find_font_path(cfg["fonts"], args.font)
    rng = random.Random(args.seed)

    made = 0
    for xml_bytes in iter_map_sources(args.source, source_cfg["coordinate_system"], args.seed):
        if made >= args.count:
            break
        data = parse_map(xml_bytes)
        sheet = pick_sheet(data, rng, source_cfg, chiban_re)
        if sheet is None:
            continue
        parcels = [data.parcels[p] for p in sheet.parcel_ids if p in data.parcels]
        drawable = [p for p in parcels if chiban_re.fullmatch(p.chiban)]
        crop = tuple(args.crop) if args.crop else None
        world_to_px, size = view_transform(sheet, drawable, cfg, crop)
        ensure_short_chiban(
            parcels_in_view(drawable, world_to_px, size),
            random.Random(args.seed + made),
            source_cfg,
            {p.chiban for p in parcels},
        )
        rendered = render_sheet(sheet, parcels, drawable, cfg, font_path, seed=args.seed + made, crop=crop)
        name = args.name if args.count == 1 else f"{args.name}_{made + 1:03d}"
        write_outputs(args.out, args.truth_out, args.ref_out, name, rendered, sheet, data, chiban_re)
        height, width = rendered.image.shape[:2]
        kinds = Counter(label.placement for label in rendered.labels)
        overlaps = sum(1 for label in rendered.labels if label.overlaps_line)
        print(
            f"{name}: {width}x{height}px, 図郭 {sheet.number}, 注記 {len(rendered.labels)} 件"
            f"（水平 {kinds['horizontal']}、回転 {kinds['rotated']}、引き出し線 {kinds['leader']}、線と重なる {overlaps}）"
        )
        made += 1

    if made < args.count:
        print(f"条件に合う図郭が足りません（{made}/{args.count}）", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
