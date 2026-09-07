#!/usr/bin/env python3
"""Generate a benzene-vs-top1-molecule figure for each concatenated CSV.

Input pattern:
    dude_50020_concat_all/*.csv
Output pattern:
    fig_top1mol/*.png

Each CSV is expected to contain at least:
    - Reward
    - ReBoltz
    - SMILES

The left panel is benzene with the initial pIC50 for that target.
The right panel is the top-1 molecule by Reward, with pIC50 derived from ReBoltz.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor

INITIAL_PIC50 = {
    "akt1": 4.737,
    "ampc": 3.766,
    "cp3a4": 3.644,
    "cxcr4": 5.099,
    "gcr": 3.803,
    "hivpr": 4.536,
    "hivrt": 3.411,
    "kif11": 3.843,
}

BENZENE_SMILES = "c1ccccc1"
TARGET_PATTERN = re.compile(
    r"(?:^|[_-])(?P<target>akt1|ampc|cp3a4|cxcr4|gcr|hivpr|hivrt|kif11)(?=$|[_-]|\d|\.csv)",
    re.IGNORECASE,
)


def normalize_target(name: str) -> str | None:
    match = TARGET_PATTERN.search(name)
    if not match:
        return None
    return match.group("target").lower()


def reboltz_to_pic50(reboltz: float | str | None) -> float | None:
    if reboltz is None or pd.isna(reboltz):
        return None
    try:
        value = float(reboltz)
    except (TypeError, ValueError):
        return None
    return (6.0 - value) #* 1.364


def top1_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise ValueError("No rows in CSV")
    required = {"Reward", "SMILES", "ReBoltz"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return df.sort_values("Reward", ascending=False).iloc[[0]].reset_index(drop=True)


def draw_top1_figure(target: str, top_df: pd.DataFrame, out_png: Path) -> None:
    top = top_df.iloc[0]
    benzene = Chem.MolFromSmiles(BENZENE_SMILES)
    if benzene is None:
        raise ValueError("Could not parse benzene SMILES")
    rdDepictor.Compute2DCoords(benzene)

    mol = Chem.MolFromSmiles(str(top["SMILES"]))
    if mol is None:
        raise ValueError(f"Invalid SMILES in top-1 row: {top['SMILES']!r}")
    rdDepictor.Compute2DCoords(mol)

    initial_pic50 = INITIAL_PIC50.get(target, float("nan"))
    top_pic50 = reboltz_to_pic50(top["ReBoltz"])
    reward = float(top["Reward"])

    legends = [
        f"Initial compound: benzene\npIC50 = {initial_pic50:.3f}",
        f"Top-1 by Reward\nReward = {reward:.4f}\npIC50 = {top_pic50:.3f}",
    ]

    image = Draw.MolsToGridImage(
        [benzene, mol],
        molsPerRow=2,
        subImgSize=(320, 320),
        legends=legends,
        useSVG=False,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_png)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create benzene vs. top-1 molecule figures for concatenated DUDE CSVs."
    )
    parser.add_argument(
        "input_dir",
        nargs="?",
        type=Path,
        default=Path("dude_50020_concat_all"),
        help="Directory containing concatenated CSV files (default: dude_50020_concat_all)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("fig_top1mol_pic50_upd"),
        help="Output directory for PNG files (default: fig_top1mol_pic50_upd)",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not input_dir.exists():
        fallback = Path("dude_50020_constraint_all")
        if fallback.exists():
            input_dir = fallback.resolve()
        else:
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    for csv_path in csv_paths:
        target = normalize_target(csv_path.name)
        if target is None:
            print(f"[skip] unable to infer target from file name: {csv_path.name}")
            continue

        df = pd.read_csv(csv_path)
        top_df = top1_row(df)
        out_png = output_dir / f"{csv_path.stem}.png"
        draw_top1_figure(target, top_df, out_png)
        print(f"[written] {out_png}")


if __name__ == "__main__":
    main()
