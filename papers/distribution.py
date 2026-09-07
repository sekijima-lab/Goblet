import argparse
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


METRICS = {
    "Reward": "Reward",
    "ReBoltz": "ReBoltz",
    "ReVina": "ReVina",
}

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


def reboltz_to_pic50(values: pd.Series) -> pd.Series:
    converted = pd.to_numeric(values, errors="coerce")
    return (6.0 - converted) #* 1.364


def target_name_from_csv(csv_path: Path) -> str:
    stem = csv_path.stem.lower()
    if stem.startswith("concat_"):
        stem = stem[len("concat_") :]
    if "_" in stem:
        stem = stem.rsplit("_", 1)[0]
    return stem


def load_all_csvs(input_dir: Path) -> tuple[list[Path], pd.DataFrame]:
    csv_paths = sorted(input_dir.glob("*.csv"))
    if not csv_paths:
        raise FileNotFoundError(f"No CSV files found under: {input_dir}")

    frames = []
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)
    return csv_paths, combined


def make_histogram_plot(
    values: pd.Series,
    metric_name: str,
    out_png: Path,
    bins: int = 40,
    x_label: Optional[str] = None,
    reference_value: Optional[float] = None,
    reference_label: Optional[str] = None,
) -> None:
    series = pd.to_numeric(values, errors="coerce").dropna()
    if series.empty:
        raise ValueError(f"No valid numeric values found for {metric_name}.")

    fig, ax = plt.subplots(figsize=(7, 5), dpi=200)
    ax.hist(series.to_numpy(), bins=bins, color="#4C72B0", edgecolor="black", alpha=0.8)
    ax.set_title(f"{metric_name} histogram\nN={len(series):,}")
    ax.set_xlabel(x_label if x_label is not None else metric_name)
    ax.set_ylabel("Count")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    if reference_value is not None:
        ax.axvline(
            reference_value,
            color="red",
            linestyle="--",
            linewidth=2,
            label=reference_label or f"Initial pIC50={reference_value:.3f}",
        )
        ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_combined_figure(combined: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), dpi=200)
    for ax, (metric_label, column_name) in zip(axes, METRICS.items()):
        if metric_label == "ReBoltz":
            series = reboltz_to_pic50(combined[column_name]).dropna()
            x_label = "pIC50"
            title = "pIC50 histogram"
        else:
            series = pd.to_numeric(combined[column_name], errors="coerce").dropna()
            x_label = metric_label
            title = f"{metric_label} histogram"

        if series.empty:
            raise ValueError(f"No valid numeric values found for {column_name}.")
        ax.hist(series.to_numpy(), bins=40, color="#55A868", edgecolor="black", alpha=0.85)
        ax.set_title(f"{title} (N={len(series):,})")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Count")
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)


def make_single_csv_histograms(csv_path: Path, out_dir: Path, bins: int = 40) -> list[Path]:
    df = pd.read_csv(csv_path)
    missing = [col for col in METRICS.values() if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path.name}: missing required columns {missing}")

    target = target_name_from_csv(csv_path)
    initial_pic50 = INITIAL_PIC50.get(target)

    saved = []
    for metric_label, column_name in METRICS.items():
        values = pd.to_numeric(df[column_name], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"{csv_path.name}: no valid numeric values in {column_name}")

        if metric_label == "ReBoltz":
            values = reboltz_to_pic50(df[column_name])
            values = values.dropna()
            x_label = "pIC50"
            reference_value = initial_pic50
            reference_label = f"Initial pIC50={initial_pic50:.3f}" if initial_pic50 is not None else None
        else:
            x_label = metric_label
            reference_value = None
            reference_label = None

        out_png = out_dir / f"{csv_path.stem}_{metric_label.lower()}_hist.png"
        make_histogram_plot(
            values,
            f"{metric_label} (pIC50)" if metric_label == "ReBoltz" else metric_label,
            out_png,
            bins=bins,
            x_label=x_label,
            reference_value=reference_value,
            reference_label=reference_label,
        )
        saved.append(out_png)
    return saved


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Plot histograms of Reward, ReBoltz, and ReVina for each CSV in a given input directory."
        )
    )
    ap.add_argument(
        "--input_dir",
        type=Path,
        default=Path("dude_50020_constraint_all"),
        help="Directory containing CSV files such as concat_*.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path("fig_objective_space_pic50_upd"),
        help="Directory to save histogram PNG outputs",
    )
    ap.add_argument("--bins", type=int, default=40, help="Number of histogram bins")
    args = ap.parse_args()

    input_dir = args.input_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    for old_png in out_dir.glob("*.png"):
        old_png.unlink()

    csv_paths, combined = load_all_csvs(input_dir)

    saved_figs = []
    for csv_path in csv_paths:
        saved_figs.extend(make_single_csv_histograms(csv_path, out_dir, bins=args.bins))

    combined_png = out_dir / "objective_space_histograms.png"
    make_combined_figure(combined, combined_png)
    saved_figs.append(combined_png)

    print(f"Generated {len(saved_figs)} PNG files from {len(csv_paths)} CSVs.")
    print(f"Per-CSV histograms: {len(csv_paths) * len(METRICS)} PNGs")
    print(f"Saved under: {out_dir}")


if __name__ == "__main__":
    main()
