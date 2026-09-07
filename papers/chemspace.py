#!/usr/bin/env python3
"""Generate chemspace PCA/KDE figures for each CSV in dude_50020_constraint_all.

Workflow:
- Sample 10,000 molecules from zinc_250k.smi with random.Random(42)
- Convert them to ECFP4 (2048 bits) and fit PCA(n_components=2)
- For each CSV in the target directory, transform all valid SMILES into the same
  PCA space and plot a red KDE for the Zinc reference set and a blue KDE for the
  target molecules
- Plot benzene (c1ccccc1) as a red cross and the top 10 reward-scoring molecules
  as blue dots
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from sklearn.decomposition import PCA
from tqdm import tqdm


DEFAULT_SEED = 42
DEFAULT_REF_COUNT = 10_000
DEFAULT_N_BITS = 2048
BENZENE_SMILES = "c1ccccc1"


def parse_smiles_file(path: Path) -> list[str]:
    """Read a SMILES file and return non-empty lines."""
    with path.open("r", encoding="utf-8") as handle:
        return [line.strip() for line in handle if line.strip() and not line.startswith("#")]


def sample_reference_smiles(path: Path, n: int = DEFAULT_REF_COUNT, seed: int = DEFAULT_SEED) -> list[str]:
    """Randomly sample n SMILES from the reference library."""
    smiles = parse_smiles_file(path)
    if len(smiles) <= n:
        return smiles
    rng = random.Random(seed)
    return rng.sample(smiles, n)


def smiles_to_ecfp4_2048(smiles: str) -> np.ndarray | None:
    """Convert a SMILES string to a 2048-bit ECFP4 vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=DEFAULT_N_BITS)
    arr = np.zeros(DEFAULT_N_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(float)


def build_ecfp_matrix(smiles_list: list[str]) -> tuple[list[str], np.ndarray]:
    """Build a 2D matrix of valid ECFP4 vectors for a list of SMILES."""
    valid_smiles: list[str] = []
    vectors: list[np.ndarray] = []
    for smiles in smiles_list:
        if not isinstance(smiles, str):
            continue
        cleaned = smiles.strip()
        if not cleaned:
            continue
        vec = smiles_to_ecfp4_2048(cleaned)
        if vec is None:
            continue
        valid_smiles.append(cleaned)
        vectors.append(vec)

    if not vectors:
        return [], np.empty((0, DEFAULT_N_BITS), dtype=float)
    return valid_smiles, np.vstack(vectors)


def fit_pca(reference_vectors: np.ndarray) -> PCA:
    """Fit a 2-component PCA model using the reference vectors."""
    if reference_vectors.shape[0] < 2:
        raise ValueError("At least 2 reference molecules are required to fit PCA.")
    model = PCA(n_components=2, random_state=DEFAULT_SEED)
    model.fit(reference_vectors)
    return model


def transform_with_pca(model: PCA, vectors: np.ndarray) -> np.ndarray:
    """Project vectors into the learned 2D PCA space."""
    if vectors.size == 0:
        return np.empty((0, 2), dtype=float)
    return model.transform(vectors)


def plot_chemspace(
    ref_xy: np.ndarray,
    target_xy: np.ndarray,
    benzene_xy: np.ndarray,
    top10_xy: np.ndarray,
    out_png: Path,
    csv_name: str,
) -> None:
    """Create a chemspace KDE plot with red reference and blue target distributions."""
    out_png.parent.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    fig, ax = plt.subplots(figsize=(7, 6), dpi=180)

    if ref_xy.shape[0] > 1:
        sns.kdeplot(
            x=ref_xy[:, 0],
            y=ref_xy[:, 1],
            cmap="Reds",
            fill=True,
            alpha=0.45,
            levels=20,
            thresh=0.05,
            ax=ax,
            warn_singular=False,
            label="Zinc random 10k",
        )
    if target_xy.shape[0] > 1:
        sns.kdeplot(
            x=target_xy[:, 0],
            y=target_xy[:, 1],
            cmap="Blues",
            fill=True,
            alpha=0.45,
            levels=20,
            thresh=0.05,
            ax=ax,
            warn_singular=False,
            label="Generated molecules",
        )

    if benzene_xy.shape[0] > 0:
        ax.scatter(
            benzene_xy[0, 0],
            benzene_xy[0, 1],
            marker="x",
            s=120,
            color="red",
            linewidths=2,
            label="benzene (c1ccccc1)",
        )

    """ if top10_xy.shape[0] > 0:
        ax.scatter(
            top10_xy[:, 0],
            top10_xy[:, 1],
            s=32,
            color="blue",
            edgecolors="black",
            linewidths=0.5,
            alpha=0.9,
            label="Top 10 reward",
        ) """

    tgt_name = csv_name.replace("concat_", "").rsplit("_", 1)[0] if "_" in csv_name else csv_name
    ax.set_title(f"Chemspace PCA (ECFP4 2048) - {tgt_name}")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.legend(loc="best")
    plt.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def process_csv(csv_path: Path, pca_model: PCA, ref_xy: np.ndarray, reference_path: Path) -> None:
    """Process one CSV file and save its chemspace plot."""
    df = pd.read_csv(csv_path)
    df.columns = [str(col).strip() for col in df.columns]

    if "SMILES" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a SMILES column.")
    if "Reward" not in df.columns:
        raise ValueError(f"{csv_path} does not contain a Reward column.")

    df = df[["SMILES", "Reward"]].copy()
    df["Reward"] = pd.to_numeric(df["Reward"], errors="coerce")
    df = df.dropna(subset=["Reward"]).copy()
    df["SMILES"] = df["SMILES"].astype(str).str.strip()
    df = df[df["SMILES"] != "nan"]

    valid_smiles, vectors = build_ecfp_matrix(df["SMILES"].tolist())
    if valid_smiles:
        df_valid = df[df["SMILES"].isin(valid_smiles)].copy()
        df_valid = df_valid.sort_values("Reward", ascending=False).reset_index(drop=True)
        target_xy = transform_with_pca(pca_model, vectors)
    else:
        target_xy = np.empty((0, 2), dtype=float)
        df_valid = df.iloc[0:0].copy()

    benzene_vec = smiles_to_ecfp4_2048(BENZENE_SMILES)
    benzene_xy = np.empty((0, 2), dtype=float)
    if benzene_vec is not None:
        benzene_xy = transform_with_pca(pca_model, benzene_vec.reshape(1, -1))

    top10 = df_valid.head(10)
    top10_xy = np.empty((0, 2), dtype=float)
    if not top10.empty:
        top10_smiles = top10["SMILES"].tolist()
        _, top10_vectors = build_ecfp_matrix(top10_smiles)
        if top10_vectors.size > 0:
            top10_xy = transform_with_pca(pca_model, top10_vectors)

    out_png = Path("fig_chemspace") / f"{csv_path.stem}.png"
    plot_chemspace(ref_xy, target_xy, benzene_xy, top10_xy, out_png, csv_path.stem)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot chemspace PCA distribution for each target CSV.")
    ap.add_argument("--input-dir", type=str, default="dude_50020_constraint_all", help="Directory containing the CSV files to plot.")
    ap.add_argument("--reference", type=str, default="zinc_250k.smi", help="Reference SMILES file used for the random sampling and PCA fitting.")
    ap.add_argument("--out-dir", type=str, default="fig_chemspace", help="Directory to save the PNG outputs.")
    ap.add_argument("--sample-size", type=int, default=DEFAULT_REF_COUNT, help="Number of Zinc molecules to randomly sample for PCA fitting.")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed used for sampling the reference set.")
    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    reference_path = Path(args.reference)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference file not found: {reference_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    sampled_smiles = sample_reference_smiles(reference_path, n=args.sample_size, seed=args.seed)
    _, reference_vectors = build_ecfp_matrix(sampled_smiles)
    if reference_vectors.size == 0:
        raise ValueError(f"No valid reference molecules were found in {reference_path}.")

    pca_model = fit_pca(reference_vectors)
    ref_xy = transform_with_pca(pca_model, reference_vectors)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}.")

    for csv_path in tqdm(csv_files, desc="Processing CSV files"):
        process_csv(csv_path, pca_model, ref_xy, reference_path)

    print(f"Saved chemspace plots for {len(csv_files)} CSV files under {out_dir}..")


if __name__ == "__main__":
    main()
