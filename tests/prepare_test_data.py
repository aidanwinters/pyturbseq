"""
Script to prepare test datasets for pyturbseq testing

This script creates smaller test datasets from the full datasets for faster testing.

Note: This script references local data paths and is included for transparency
and reproducibility. End users don't need to run this - the generated small
datasets are included directly in the repository.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc


def prepare_norman2019_singles_test_data(
    input_path=(
        "/large_storage/gilbertlab/aidanw/projects/interactionModel/"
        "data/adata/NormanWeissman2019.h5ad.gz"
    ),
    output_path="tests/data/norman2019_subset_singles.h5ad.gz",
    n_control=500,
    n_top_genes=1000,
):
    """
    Prepare a small test dataset from Norman2019 data with only single perturbations
    """
    print(f"Loading data from {input_path}")

    # Load the full dataset
    adata = sc.read_h5ad(input_path, backed="r")

    # Subset perturbations
    control = "control"
    targets = ["SET", "KLF1", "SAMD1", "PTPN12", "COL2A1"]
    perts = ["control"] + targets

    # Filter to selected perturbations and gemgroup 1
    singles = (
        adata[adata.obs["perturbation"].isin(perts) & (adata.obs["gemgroup"] == 1)]
        .to_memory()
        .copy()
    )
    singles.obs = singles.obs[["perturbation", "nperts"]].copy()

    print(f"Original singles data shape: {singles.shape}")
    print(f"Perturbations: {singles.obs['perturbation'].value_counts()}")

    # Subset control conditions to specified number
    control_idx = singles.obs.index[singles.obs["perturbation"] == control]
    to_remove_n = len(control_idx) - n_control

    if to_remove_n > 0:
        control_remove_idx = singles.obs.loc[control_idx].sample(to_remove_n).index
        singles = singles[singles.obs.index.difference(control_remove_idx), :].copy()

    # Store raw counts and normalize for gene selection
    singles.layers["counts"] = singles.X.copy()
    sc.pp.normalize_total(singles, exclude_highly_expressed=True)
    sc.pp.log1p(singles)
    sc.pp.highly_variable_genes(singles, n_top_genes=n_top_genes)

    # Select genes: highly variable + target genes
    genes = np.unique(singles.var.query("highly_variable").index.tolist() + targets)
    assert pd.Series(genes).isin(targets).sum() == len(
        targets
    ), "Not all target genes found"

    # Subset to selected genes and restore counts
    singles = singles[:, genes].copy()
    singles.X = singles.layers["counts"]
    singles.var = singles.var.drop(columns=singles.var.columns)

    # Clean up metadata
    singles.uns = {}
    del singles.layers["counts"]

    print(f"Final singles data shape: {singles.shape}")

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the test dataset
    singles.write_h5ad(output_path)
    print(f"Saved singles test dataset to {output_path}")

    return singles


def prepare_norman2019_doubles_test_data(
    input_path=(
        "/large_storage/gilbertlab/aidanw/projects/interactionModel/"
        "data/adata/NormanWeissman2019.h5ad.gz"
    ),
    output_path="tests/data/norman2019_subset_doubles.h5ad.gz",
    n_control=500,
    n_top_genes=1000,
):
    """
    Prepare a small test dataset from Norman2019 data with single and dual perturbations
    """
    print(f"Loading data from {input_path}")

    # Load the full dataset
    adata = sc.read_h5ad(input_path, backed="r")

    # Subset perturbations
    control = "control"
    targets = ["SET", "KLF1", "SAMD1", "PTPN12", "COL2A1"]
    dual_perts = ["SET_KLF1", "SAMD1_PTPN12", "KLF1_COL2A1"]
    perts = ["control"] + targets + dual_perts

    # Filter to selected perturbations and gemgroup 1
    doubles = (
        adata[adata.obs["perturbation"].isin(perts) & (adata.obs["gemgroup"] == 1)]
        .to_memory()
        .copy()
    )
    doubles.obs = doubles.obs[["perturbation", "nperts"]].copy()

    print(f"Original doubles data shape: {doubles.shape}")
    print(f"Perturbations: {doubles.obs['perturbation'].value_counts()}")

    # Subset control conditions to specified number
    control_idx = doubles.obs.index[doubles.obs["perturbation"] == control]
    to_remove_n = len(control_idx) - n_control

    if to_remove_n > 0:
        control_remove_idx = doubles.obs.loc[control_idx].sample(to_remove_n).index
        doubles = doubles[doubles.obs.index.difference(control_remove_idx), :].copy()

    # Store raw counts and normalize for gene selection
    doubles.layers["counts"] = doubles.X.copy()
    sc.pp.normalize_total(doubles, exclude_highly_expressed=True)
    sc.pp.log1p(doubles)
    sc.pp.highly_variable_genes(doubles, n_top_genes=n_top_genes)

    # Select genes: highly variable + target genes
    genes = np.unique(doubles.var.query("highly_variable").index.tolist() + targets)
    assert pd.Series(genes).isin(targets).sum() == len(
        targets
    ), "Not all target genes found"

    # Subset to selected genes and restore counts
    doubles = doubles[:, genes].copy()
    doubles.X = doubles.layers["counts"]
    doubles.var = doubles.var.drop(columns=doubles.var.columns)

    # Clean up metadata
    doubles.uns = {}
    del doubles.layers["counts"]

    print(f"Final doubles data shape: {doubles.shape}")

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save the test dataset
    doubles.write_h5ad(output_path)
    print(f"Saved doubles test dataset to {output_path}")

    return doubles


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    print("Preparing test datasets...")
    print("Note: This script uses local data paths not available to end users.")
    print("The generated datasets are included in the repository for testing.\n")

    # Prepare Norman2019 singles test data
    try:
        singles_data = prepare_norman2019_singles_test_data()
        print("✓ Norman2019 singles test data prepared successfully")
    except Exception as e:
        print(f"✗ Failed to prepare Norman2019 singles test data: {e}")

    # Prepare Norman2019 doubles test data
    try:
        doubles_data = prepare_norman2019_doubles_test_data()
        print("✓ Norman2019 doubles test data prepared successfully")
    except Exception as e:
        print(f"✗ Failed to prepare Norman2019 doubles test data: {e}")

    print("\nTest data preparation complete!")
    print("Generated files:")
    print("- tests/data/norman2019_subset_singles.h5ad.gz")
    print("- tests/data/norman2019_subset_doubles.h5ad.gz")
