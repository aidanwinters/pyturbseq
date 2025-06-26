"""
Script to prepare test datasets for pyturbseq testing

This script creates smaller test datasets from the full datasets for faster testing.
"""

import scanpy as sc
import pandas as pd
import numpy as np
from pathlib import Path
import gzip
import shutil


def prepare_dai2024_molm13_test_data(
    input_path="/large_storage/gilbertlab/aidanw/projects/interactionModel/data/adata/Dai2024_MOLM13.h5ad.gz",
    output_path="tests/data/dai2024_molm13_small.h5ad",
    n_cells_per_perturbation=50,
    n_genes=500
):
    """
    Prepare a small test dataset from Dai2024 MOLM13 data
    """
    print(f"Loading data from {input_path}")
    
    # Load the full dataset
    adata = sc.read_h5ad(input_path)
    
    print(f"Original data shape: {adata.shape}")
    print(f"Perturbations: {adata.obs['feature_call'].value_counts().head()}")
    
    # Subsample cells per perturbation
    sampled_indices = []
    for perturbation in adata.obs['feature_call'].unique():
        pert_indices = adata.obs[adata.obs['feature_call'] == perturbation].index
        n_sample = min(n_cells_per_perturbation, len(pert_indices))
        sampled_indices.extend(
            np.random.choice(pert_indices, size=n_sample, replace=False)
        )
    
    # Subset to sampled cells
    adata_subset = adata[sampled_indices, :].copy()
    
    # Select top variable genes
    if adata_subset.n_vars > n_genes:
        sc.pp.highly_variable_genes(adata_subset, n_top_genes=n_genes)
        adata_subset = adata_subset[:, adata_subset.var.highly_variable].copy()
    
    print(f"Subsampled data shape: {adata_subset.shape}")
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the test dataset
    adata_subset.write(output_path)
    print(f"Saved test dataset to {output_path}")
    
    return adata_subset


def prepare_norman2019_test_data(
    input_path="/large_storage/gilbertlab/aidanw/projects/interactionModel/data/adata/NormanWeissman2019.h5ad.gz",
    output_path="tests/data/norman2019_small.h5ad",
    n_cells_per_perturbation=30,
    n_genes=300
):
    """
    Prepare a small test dataset from Norman2019 data (dual perturbations)
    """
    print(f"Loading data from {input_path}")
    
    # Load the full dataset
    adata = sc.read_h5ad(input_path)
    
    print(f"Original data shape: {adata.shape}")
    print(f"Perturbations: {adata.obs['feature_call'].value_counts().head()}")
    
    # Focus on a subset of perturbations including dual perturbations
    perturbations_to_keep = []
    
    # Get some single perturbations
    single_perts = adata.obs['feature_call'][~adata.obs['feature_call'].str.contains('|', na=False)].unique()
    perturbations_to_keep.extend(single_perts[:5])  # Keep first 5 single perturbations
    
    # Get some dual perturbations
    dual_perts = adata.obs['feature_call'][adata.obs['feature_call'].str.contains('|', na=False)].unique()
    perturbations_to_keep.extend(dual_perts[:10])  # Keep first 10 dual perturbations
    
    # Filter to selected perturbations
    adata_filtered = adata[adata.obs['feature_call'].isin(perturbations_to_keep)].copy()
    
    # Subsample cells per perturbation
    sampled_indices = []
    for perturbation in perturbations_to_keep:
        pert_indices = adata_filtered.obs[adata_filtered.obs['feature_call'] == perturbation].index
        if len(pert_indices) > 0:
            n_sample = min(n_cells_per_perturbation, len(pert_indices))
            sampled_indices.extend(
                np.random.choice(pert_indices, size=n_sample, replace=False)
            )
    
    # Subset to sampled cells
    adata_subset = adata_filtered[sampled_indices, :].copy()
    
    # Select top variable genes
    if adata_subset.n_vars > n_genes:
        sc.pp.highly_variable_genes(adata_subset, n_top_genes=n_genes)
        adata_subset = adata_subset[:, adata_subset.var.highly_variable].copy()
    
    print(f"Subsampled data shape: {adata_subset.shape}")
    
    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the test dataset
    adata_subset.write(output_path)
    print(f"Saved test dataset to {output_path}")
    
    return adata_subset


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)
    
    print("Preparing test datasets...")
    
    # Prepare Dai2024 MOLM13 test data
    try:
        dai_data = prepare_dai2024_molm13_test_data()
        print("✓ Dai2024 MOLM13 test data prepared successfully")
    except Exception as e:
        print(f"✗ Failed to prepare Dai2024 MOLM13 test data: {e}")
    
    # Prepare Norman2019 test data
    try:
        norman_data = prepare_norman2019_test_data()
        print("✓ Norman2019 test data prepared successfully")
    except Exception as e:
        print(f"✗ Failed to prepare Norman2019 test data: {e}")
    
    print("Test data preparation complete!") 