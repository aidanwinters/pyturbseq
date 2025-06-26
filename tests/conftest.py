"""
Pytest configuration and fixtures for pyturbseq tests
"""

import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
import tempfile
import os


@pytest.fixture
def small_test_adata():
    """Create a small test AnnData object for testing"""
    np.random.seed(42)
    
    # Create synthetic data
    n_obs = 100
    n_vars = 50
    
    # Gene expression data
    X = np.random.negative_binomial(5, 0.3, size=(n_obs, n_vars))
    
    # Cell metadata
    obs = pd.DataFrame({
        'cell_id': [f'cell_{i}' for i in range(n_obs)],
        'feature_call': ['NTC'] * 20 + ['GENE1'] * 20 + ['GENE2'] * 20 + 
                       ['GENE1|GENE2'] * 20 + ['GENE3'] * 20,
        'n_genes_by_counts': np.random.randint(1000, 5000, n_obs),
        'total_counts': np.random.randint(5000, 20000, n_obs),
        'pct_counts_mt': np.random.uniform(0, 20, n_obs),
        'batch': np.random.choice(['batch1', 'batch2'], n_obs)
    })
    
    # Gene metadata
    var = pd.DataFrame({
        'gene_ids': [f'ENSG{i:08d}' for i in range(n_vars)],
        'feature_types': ['Gene Expression'] * n_vars,
        'highly_variable': np.random.choice([True, False], n_vars)
    })
    var.index = [f'GENE_{i}' for i in range(n_vars)]
    
    # Create AnnData object
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obs.index = adata.obs['cell_id']
    
    return adata


@pytest.fixture
def dual_perturbation_adata():
    """Create test data specifically for dual perturbation analysis"""
    np.random.seed(123)
    
    n_obs = 200
    n_vars = 30
    
    X = np.random.negative_binomial(3, 0.4, size=(n_obs, n_vars))
    
    # Create systematic dual perturbation design
    perturbations = []
    genes = ['GENE1', 'GENE2', 'GENE3', 'GENE4']
    
    # Single perturbations
    for gene in genes:
        perturbations.extend([gene] * 20)
    
    # Dual perturbations
    for i, gene1 in enumerate(genes):
        for j, gene2 in enumerate(genes[i+1:], i+1):
            perturbations.extend([f'{gene1}|{gene2}'] * 15)
    
    # Controls
    n_controls = n_obs - len(perturbations)
    perturbations.extend(['NTC'] * n_controls)
    
    obs = pd.DataFrame({
        'cell_id': [f'cell_{i}' for i in range(n_obs)],
        'feature_call': perturbations[:n_obs],
        'n_genes_by_counts': np.random.randint(800, 4000, n_obs),
        'total_counts': np.random.randint(3000, 15000, n_obs)
    })
    
    var = pd.DataFrame({
        'gene_ids': [f'ENSG{i:08d}' for i in range(n_vars)],
        'feature_types': ['Gene Expression'] * n_vars
    })
    var.index = [f'TARGET_{i}' for i in range(n_vars)]
    
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obs.index = adata.obs['cell_id']
    
    return adata


@pytest.fixture
def temp_h5ad_file(small_test_adata):
    """Create a temporary h5ad file for testing file I/O"""
    with tempfile.NamedTemporaryFile(suffix='.h5ad', delete=False) as tmp:
        small_test_adata.write(tmp.name)
        yield tmp.name
    os.unlink(tmp.name)


@pytest.fixture
def sample_perturbation_matrix():
    """Create a sample perturbation matrix for testing"""
    np.random.seed(42)
    
    cell_ids = [f'cell_{i}' for i in range(50)]
    genes = ['GENE1', 'GENE2', 'GENE3', 'GENE4']
    
    # Create binary perturbation matrix
    matrix = np.random.choice([0, 1], size=(50, 4), p=[0.7, 0.3])
    
    return pd.DataFrame(matrix, index=cell_ids, columns=genes) 