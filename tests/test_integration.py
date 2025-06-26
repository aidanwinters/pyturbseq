"""
Integration tests for pyturbseq using real datasets

These tests use actual perturbation screen data to ensure the package works
with real-world data.
"""

import pytest
import scanpy as sc
import numpy as np
import pandas as pd
from pathlib import Path
import pyturbseq as pts


@pytest.mark.integration
@pytest.mark.slow
class TestRealDataIntegration:
    """Integration tests using real perturbation screen datasets"""
    
    def test_dai2024_molm13_basic_workflow(self):
        """Test basic workflow with Dai2024 MOLM13 data"""
        data_path = Path("tests/data/dai2024_molm13_small.h5ad")
        
        if not data_path.exists():
            pytest.skip("Test data not available. Run tests/prepare_test_data.py first.")
        
        # Load test data
        adata = sc.read_h5ad(data_path)
        
        # Basic checks
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert 'feature_call' in adata.obs.columns
        
        # Test perturbation matrix generation
        pts.utils.get_perturbation_matrix(
            adata, 
            perturbation_col='feature_call',
            inplace=True
        )
        
        assert 'perturbation' in adata.obsm
        assert adata.obsm['perturbation'].shape[0] == adata.n_obs
        
        # Test target change calculation if target genes are available
        if 'target_gene' in adata.var.columns:
            pts.utils.calculate_target_change(
                adata,
                perturbation_column='feature_call',
                inplace=True
            )
            assert 'target_change' in adata.obs.columns
    
    def test_norman2019_dual_perturbation_workflow(self):
        """Test dual perturbation workflow with Norman2019 data"""
        data_path = Path("tests/data/norman2019_small.h5ad")
        
        if not data_path.exists():
            pytest.skip("Test data not available. Run tests/prepare_test_data.py first.")
        
        # Load test data
        adata = sc.read_h5ad(data_path)
        
        # Basic checks
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert 'feature_call' in adata.obs.columns
        
        # Check for dual perturbations
        dual_perturbs = adata.obs['feature_call'].str.contains('|', na=False)
        assert dual_perturbs.sum() > 0, "No dual perturbations found in test data"
        
        # Test perturbation matrix generation
        pts.utils.get_perturbation_matrix(
            adata, 
            perturbation_col='feature_call',
            inplace=True
        )
        
        assert 'perturbation' in adata.obsm
        assert adata.obsm['perturbation'].shape[0] == adata.n_obs
        
        # Test that dual perturbations are properly encoded
        pert_matrix = adata.obsm['perturbation']
        dual_cells = adata.obs[dual_perturbs].index
        
        # Dual perturbation cells should have sum > 1 in perturbation matrix
        dual_sums = pert_matrix.loc[dual_cells].sum(axis=1)
        assert (dual_sums > 1).any(), "Dual perturbations not properly encoded"


@pytest.mark.integration
class TestPackageImports:
    """Test that all package modules can be imported correctly"""
    
    def test_import_main_modules(self):
        """Test importing all main modules"""
        import pyturbseq
        
        # Test that all expected modules are available
        expected_modules = ['utils', 'de', 'interaction', 'plot', 'calling', 'cellranger', 'guides']
        
        for module_name in expected_modules:
            assert hasattr(pyturbseq, module_name), f"Module {module_name} not available"
            
        # Test version is available
        assert hasattr(pyturbseq, '__version__')
        assert pyturbseq.__version__ is not None
    
    def test_import_key_functions(self):
        """Test that key functions are available"""
        from pyturbseq.utils import generate_perturbation_matrix, calculate_target_change
        from pyturbseq.utils import filter_adata, pseudobulk
        
        # Test that functions are callable
        assert callable(generate_perturbation_matrix)
        assert callable(calculate_target_change)
        assert callable(filter_adata)
        assert callable(pseudobulk)


@pytest.mark.integration
class TestDataCompatibility:
    """Test compatibility with different data formats and structures"""
    
    def test_anndata_compatibility(self, small_test_adata):
        """Test that package works with standard AnnData objects"""
        # Test basic operations
        pts.utils.get_perturbation_matrix(
            small_test_adata,
            perturbation_col='feature_call',
            inplace=True
        )
        
        assert 'perturbation' in small_test_adata.obsm
        
    def test_missing_columns_handling(self, small_test_adata):
        """Test handling of missing required columns"""
        # Remove feature_call column
        adata_no_feature = small_test_adata.copy()
        del adata_no_feature.obs['feature_call']
        
        # Should raise an error or handle gracefully
        with pytest.raises((KeyError, ValueError)):
            pts.utils.get_perturbation_matrix(
                adata_no_feature,
                perturbation_col='feature_call',
                inplace=True
            ) 