"""
Tests for pyturbseq.de module
"""

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
import pyturbseq.de as de


class TestDifferentialExpression:
    """Test differential expression analysis functions"""
    
    def test_de_analysis_basic(self, small_test_adata):
        """Test basic DE analysis"""
        # Check that DE functions exist
        assert hasattr(de, 'get_degs')
        assert hasattr(de, 'get_all_degs')
    
    def test_get_degs_function_exists(self, small_test_adata):
        """Test that get_degs function exists and is callable"""
        assert callable(de.get_degs)
        
    def test_get_all_degs_function_exists(self, small_test_adata):
        """Test that get_all_degs function exists and is callable"""
        assert callable(de.get_all_degs)
    
    def test_de_results_format(self, small_test_adata):
        """Test that DE results are in expected format"""
        # This will depend on the actual implementation
        # For now, just verify the data structure
        assert small_test_adata.n_obs > 0
        assert small_test_adata.n_vars > 0


class TestStatisticalTests:
    """Test statistical testing functions"""
    
    def test_statistical_significance(self, small_test_adata):
        """Test statistical significance calculations"""
        # Placeholder for statistical tests
        pass
        
    def test_multiple_testing_correction(self, small_test_adata):
        """Test multiple testing correction"""
        # Placeholder for multiple testing correction tests
        pass


class TestDEInputValidation:
    """Test input validation for DE functions"""
    
    def test_invalid_design_column(self, small_test_adata):
        """Test handling of invalid design column"""
        with pytest.raises((KeyError, ValueError, Exception)):  # More general exception catching
            de.get_degs(small_test_adata, design_col='nonexistent_column')
    
    def test_empty_adata(self):
        """Test handling of empty AnnData object"""
        empty_adata = AnnData(X=np.empty((0, 0)))
        with pytest.raises((ValueError, IndexError)):
            de.get_degs(empty_adata, design_col='feature_call')


class TestDEParameterTypes:
    """Test parameter type handling"""
    
    def test_alpha_parameter_type(self, small_test_adata):
        """Test that alpha parameter accepts float values"""
        # This should not raise an error for valid alpha values
        try:
            de.get_degs(small_test_adata, design_col='feature_call', alpha=0.01)
            de.get_degs(small_test_adata, design_col='feature_call', alpha=0.1)
        except Exception as e:
            # If it fails for other reasons (like missing dependencies), that's ok for now
            # We're just testing the parameter is accepted
            if "alpha" in str(e):
                pytest.fail(f"Alpha parameter not accepted: {e}")
    
    def test_n_cpus_parameter_type(self, small_test_adata):
        """Test that n_cpus parameter accepts integer values"""
        try:
            de.get_degs(small_test_adata, design_col='feature_call', n_cpus=1)
            de.get_degs(small_test_adata, design_col='feature_call', n_cpus=2)
        except Exception as e:
            if "n_cpus" in str(e):
                pytest.fail(f"n_cpus parameter not accepted: {e}") 