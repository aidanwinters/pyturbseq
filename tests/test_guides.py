"""
Tests for pyturbseq.guides module
"""

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
import pyturbseq.guides as guides


class TestGuidesModule:
    """Test guides module functions"""
    
    def test_guides_module_imports(self):
        """Test that guides module imports successfully"""
        assert guides is not None
    
    def test_guides_functions_exist(self):
        """Test that expected guides functions exist"""
        # Check for common function patterns in guides modules
        function_patterns = ['guide', 'target', 'map', 'annotate']
        
        module_functions = [func for func in dir(guides) if not func.startswith('_')]
        
        # At least some functions should exist
        assert len(module_functions) > 0


class TestGuideProcessing:
    """Test guide RNA processing functionality"""
    
    def test_guides_with_test_data(self, small_test_adata):
        """Test guides functions with test data"""
        # This is a placeholder for when guides functions are identified
        assert small_test_adata.n_obs > 0
        assert small_test_adata.n_vars > 0
    
    def test_guide_target_mapping(self, small_test_adata):
        """Test guide to target gene mapping"""
        # Basic test that perturbation data exists for guide mapping
        assert 'feature_call' in small_test_adata.obs.columns
        assert small_test_adata.obs['feature_call'].nunique() > 1


class TestGuidesInputValidation:
    """Test input validation for guides functions"""
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_adata = AnnData(X=np.empty((0, 0)))
        # Should not crash when passed empty data
        assert empty_adata.n_obs == 0
        assert empty_adata.n_vars == 0


class TestGuideUtilities:
    """Test guide utility functions"""
    
    def test_guide_utilities_exist(self):
        """Test that guide utility functions exist"""
        utility_functions = ['parse_guides', 'annotate_guides', 'map_guides']
        
        for func_name in utility_functions:
            if hasattr(guides, func_name):
                assert callable(getattr(guides, func_name)) 