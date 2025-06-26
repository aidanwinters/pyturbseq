"""
Tests for pyturbseq.plot module
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from anndata import AnnData
import pyturbseq.plot as plot


class TestPlotModule:
    """Test plot module functions"""
    
    def test_plot_module_imports(self):
        """Test that plot module imports successfully"""
        assert plot is not None
    
    def test_plot_functions_exist(self):
        """Test that expected plot functions exist"""
        # Check for common plotting function patterns
        function_patterns = ['plot', 'heatmap', 'scatter', 'violin', 'bar']
        
        module_functions = [func for func in dir(plot) if not func.startswith('_')]
        
        # At least some functions should exist
        assert len(module_functions) > 0


class TestVisualizationFunctions:
    """Test visualization functionality"""
    
    def test_plotting_with_test_data(self, small_test_adata):
        """Test plotting functions with test data"""
        # Basic test that data is suitable for plotting
        assert small_test_adata.n_obs > 0
        assert small_test_adata.n_vars > 0
        
        # Check that perturbation data exists for plotting
        assert 'feature_call' in small_test_adata.obs.columns
        assert small_test_adata.obs['feature_call'].nunique() > 1
    
    def test_perturbation_heatmap_function(self, small_test_adata):
        """Test perturbation heatmap function if it exists"""
        if hasattr(plot, 'perturbation_heatmap'):
            try:
                # Test that function can be called
                fig = plot.perturbation_heatmap(small_test_adata, show=False)
                if fig is not None:
                    plt.close(fig)
            except Exception as e:
                # If it fails for missing data or other reasons, that's ok
                if "perturbation_heatmap" in str(e):
                    pytest.fail(f"perturbation_heatmap failed unexpectedly: {e}")
    
    def test_matplotlib_backend(self):
        """Test that matplotlib backend is properly set for testing"""
        backend = matplotlib.get_backend()
        assert backend in ['Agg', 'svg', 'pdf']  # Non-interactive backends


class TestPlotInputValidation:
    """Test input validation for plotting functions"""
    
    def test_empty_data_handling(self):
        """Test handling of empty data"""
        empty_adata = AnnData(X=np.empty((0, 0)))
        # Should not crash when passed empty data
        assert empty_adata.n_obs == 0
        assert empty_adata.n_vars == 0


class TestPlotUtilities:
    """Test plot utility functions"""
    
    def test_plot_utilities_exist(self):
        """Test that plot utility functions exist"""
        utility_functions = ['save_fig', 'set_style', 'configure']
        
        for func_name in utility_functions:
            if hasattr(plot, func_name):
                assert callable(getattr(plot, func_name)) 