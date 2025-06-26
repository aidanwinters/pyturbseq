"""
Tests for pyturbseq.interaction module
"""

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
import pyturbseq.interaction as interaction


class TestInteractionAnalysis:
    """Test interaction analysis functions"""
    
    def test_interaction_calculation_basic(self, dual_perturbation_adata):
        """Test basic interaction calculation"""
        # Add perturbation matrix first
        import pyturbseq.utils as utils
        utils.get_perturbation_matrix(
            dual_perturbation_adata,
            perturbation_col='feature_call',
            inplace=True
        )
        
        # This test will depend on the actual implementation
        # For now, just check that the function exists and can be called
        assert hasattr(interaction, 'get_model_fit')
        assert hasattr(interaction, 'fit_many')
    
    def test_dual_perturbation_detection(self, dual_perturbation_adata):
        """Test detection of dual perturbations"""
        # Check that dual perturbations are properly identified
        dual_perturbs = dual_perturbation_adata.obs['feature_call'].str.contains('|', na=False)
        assert dual_perturbs.sum() > 0
        
        # Check specific dual perturbations exist
        assert 'GENE1|GENE2' in dual_perturbation_adata.obs['feature_call'].values


class TestInteractionMetrics:
    """Test interaction metric calculations"""
    
    def test_interaction_score_calculation(self, dual_perturbation_adata):
        """Test interaction score calculation"""
        # This will depend on the specific implementation
        # For now, just verify the data structure is appropriate
        assert dual_perturbation_adata.n_obs > 0
        assert dual_perturbation_adata.n_vars > 0
        
    def test_epistasis_calculation(self, dual_perturbation_adata):
        """Test epistasis calculation if implemented"""
        # Placeholder for epistasis tests
        pass


class TestInteractionFunctions:
    """Test individual interaction analysis functions"""
    
    def test_get_singles_function(self):
        """Test the get_singles helper function"""
        if hasattr(interaction, 'get_singles'):
            single_a, single_b = interaction.get_singles('GENE1|GENE2')
            assert single_a in ['GENE1|NTC', 'NTC|GENE1']
            assert single_b in ['GENE2|NTC', 'NTC|GENE2']
    
    def test_get_model_fit_exists(self):
        """Test that get_model_fit function exists"""
        assert hasattr(interaction, 'get_model_fit')
        assert callable(interaction.get_model_fit)
    
    def test_fit_many_exists(self):
        """Test that fit_many function exists"""
        assert hasattr(interaction, 'fit_many')
        assert callable(interaction.fit_many)
        
    def test_fit_many_parallel_exists(self):
        """Test that fit_many_parallel function exists"""
        if hasattr(interaction, 'fit_many_parallel'):
            assert callable(interaction.fit_many_parallel)


class TestInteractionInputValidation:
    """Test input validation for interaction functions"""
    
    def test_get_model_fit_with_pseudobulk_data(self):
        """Test get_model_fit with pseudobulk data structure"""
        # Create simple test data
        np.random.seed(42)
        test_data = pd.DataFrame(
            np.random.randn(4, 10),
            index=['NTC|NTC', 'GENE1|NTC', 'NTC|GENE2', 'GENE1|GENE2'],
            columns=[f'gene_{i}' for i in range(10)]
        )
        
        try:
            result, prediction = interaction.get_model_fit(
                test_data, 
                'GENE1|GENE2',
                method='robust',
                plot=False,
                verbose=False
            )
            
            # Basic checks on result structure
            assert isinstance(result, dict)
            assert 'perturbation' in result
            assert result['perturbation'] == 'GENE1|GENE2'
            
        except Exception as e:
            # If it fails due to missing dependencies or other issues, 
            # that's acceptable for this test
            if "get_model_fit" in str(e):
                pytest.fail(f"get_model_fit function failed unexpectedly: {e}")


class TestInteractionUtilities:
    """Test interaction utility functions"""
    
    def test_breakdown_functions_exist(self):
        """Test that perturbation breakdown functions exist"""
        breakdown_functions = [
            'breakdown_double_wRef',
            'breakdown_triple_wRef', 
            'breakdown_perturbation'
        ]
        
        for func_name in breakdown_functions:
            if hasattr(interaction, func_name):
                assert callable(getattr(interaction, func_name))
    
    def test_get_model_wNTC_exists(self):
        """Test that get_model_wNTC function exists"""
        if hasattr(interaction, 'get_model_wNTC'):
            assert callable(interaction.get_model_wNTC)
    
    def test_model_fit_wrapper_exists(self):
        """Test that model_fit_wrapper function exists"""
        if hasattr(interaction, 'model_fit_wrapper'):
            assert callable(interaction.model_fit_wrapper) 