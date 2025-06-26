"""
Tests for pyturbseq.utils module
"""

import pytest
import numpy as np
import pandas as pd
from anndata import AnnData
import pyturbseq.utils as utils


class TestStringParsing:
    """Test string parsing utility functions"""
    
    def test_split_sort_trim(self):
        # Test normal case
        result = utils.split_sort_trim("GENE2|GENE1", delim='|', delim2='_')
        assert result == "GENE1|GENE2"
        
        # Test with underscores
        result = utils.split_sort_trim("GENE2_1|GENE1_2", delim='|', delim2='_')
        assert result == "GENE1|GENE2"
        
        # Test non-string input
        result = utils.split_sort_trim(None)
        assert result is None
        
    def test_split_compare(self):
        # Test identical genes
        result = utils.split_compare("GENE1|GENE1", expected_num=2)
        assert result == "GENE1"
        
        # Test different genes
        result = utils.split_compare("GENE1|GENE2", expected_num=2)
        assert result is None
        
        # Test wrong number of genes
        result = utils.split_compare("GENE1|GENE2|GENE3", expected_num=2)
        assert result is None


class TestPerturbationMatrix:
    """Test perturbation matrix generation functions"""
    
    def test_generate_perturbation_matrix_basic(self, small_test_adata):
        """Test basic perturbation matrix generation"""
        pert_matrix = utils.generate_perturbation_matrix(
            small_test_adata,
            perturbation_col='feature_call',
            control_value='NTC'
        )
        
        assert isinstance(pert_matrix, pd.DataFrame)
        assert pert_matrix.shape[0] == small_test_adata.n_obs
        # Should exclude the reference by default (keep_ref=False)
        assert 'NTC' not in pert_matrix.columns
        
    def test_generate_perturbation_matrix_keep_ref(self, small_test_adata):
        """Test perturbation matrix with reference kept"""
        pert_matrix = utils.generate_perturbation_matrix(
            small_test_adata,
            perturbation_col='feature_call',
            control_value='NTC',
            keep_ref=True
        )
        
        assert isinstance(pert_matrix, pd.DataFrame)
        assert pert_matrix.shape[0] == small_test_adata.n_obs
        # Should include the reference when keep_ref=True
        assert 'NTC' in pert_matrix.columns
        
    def test_get_perturbation_matrix_inplace(self, small_test_adata):
        """Test adding perturbation matrix to adata object"""
        original_obsm_keys = list(small_test_adata.obsm.keys())
        
        utils.get_perturbation_matrix(
            small_test_adata,
            perturbation_col='feature_call',
            inplace=True
        )
        
        # Check that perturbation matrix was added
        assert 'perturbation' in small_test_adata.obsm
        assert small_test_adata.obsm['perturbation'].shape[0] == small_test_adata.n_obs


class TestTargetChange:
    """Test target change calculation functions"""
    
    def test_calculate_target_change_basic(self, small_test_adata):
        """Test basic target change calculation"""
        # Add some target genes to var
        small_test_adata.var.index = [f"GENE1_{i}" for i in range(10)] + [f"GENE2_{i}" for i in range(10)] + [f"GENE3_{i}" for i in range(10)] + [f"OTHER_GENE_{i}" for i in range(20)]
        small_test_adata.var['target_gene'] = ['GENE1'] * 10 + ['GENE2'] * 10 + ['GENE3'] * 10 + [None] * 20
        
        # Create a perturbation gene map to map perturbation names to actual gene names
        perturbation_gene_map = {
            "GENE1": "GENE1_0",  # Map to the first gene of each type
            "GENE2": "GENE2_0",
            "GENE3": "GENE3_0"
        }
        result = utils.calculate_target_change(
            small_test_adata,
            perturbation_column='feature_call',
            control_value='NTC',
            perturbation_gene_map=perturbation_gene_map,
            inplace=False
        )
        
        assert isinstance(result, AnnData)
        assert 'target_pct_change' in result.obsm


class TestDataFiltering:
    """Test data filtering functions"""
    
    def test_filter_adata_obs(self, small_test_adata):
        """Test filtering adata by obs"""
        # Filter to cells with high gene counts
        filtered = utils.filter_adata(
            small_test_adata,
            obs_filters=['n_genes_by_counts > 3000'],
            copy=True
        )
        
        assert filtered.n_obs <= small_test_adata.n_obs
        assert (filtered.obs['n_genes_by_counts'] > 3000).all()
        
    def test_filter_adata_var(self, small_test_adata):
        """Test filtering adata by var"""
        filtered = utils.filter_adata(
            small_test_adata,
            var_filters=['highly_variable == True'],
            copy=True
        )
        
        assert filtered.n_vars <= small_test_adata.n_vars
        assert filtered.var['highly_variable'].all()
        
    def test_filter_to_feature_type(self, small_test_adata):
        """Test filtering to specific feature type"""
        filtered = utils.filter_to_feature_type(
            small_test_adata,
            feature_type='Gene Expression'
        )
        
        assert (filtered.var['feature_types'] == 'Gene Expression').all()


class TestDataProcessing:
    """Test data processing functions"""
    
    def test_subsample_on_covariate(self, small_test_adata):
        """Test subsampling on covariate"""
        subsampled = utils.subsample_on_covariate(
            small_test_adata,
            column='feature_call',
            num_cells=5,
            copy=True
        )
        
        # Check that each group has at most 5 cells
        group_counts = subsampled.obs['feature_call'].value_counts()
        assert (group_counts <= 5).all()
        
    def test_pseudobulk(self, small_test_adata):
        """Test pseudobulk aggregation"""
        pb_adata = utils.pseudobulk(
            small_test_adata,
            groupby='feature_call'
        )
        
        # Check that we have one sample per perturbation
        expected_n_obs = small_test_adata.obs['feature_call'].nunique()
        assert pb_adata.n_obs == expected_n_obs
        assert pb_adata.n_vars == small_test_adata.n_vars


class TestUtilityFunctions:
    """Test miscellaneous utility functions"""
    
    def test_cells_not_normalized(self, small_test_adata):
        """Test normalization check"""
        # Raw counts should not be normalized
        assert utils.cells_not_normalized(small_test_adata)
        
        # Normalize and check again
        small_test_adata.X = small_test_adata.X / small_test_adata.X.sum(axis=1, keepdims=True)
        assert not utils.cells_not_normalized(small_test_adata)
        
    def test_cluster_df(self, sample_perturbation_matrix):
        """Test dataframe clustering"""
        clustered = utils.cluster_df(
            sample_perturbation_matrix,
            cluster_rows=True,
            cluster_cols=True
        )
        
        # Check that result is a DataFrame with same shape
        assert isinstance(clustered, pd.DataFrame)
        assert clustered.shape == sample_perturbation_matrix.shape 