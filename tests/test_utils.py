import pytest
import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix
import warnings
import re

from pyturbseq.utils import (
    split_sort_trim,
    split_compare,
    split_sort_paste,
    add_pattern_to_adata,
    filter_adata,
    filter_to_feature_type,
    split_by_feature_type,
    generate_perturbation_matrix,
    get_perturbation_matrix,
    cluster_df,
    cells_not_normalized,
    calculate_target_change,
    calculate_adjacency,
    cluster_adjacency,
    calculate_edistances,
    zscore,
    pseudobulk,
    subsample_on_covariate,
    subsample_on_multiple_covariates,
    calculate_label_similarity,
    get_average_precision_score,
)


@pytest.fixture
def sample_adata():
    """Create a sample AnnData object for testing."""
    n_obs, n_vars = 100, 50
    X = np.random.negative_binomial(5, 0.3, (n_obs, n_vars)).astype(np.float32)
    
    obs = pd.DataFrame({
        'cell_type': np.random.choice(['TypeA', 'TypeB', 'TypeC'], n_obs),
        'batch': np.random.choice(['batch1', 'batch2'], n_obs),
        'feature_call': np.random.choice(['NTC', 'gene1', 'gene2', 'gene1|gene2'], n_obs),
        'perturbation': np.random.choice(['control', 'pert1', 'pert2'], n_obs),
        'quality_score': np.random.uniform(0, 1, n_obs)
    })
    
    var = pd.DataFrame({
        'gene_symbols': [f'Gene_{i}' for i in range(n_vars)],
        'feature_types': ['Gene Expression'] * 40 + ['CRISPR Guide Capture'] * 10
    }, index=[f'Gene_{i}' for i in range(n_vars)])
    
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obs_names = [f'Cell_{i}' for i in range(n_obs)]
    return adata


@pytest.fixture
def sample_perturbation_adata():
    """Create AnnData with perturbation data for target change testing."""
    n_obs, n_vars = 200, 100
    X = np.random.negative_binomial(10, 0.4, (n_obs, n_vars)).astype(np.float32)
    
    # Create perturbation calls with some controls
    perturbations = ['NTC'] * 50 + ['gene1'] * 50 + ['gene2'] * 50 + ['gene1|gene2'] * 50
    
    obs = pd.DataFrame({
        'feature_call': perturbations,
        'batch': np.random.choice(['batch1', 'batch2'], n_obs)
    })
    
    var = pd.DataFrame(index=[f'gene{i}' for i in range(n_vars)])
    
    adata = AnnData(X=X, obs=obs, var=var)
    adata.obs_names = [f'Cell_{i}' for i in range(n_obs)]
    return adata


class TestStringParsing:
    """Test string parsing utility functions."""
    
    def test_split_sort_trim(self):
        """Test split_sort_trim function."""
        # Normal case
        result = split_sort_trim("gene2_1|gene1_2", "|", "_")
        assert result == "gene1|gene2"
        
        # Single item
        result = split_sort_trim("gene1_1", "|", "_")
        assert result == "gene1"
        
        # Non-string input
        result = split_sort_trim(123)
        assert result is None
        
        # Empty string
        result = split_sort_trim("")
        assert result == ""
    
    def test_split_compare(self):
        """Test split_compare function."""
        # Matching components
        result = split_compare("gene1_1|gene1_2", "|", "_", 2)
        assert result == "gene1"
        
        # Non-matching components
        result = split_compare("gene1_1|gene2_2", "|", "_", 2)
        assert result is None
        
        # Wrong number of components
        result = split_compare("gene1_1|gene1_2|gene1_3", "|", "_", 2)
        assert result is None
        
        # Non-string input
        result = split_compare(123)
        assert result is None
    
    def test_split_sort_paste(self):
        """Test split_sort_paste function."""
        # List input
        result = split_sort_paste(['gene2_1', 'gene1_2'], '_', '|')
        assert result == "gene1|gene2"
        
        # Series input
        series = pd.Series(['gene3_x', 'gene1_y', 'gene2_z'])
        result = split_sort_paste(series, '_', '|')
        assert result == "gene1|gene2|gene3"
    
    def test_add_pattern_to_adata(self, sample_adata):
        """Test add_pattern_to_adata function."""
        pattern = r"(?P<sample>\w+)_(?P<replicate>\w+)"
        search_string = "sample1_rep2"
        
        add_pattern_to_adata(sample_adata, search_string, pattern, quiet=True)
        
        assert 'sample' in sample_adata.obs.columns
        assert 'replicate' in sample_adata.obs.columns
        assert sample_adata.obs['sample'].iloc[0] == 'sample1'
        assert sample_adata.obs['replicate'].iloc[0] == 'rep2'
        
        # Test strict mode with non-matching pattern
        with pytest.raises(ValueError):
            add_pattern_to_adata(sample_adata, "nomatch", pattern, strict=True)


class TestAdataFiltering:
    """Test AnnData filtering functions."""
    
    def test_filter_adata(self, sample_adata):
        """Test filter_adata function."""
        # Filter observations
        obs_filters = ["quality_score > 0.5"]
        filtered = filter_adata(sample_adata, obs_filters=obs_filters)
        assert all(filtered.obs['quality_score'] > 0.5)
        
        # Filter variables
        var_filters = ["feature_types == 'Gene Expression'"]
        filtered = filter_adata(sample_adata, var_filters=var_filters)
        assert all(filtered.var['feature_types'] == 'Gene Expression')
        
        # Filter both
        filtered = filter_adata(sample_adata, obs_filters=obs_filters, var_filters=var_filters)
        assert all(filtered.obs['quality_score'] > 0.5)
        assert all(filtered.var['feature_types'] == 'Gene Expression')
    
    def test_filter_to_feature_type(self, sample_adata):
        """Test filter_to_feature_type function."""
        filtered = filter_to_feature_type(sample_adata, 'Gene Expression')
        assert all(filtered.var['feature_types'] == 'Gene Expression')
        assert filtered.n_vars == 40  # Only Gene Expression features
    
    def test_split_by_feature_type(self, sample_adata):
        """Test split_by_feature_type function."""
        split_dict = split_by_feature_type(sample_adata)
        
        assert 'Gene Expression' in split_dict
        assert 'CRISPR Guide Capture' in split_dict
        assert split_dict['Gene Expression'].n_vars == 40
        assert split_dict['CRISPR Guide Capture'].n_vars == 10


class TestPerturbationMatrix:
    """Test perturbation matrix generation functions."""
    
    def test_generate_perturbation_matrix(self, sample_adata):
        """Test generate_perturbation_matrix function."""
        pm = generate_perturbation_matrix(
            sample_adata, 
            perturbation_col='feature_call',
            reference_value='NTC',
            verbose=False
        )
        
        assert isinstance(pm, pd.DataFrame)
        assert pm.shape[0] == sample_adata.n_obs
        assert 'NTC' not in pm.columns  # Reference should be removed by default
        assert pm.dtypes.apply(lambda x: x == bool).all()  # Should be boolean
    
    def test_generate_perturbation_matrix_keep_ref(self, sample_adata):
        """Test generate_perturbation_matrix with keep_ref=True."""
        pm = generate_perturbation_matrix(
            sample_adata,
            perturbation_col='feature_call',
            reference_value='NTC',
            keep_ref=True,
            verbose=False
        )
        
        assert 'NTC' in pm.columns
        assert pm.shape[0] == sample_adata.n_obs
    
    def test_get_perturbation_matrix(self, sample_adata):
        """Test get_perturbation_matrix function."""
        # Test inplace=True
        get_perturbation_matrix(
            sample_adata,
            perturbation_col='feature_call',
            inplace=True,
            verbose=False
        )
        
        assert 'perturbation' in sample_adata.obsm
        assert isinstance(sample_adata.obsm['perturbation'], pd.DataFrame)
        
        # Test inplace=False
        pm = get_perturbation_matrix(
            sample_adata,
            perturbation_col='feature_call',
            inplace=False,
            verbose=False
        )
        
        assert isinstance(pm, pd.DataFrame)


class TestDataProcessing:
    """Test data processing utility functions."""
    
    def test_cluster_df(self):
        """Test cluster_df function."""
        # Create test DataFrame
        data = np.random.rand(10, 5)
        df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(5)])
        
        clustered = cluster_df(df, cluster_rows=True, cluster_cols=True)
        
        assert clustered.shape == df.shape
        assert isinstance(clustered, pd.DataFrame)
        # Check that order has changed (with high probability)
        assert not clustered.columns.equals(df.columns) or not clustered.index.equals(df.index)
    
    def test_cells_not_normalized(self, sample_adata):
        """Test cells_not_normalized function."""
        # Test with raw counts (should return True)
        assert cells_not_normalized(sample_adata) == True
        
        # Test with normalized data
        sc.pp.normalize_total(sample_adata, target_sum=1e4)
        assert cells_not_normalized(sample_adata) == False
    
    def test_calculate_target_change(self, sample_perturbation_adata):
        """Test calculate_target_change function."""
        # Normalize first
        sc.pp.normalize_total(sample_perturbation_adata, target_sum=1e4)
        
        calculate_target_change(
            sample_perturbation_adata,
            perturbation_column='feature_call',
            reference_value='NTC',
            quiet=True,
            check_norm=False
        )
        
        # Check that metrics were added
        expected_metrics = ['target_pct_change', 'target_log2fc', 'target_zscore', 
                          'target_gene_expression', 'target_reference_mean', 'target_reference_std']
        
        for metric in expected_metrics:
            assert metric in sample_perturbation_adata.obsm
    
    def test_calculate_adjacency(self, sample_adata):
        """Test calculate_adjacency function."""
        calculate_adjacency(sample_adata, metric='correlation', inplace=True)
        
        assert 'adjacency' in sample_adata.obsm
        adj_matrix = sample_adata.obsm['adjacency']
        assert adj_matrix.shape == (sample_adata.n_obs, sample_adata.n_obs)
        assert np.allclose(adj_matrix, adj_matrix.T)  # Should be symmetric
    
    def test_calculate_edistances(self, sample_adata):
        """Test calculate_edistances function."""
        # Add PCA for testing
        sc.tl.pca(sample_adata)
        
        edists = calculate_edistances(
            sample_adata,
            obs_key='perturbation',
            control='control',
            verbose=False
        )
        
        assert isinstance(edists, pd.Series)
        assert edists.name == 'edistance'
        assert len(edists) > 0


class TestNormalization:
    """Test normalization and transformation functions."""
    
    def test_zscore(self, sample_adata):
        """Test zscore function."""
        # Add a reference column
        sample_adata.obs['ref_group'] = 'control'
        sample_adata.obs.iloc[:10, sample_adata.obs.columns.get_loc('ref_group')] = 'reference'
        
        zscores = zscore(
            sample_adata,
            ref_col='ref_group',
            ref_val='reference'
        )
        
        assert zscores.shape == sample_adata.X.shape
        assert isinstance(zscores, np.ndarray)
    
    def test_pseudobulk(self, sample_adata):
        """Test pseudobulk function."""
        bulk_adata = pseudobulk(sample_adata, groupby='cell_type')
        
        assert isinstance(bulk_adata, AnnData)
        assert bulk_adata.n_obs <= len(sample_adata.obs['cell_type'].unique())
        assert bulk_adata.n_vars == sample_adata.n_vars


class TestSubsampling:
    """Test subsampling utility functions."""
    
    def test_subsample_on_covariate(self, sample_adata):
        """Test subsample_on_covariate function."""
        subsampled = subsample_on_covariate(
            sample_adata,
            column='cell_type',
            num_cells=10,
            seed=42
        )
        
        # Check that all cell types have the same count
        counts = subsampled.obs['cell_type'].value_counts()
        assert all(counts == 10)
    
    def test_subsample_on_multiple_covariates(self, sample_adata):
        """Test subsample_on_multiple_covariates function."""
        subsampled = subsample_on_multiple_covariates(
            sample_adata,
            columns=['cell_type', 'batch'],
            num_cells=5,
            seed=42
        )
        
        # Check that combinations are balanced
        counts = subsampled.obs.groupby(['cell_type', 'batch']).size()
        assert all(counts <= 5)
    
    def test_calculate_label_similarity(self, sample_adata):
        """Test calculate_label_similarity function."""
        # Normalize data first
        sc.pp.normalize_total(sample_adata, target_sum=1e4)
        sc.pp.log1p(sample_adata)
        
        similarity_df = calculate_label_similarity(
            sample_adata,
            label_column='cell_type',
            metric='euclidean',
            subset=50,
            verbose=False,
            n_jobs=1
        )
        
        assert isinstance(similarity_df, pd.DataFrame)
        assert 'similarity' in similarity_df.columns
        assert 'within' in similarity_df.columns
        assert len(similarity_df) > 0
    
    def test_get_average_precision_score(self):
        """Test get_average_precision_score function."""
        # Create mock similarity results
        similarity_df = pd.DataFrame({
            'similarity': np.random.rand(100),
            'within': np.random.choice([True, False], 100),
            'label1': ['A'] * 50 + ['B'] * 50,
            'label2': ['A'] * 50 + ['B'] * 50
        })
        
        ap_score = get_average_precision_score(similarity_df)
        
        assert isinstance(ap_score, float)
        assert 0 <= ap_score <= 1


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_generate_perturbation_matrix_missing_reference(self, sample_adata):
        """Test error when reference value is not found."""
        with pytest.raises(ValueError, match="not found in feature list"):
            generate_perturbation_matrix(
                sample_adata,
                perturbation_col='feature_call',
                reference_value='MISSING_REF',
                verbose=False
            )
    
    def test_calculate_target_change_duplicated_genes(self, sample_perturbation_adata):
        """Test error handling for duplicated gene names."""
        # Create duplicated gene names
        sample_perturbation_adata.var.index = ['gene1'] * sample_perturbation_adata.n_vars
        
        with pytest.raises(ValueError, match="Duplicated gene names found"):
            calculate_target_change(
                sample_perturbation_adata,
                perturbation_column='feature_call',
                quiet=True
            )
    
    def test_calculate_target_change_duplicated_obs(self, sample_perturbation_adata):
        """Test error handling for duplicated observation names."""
        # Create duplicated observation names
        sample_perturbation_adata.obs.index = ['cell1'] * sample_perturbation_adata.n_obs
        
        with pytest.raises(ValueError, match="Observation names are not unique"):
            calculate_target_change(
                sample_perturbation_adata,
                perturbation_column='feature_call',
                quiet=True
            )


@pytest.mark.parametrize("metric", ['euclidean', 'correlation', 'cosine'])
def test_calculate_adjacency_metrics(sample_adata, metric):
    """Test calculate_adjacency with different metrics."""
    calculate_adjacency(sample_adata, metric=metric, inplace=True)
    
    assert 'adjacency' in sample_adata.obsm
    adj_matrix = sample_adata.obsm['adjacency']
    assert adj_matrix.shape == (sample_adata.n_obs, sample_adata.n_obs)


@pytest.mark.parametrize("return_boolean", [True, False])
def test_generate_perturbation_matrix_types(sample_adata, return_boolean):
    """Test perturbation matrix with different return types."""
    pm = generate_perturbation_matrix(
        sample_adata,
        perturbation_col='feature_call',
        return_boolean=return_boolean,
        verbose=False
    )
    
    if return_boolean:
        assert pm.dtypes.apply(lambda x: x == bool).all()
    else:
        assert pm.dtypes.apply(lambda x: np.issubdtype(x, np.floating)).all()


if __name__ == "__main__":
    pytest.main([__file__]) 