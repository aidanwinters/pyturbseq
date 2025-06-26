import numpy as np
import pandas as pd
import pytest
import scanpy as sc
from anndata import AnnData
from scipy.sparse import csr_matrix

from pyturbseq import calling


@pytest.fixture
def guide_adata():
    """Small AnnData object for feature calling tests."""
    n_obs, n_var = 20, 3
    X = np.random.poisson(5, (n_obs, n_var)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_obs)])
    var = pd.DataFrame({"feature_types": ["CRISPR Guide Capture"] * n_var},
                       index=[f"guide{i}" for i in range(n_var)])
    ad = AnnData(X=csr_matrix(X), obs=obs, var=var)
    return ad


@pytest.fixture
def hto_adata():
    """AnnData object for HTO calling tests."""
    n_obs, n_var = 30, 2
    X = np.random.poisson(5, (n_obs, n_var)).astype(np.float32)
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(n_obs)])
    var = pd.DataFrame({"feature_types": ["HTO"] * n_var},
                       index=[f"hto{i}" for i in range(n_var)])
    ad = AnnData(X=csr_matrix(X), obs=obs, var=var)
    return ad


def test_gm_low_counts_returns_zero():
    counts = np.array([0, 0, 0, 0])
    result = calling.gm(counts, calling_min_count=1)
    assert np.array_equal(result, np.zeros_like(counts))


def test_gm_too_few_cells_returns_neg1():
    counts = np.arange(1, 9)  # less than 10 cells
    result = calling.gm(counts)
    assert isinstance(result, tuple)
    assert np.all(result[0] == -1)
    assert np.all(result[1] == -1)


def test_gm_basic_output_shape():
    counts = np.concatenate([np.random.poisson(1, 20), np.random.poisson(8, 20)])
    result = calling.gm(counts)
    assert result.shape == counts.shape
    assert result.dtype == bool


def test_call_features_inplace(guide_adata):
    calling.call_features(guide_adata,
                          feature_type="CRISPR Guide Capture",
                          feature_key="guide",
                          n_jobs=1,
                          quiet=True,
                          inplace=True)
    assert "guide_calls" in guide_adata.obsm
    assert "guide" in guide_adata.uns
    assert "num_guide" in guide_adata.obs.columns
    assert "guide_call" in guide_adata.obs.columns
    assert guide_adata.obsm["guide_calls"].shape == (guide_adata.n_obs, guide_adata.n_vars)


def test_calculate_feature_call_metrics(guide_adata):
    calling.call_features(guide_adata, feature_type="CRISPR Guide Capture",
                          feature_key="guide", quiet=True)
    calling.calculate_feature_call_metrics(guide_adata,
                                           feature_type="CRISPR Guide Capture",
                                           inplace=True,
                                           topN=[1, 2],
                                           quiet=True)
    expected_cols = [
        "total_feature_counts",
        "log1p_total_feature_counts",
        "log10_total_feature_counts",
        "ratio_2nd_1st_feature",
        "log2_ratio_2nd_1st_feature",
        "pct_top1_features",
        "pct_top2_features",
    ]
    for col in expected_cols:
        assert col in guide_adata.obs.columns


def test_parse_dual_guide_df_basic():
    df = pd.DataFrame({
        "num_features": [2, 1],
        "feature_call": ["geneA_A|geneB_B", "geneC_A"]
    }, index=["cell1", "cell2"])
    parsed = calling.parse_dual_guide_df(df, position_annotation=["A", "B"])
    for col in [
        "sgRNA_fullID_A", "sgRNA_fullID_B",
        "sgRNA_A", "sgRNA_B",
        "perturbation_fullID", "perturbation"
    ]:
        assert col in parsed.columns
    assert parsed.loc["cell1", "sgRNA_A"] == "geneA"
    assert pd.isna(parsed.loc["cell2", "sgRNA_A"])


def test_parse_dual_guide_wrapper():
    df = pd.DataFrame({
        "num_features": [2],
        "feature_call": ["geneA_A|geneB_B"]
    }, index=["cell1"])
    ad = AnnData(obs=df)
    out = calling.parse_dual_guide(ad, inplace=False, position_annotation=["A", "B"])
    assert "perturbation" in out.obs.columns
    assert out is not ad


def test_CLR_transformation():
    x = np.array([[1., 2.], [3., 4.]])
    expected = np.log1p(x) - np.log1p(x).mean(axis=0)
    result = calling.CLR(x)
    assert np.allclose(result, expected)


def test_multivariate_clr_gm_shapes(hto_adata):
    x = hto_adata.X.toarray()
    clr, assigned, probs = calling._multivariate_clr_gm(x)
    assert clr.shape == x.shape
    assert assigned.shape[0] == x.shape[0]
    assert probs.shape[0] == x.shape[0]
    assert np.all((probs >= 0) & (probs <= 1))


def test_call_hto_basic(hto_adata):
    res = calling.call_hto(hto_adata, inplace=False)
    assert isinstance(res, AnnData)
    for col in ["HTO_total_counts", "HTO", "HTO_max_probability"]:
        assert col in res.obs.columns
    assert set(res.obs["HTO"].dropna().unique()).issubset(res.var.index)


def test_binarize_guides_and_check_calls():
    X = np.array([[3, 0], [2, 0], [4, 1], [0, 0]])
    obs = pd.DataFrame(index=[f"cell{i}" for i in range(X.shape[0])])
    var = pd.DataFrame(index=["g1", "g2"])
    ad = AnnData(X=csr_matrix(X), obs=obs, var=var)
    thresh = pd.DataFrame({"UMI_threshold": [2, 1]}, index=var.index)
    binarized = calling.binarize_guides(ad, threshold_df=thresh, inplace=False)
    assert isinstance(binarized, AnnData)
    assert set(np.unique(binarized.X.toarray())) <= {0, 1}
    flagged = calling.check_calls(binarized, expected_max_proportion=0.5)
    assert "g1" in flagged
