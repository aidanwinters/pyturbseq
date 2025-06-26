import types
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pytest
from anndata import AnnData


def _load_plot_module(monkeypatch):
    """Import pyturbseq.plot with dummy optional dependencies."""
    sm = types.ModuleType('statsmodels')
    sm_api = types.ModuleType('statsmodels.api')
    sm_formula_pkg = types.ModuleType('statsmodels.formula')
    sm_formula_api = types.ModuleType('statsmodels.formula.api')
    sm.api = sm_api
    sm.formula = sm_formula_pkg
    sm_formula_pkg.api = sm_formula_api
    monkeypatch.setitem(sys.modules, 'statsmodels', sm)
    monkeypatch.setitem(sys.modules, 'statsmodels.api', sm_api)
    monkeypatch.setitem(sys.modules, 'statsmodels.formula', sm_formula_pkg)
    monkeypatch.setitem(sys.modules, 'statsmodels.formula.api', sm_formula_api)

    up = types.ModuleType('upsetplot')
    up.plot = lambda *a, **k: plt.figure()
    monkeypatch.setitem(sys.modules, 'upsetplot', up)

    import importlib
    import pyturbseq.plot as p
    return importlib.reload(p)


@pytest.fixture
def plot(monkeypatch):
    return _load_plot_module(monkeypatch)


@pytest.fixture
def sample_features_adata():
    n_obs, n_vars = 5, 3
    X = np.random.poisson(1, (n_obs, n_vars)).astype(np.float32)
    obs = pd.DataFrame({
        'group': ['A', 'B', 'A', 'B', 'A'],
        'num_features': [1, 2, 1, 2, 3],
        'log10_total_feature_counts': np.random.rand(n_obs),
        'log2_ratio_2nd_1st_feature': np.random.rand(n_obs)
    }, index=[f'cell{i}' for i in range(n_obs)])
    var = pd.DataFrame(index=[f'g{i}' for i in range(n_vars)])
    return AnnData(X=X, obs=obs, var=var)


@pytest.fixture
def adata_with_adj():
    n = 4
    X = np.random.poisson(1, (n, 3)).astype(np.float32)
    obs = pd.DataFrame({'color': ['A', 'B', 'A', 'B']}, index=[f'c{i}' for i in range(n)])
    var = pd.DataFrame(index=[f'g{i}' for i in range(3)])
    ad = AnnData(X=X, obs=obs, var=var)
    ad.obsm['adjacency'] = pd.DataFrame(np.eye(n), index=ad.obs_names, columns=ad.obs_names)
    return ad


def test_dotplot_returns_ax(plot):
    sizes = pd.DataFrame([[1, 2], [3, 4]], index=['a', 'b'], columns=['x', 'y'])
    colors = pd.DataFrame([[0.1, 0.2], [0.3, 0.4]], index=['a', 'b'], columns=['x', 'y'])
    ax = plot.dotplot(sizes, colors, return_ax=True, cluster=False)
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)


def test_dotplot_invalid_cluster_on(plot):
    df = pd.DataFrame(np.ones((2, 2)), index=['a', 'b'], columns=['x', 'y'])
    with pytest.raises(ValueError):
        plot.dotplot(df, df, cluster_on='invalid')


def test_corrfunc_annotation(plot):
    fig, ax = plt.subplots()
    x = np.arange(5)
    y = np.arange(5)
    plot.corrfunc(x, y, ax=ax, method='spearman')
    assert ax.texts and 'ρ =' in ax.texts[0].get_text()
    plt.close(fig)


def test_square_plot_adds_line(plot):
    fig, ax = plt.subplots()
    x = pd.Series([1, 2, 3, 4])
    y = pd.Series([1, 3, 2, 4])
    plot.square_plot(x, y, ax=ax, show=False, corr='spearman')
    assert any('r=' in txt.get_text() for txt in ax.texts)
    assert len(ax.lines) >= 1
    plt.close(fig)


def test_plot_adj_matr_missing(plot, sample_features_adata):
    with pytest.raises(ValueError):
        plot.plot_adj_matr(sample_features_adata)


def test_plot_adj_matr_basic(plot, adata_with_adj):
    plot.plot_adj_matr(adata_with_adj, row_colors='color', col_colors='color')
    plt.close('all')


def test_plot_top2ratio_counts(plot, sample_features_adata):
    g = plot.plot_top2ratio_counts(sample_features_adata, show=False)
    from seaborn.axisgrid import JointGrid
    assert isinstance(g, JointGrid)
    plt.close(g.fig)


def test_plot_num_features(plot, sample_features_adata):
    ax = plot.plot_num_features(sample_features_adata, show=False)
    assert isinstance(ax, plt.Axes)
    plt.close(ax.figure)
