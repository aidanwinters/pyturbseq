import numpy as np
import pandas as pd
import pytest
from anndata import AnnData

import pyturbseq.de as de


@pytest.fixture
def simple_adata():
    n_obs = 6
    n_var = 2
    X = np.random.poisson(5, (n_obs, n_var)).astype(np.float32)
    obs = pd.DataFrame({
        'condition': ['A', 'A', 'B', 'B', 'C', 'C'],
        'cov': np.random.randint(0, 2, n_obs)
    }, index=[f'cell{i}' for i in range(n_obs)])
    var = pd.DataFrame(index=[f'gene{i}' for i in range(n_var)])
    return AnnData(X=X, obs=obs, var=var)


def test_get_degs_basic(monkeypatch, simple_adata):
    records = {}

    def DummyInference(n_cpus):
        records['n_cpus'] = n_cpus
        return 'inference'

    class DummyDDS:
        def __init__(self, counts, metadata, design_factors, inference, min_replicates, min_mu, ref_level, refit_cooks, quiet):
            records['counts_shape'] = counts.shape
            records['design_factors'] = design_factors
            records['ref_level'] = ref_level
            records['quiet'] = quiet
            records['inference'] = inference
        def deseq2(self):
            records['deseq2_called'] = True

    class DummyStats:
        def __init__(self, dds, contrast, quiet, inference):
            records['contrast'] = contrast
            self.results_df = pd.DataFrame({
                'log2FoldChange': [1.0, -1.0],
                'padj': [0.01, 0.5]
            }, index=['gene0', 'gene1'])
        def summary(self):
            records['summary_called'] = True

    monkeypatch.setattr(de, 'DefaultInference', DummyInference)
    monkeypatch.setattr(de, 'DeseqDataSet', DummyDDS)
    monkeypatch.setattr(de, 'DeseqStats', DummyStats)

    df = de.get_degs(simple_adata, design_col='condition', ref_val='A', n_cpus=2, quiet=True)

    assert records['n_cpus'] == 2
    assert records['design_factors'] == 'condition'
    assert records['ref_level'] == ['condition', 'A']
    assert records['contrast'] == ['condition', 'B', 'C', 'A']
    assert records['deseq2_called']
    assert records['summary_called']
    assert 'significant' in df.columns
    assert bool(df.loc['gene0', 'significant']) is True
    assert bool(df.loc['gene1', 'significant']) is False


def test_get_all_degs_parallel(monkeypatch, simple_adata):
    call_args = []

    def dummy_get_degs(adata, design_col, ref_val=None, n_cpus=None, quiet=False, **kwargs):
        call_args.append({'labels': sorted(adata.obs[design_col].unique()), 'n_cpus': n_cpus})
        return pd.DataFrame({
            'log2FoldChange': [1.0, -1.0],
            'padj': [0.01, 0.5],
            'significant': [True, False]
        }, index=['gene0', 'gene1'])

    class DummyParallel:
        def __init__(self, n_jobs):
            self.n_jobs = n_jobs
        def __call__(self, tasks):
            return [task() for task in tasks]

    def dummy_delayed(func):
        def wrapper(*args, **kwargs):
            return lambda: func(*args, **kwargs)
        return wrapper

    monkeypatch.setattr(de, 'get_degs', dummy_get_degs)
    monkeypatch.setattr(de, 'Parallel', DummyParallel)
    monkeypatch.setattr(de, 'delayed', dummy_delayed)
    monkeypatch.setattr(de, 'tqdm', lambda x, disable=False: x)
    monkeypatch.setattr(de, 'multiprocessing', type('MP', (), {'cpu_count': staticmethod(lambda: 8)}))

    res = de.get_all_degs(simple_adata, design_col='condition', reference='A', parallel=True, n_cpus=4, max_workers=2, quiet=True)

    assert len(call_args) == 2
    assert all(c['n_cpus'] == 4 for c in call_args)
    assert set(res['condition']) == {'B', 'C'}
    assert 'gene' in res.columns
    assert res.shape[0] == 4


def test_get_all_degs_sync_with_conditions(monkeypatch, simple_adata):
    calls = []

    def dummy_get_degs(adata, design_col, ref_val=None, n_cpus=None, quiet=False, **kwargs):
        calls.append(sorted(adata.obs[design_col].unique()))
        return pd.DataFrame({'log2FoldChange': [1.0], 'padj': [0.01], 'significant': [True]}, index=['gene0'])

    monkeypatch.setattr(de, 'get_degs', dummy_get_degs)

    res = de.get_all_degs(simple_adata, design_col='condition', reference='A', conditions=['B'], parallel=False, n_cpus=1, quiet=True)

    assert calls == [['A', 'B']]
    assert res['condition'].unique().tolist() == ['B']
    assert res['gene'].tolist() == ['gene0']
