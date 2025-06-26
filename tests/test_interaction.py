import numpy as np
import pandas as pd
import pytest

from pyturbseq import interaction


@pytest.fixture
def simple_df():
    data = pd.DataFrame(
        {
            "gene1": [1.0, 2.0, 3.0],
            "gene2": [2.0, 4.0, 6.0],
        },
        index=["A|NTC", "B|NTC", "A|B"],
    )
    return data


def test_get_singles():
    assert interaction.get_singles("A|B", ref="NTC") == ("A|NTC", "B|NTC")


def test_get_model_fit(simple_df):
    out, Z = interaction.get_model_fit(
        simple_df,
        "A|B",
        method="linear",
        targets=["gene1", "gene2"],
        ref="NTC",
        plot=False,
        verbose=False,
    )
    # ensure key metrics present and predictions length matches number of genes
    for key in ["coef_a", "coef_b", "fit_pearsonr", "score"]:
        assert key in out
    assert len(Z) == simple_df.shape[1]


def test_fit_many(simple_df):
    res = interaction.fit_many(
        simple_df,
        ["A|B"],
        method="linear",
        targets=["gene1", "gene2"],
        ref="NTC",
        plot=False,
        verbose=False,
    )
    assert "coef_a" in res.columns
    assert res.index[0] == "A|B"


def test_model_fit_wrapper(simple_df):
    out = interaction.model_fit_wrapper(
        simple_df,
        "A|B",
        {
            "method": "linear",
            "targets": ["gene1", "gene2"],
            "ref": "NTC",
            "plot": False,
            "verbose": False,
        },
    )
    assert isinstance(out, dict)
    assert out["perturbation"] == "A|B"


def test_fit_many_parallel(simple_df):
    res = interaction.fit_many_parallel(
        simple_df,
        ["A|B"],
        processes=1,
        method="linear",
        targets=["gene1", "gene2"],
        ref="NTC",
        plot=False,
        verbose=False,
    )
    assert "coef_b" in res.columns
    assert list(res.index) == ["A|B"]


def test_get_val_found(simple_df):
    val = interaction.get_val(simple_df, "A|NTC", "gene1")
    assert val == 1.0


def test_get_val_missing(simple_df):
    val = interaction.get_val(simple_df, "missing", "gene1")
    assert np.isnan(val)


def test_run_permutation_deterministic():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 2])
    coef1 = interaction.run_permutation(X, y, seed=123)
    coef2 = interaction.run_permutation(X, y, seed=123)
    assert np.allclose(coef1, coef2)
    assert coef1.shape[0] == X.shape[1]


def test_estimate_alpha_empirical():
    y = np.array([1, 10, 1, 10, 1, 10])
    alpha, mean_y, var_y = interaction.estimate_alpha_empirical(y)
    assert alpha is not None
    assert var_y > mean_y


def test_breakdown_double_wRef():
    term2pert, pert2term, split = interaction.breakdown_double_wRef("geneA|geneB", ref="NTC")
    assert term2pert["ref"] == "NTC|NTC"
    assert pert2term["NTC|geneA"] == "a"
    assert split == ["NTC", "geneA", "geneB"]


def test_breakdown_triple_wRef():
    term2pert, pert2term, split = interaction.breakdown_triple_wRef("A|B|C", ref="NTC")
    assert term2pert["abc"] == "A|B|C"
    assert pert2term["A|B|NTC"] == "ab"
    assert split == ["NTC", "A", "B", "C"]


def test_breakdown_perturbation_invalid():
    with pytest.raises(ValueError):
        interaction.breakdown_perturbation("A|B|C|D", 4, ref="NTC")
