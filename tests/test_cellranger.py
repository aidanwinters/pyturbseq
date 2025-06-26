import pandas as pd
import numpy as np
import pytest
import math
from anndata import AnnData
from scipy.sparse import csr_matrix

from pyturbseq import cellranger


def test_parse_umi_float():
    """Parsing a float returns expected metrics."""
    result = cellranger.parse_umi(5.0)
    assert result["CR_total_umi"] == 5.0
    assert result["CR_max_umi"] == 5.0
    assert math.isnan(result["CR_ratio_2nd_1st"])


def test_parse_umi_string():
    """Parsing a pipe delimited string computes totals and ratios."""
    result = cellranger.parse_umi("3|1")
    assert result["CR_total_umi"] == 4
    assert result["CR_max_umi"] == 3
    assert result["CR_ratio_2nd_1st"] == pytest.approx(1/3)


def test_add_CR_umi_metrics():
    """UMI metrics are appended to adata.obs."""
    obs = pd.DataFrame({"num_umis": [5.0, "3|2", "1|1|1"]}, index=["c1", "c2", "c3"])
    var = pd.DataFrame(index=["g1", "g2"])
    ad = AnnData(X=csr_matrix(np.ones((3, 2))), obs=obs, var=var)

    out = cellranger.add_CR_umi_metrics(ad)

    assert {"CR_total_umi", "CR_max_umi", "CR_ratio_2nd_1st"}.issubset(out.obs.columns)
    assert out.obs.loc["c1", "CR_total_umi"] == 5.0
    assert out.obs.loc["c2", "CR_max_umi"] == 3
    assert out.obs.loc["c2", "CR_ratio_2nd_1st"] == pytest.approx(2/3)


def test_add_CR_sgRNA(tmp_path):
    """sgRNA calls from CSV are merged with adata.obs."""
    obs = pd.DataFrame(index=["cell1", "cell2", "cell3"])
    var = pd.DataFrame(index=["g1"])
    ad = AnnData(X=csr_matrix(np.ones((3, 1))), obs=obs, var=var)

    calls = pd.DataFrame({"sgRNA": ["g1", "g3"]}, index=["cell1", "cell3"])
    csv = tmp_path / "calls.csv"
    calls.to_csv(csv)

    new = cellranger.add_CR_sgRNA(ad, calls_file=str(csv), inplace=False, quiet=True)

    assert "sgRNA" not in ad.obs.columns
    assert "sgRNA" in new.obs.columns
    assert new.obs.loc["cell1", "sgRNA"] == "g1"
    assert pd.isna(new.obs.loc["cell2", "sgRNA"])


def test_add_CR_sgRNA_missing(tmp_path):
    """Missing sgRNA CSV raises ValueError."""
    obs = pd.DataFrame(index=["cell1"])
    var = pd.DataFrame(index=["g1"])
    ad = AnnData(X=csr_matrix(np.ones((1, 1))), obs=obs, var=var)
    with pytest.raises(ValueError):
        cellranger.add_CR_sgRNA(ad, calls_file=str(tmp_path / "missing.csv"))


def test_parse_CR_flex_metrics():
    """Various metric formats are converted to floats."""
    df = pd.DataFrame({
        "Metric Name": ["pct", "paren", "comma", "num"],
        "Metric Value": ["40.00%", "100 (25.00%)", "20,000", "5"]
    })

    out = cellranger.parse_CR_flex_metrics(df)
    assert np.isclose(out.loc[0, "Metric Value"], 0.4)
    assert np.isclose(out.loc[1, "Metric Value"], 0.25)
    assert np.isclose(out.loc[2, "Metric Value"], 20000.0)
    assert np.isclose(out.loc[3, "Metric Value"], 5.0)


def test_parse_CR_h5(monkeypatch, tmp_path):
    """Basic parse_CR_h5 workflow with guide calls and pattern."""
    h5 = tmp_path / "sample1.h5"
    h5.touch()

    obs = pd.DataFrame(index=["cell1", "cell2"])
    var = pd.DataFrame(index=["g1"])
    dummy = AnnData(X=csr_matrix(np.ones((2,1))), obs=obs, var=var)

    def fake_read(path, gex_only=False):
        assert str(path) == str(h5)
        return dummy.copy()

    monkeypatch.setattr(cellranger.sc, "read_10x_h5", fake_read)

    calls = pd.DataFrame({"sgRNA": ["g1"]}, index=["cell1"])
    csv = tmp_path / "calls.csv"
    calls.to_csv(csv)

    ad = cellranger.parse_CR_h5(str(h5), guide_call_csv=str(csv), pattern=r"(?P<sample>\w+)\.h5", quiet=True)

    assert ad.n_obs == 2
    assert "sample" in ad.obs.columns
    assert ad.obs.loc["cell1", "sgRNA"] == "g1"
    assert pd.isna(ad.obs.loc["cell2", "sgRNA"])


def test_parse_CR_h5_missing_file():
    with pytest.raises(ValueError):
        cellranger.parse_CR_h5("missing.h5")


def test_parse_CR_h5_missing_calls(monkeypatch, tmp_path):
    h5 = tmp_path / "sample1.h5"
    h5.touch()

    obs = pd.DataFrame(index=["cell1"])
    var = pd.DataFrame(index=["g1"])
    dummy = AnnData(X=csr_matrix(np.ones((1,1))), obs=obs, var=var)

    monkeypatch.setattr(cellranger.sc, "read_10x_h5", lambda path, gex_only=False: dummy.copy())

    with pytest.raises(ValueError):
        cellranger.parse_CR_h5(str(h5), guide_call_csv=str(tmp_path / "missing.csv"), quiet=True)

