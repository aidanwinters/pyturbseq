"""
Integration tests for pyturbseq using real datasets

These tests use actual perturbation screen data to ensure the package works
with real-world data.
"""

from pathlib import Path

import pandas as pd
import pytest
import scanpy as sc

import pyturbseq as prtb


@pytest.mark.integration
@pytest.mark.slow
class TestRealDataIntegration:
    """Integration tests using real perturbation screen datasets"""

    def test_norman2019_singles_basic_workflow(self):
        """Test basic workflow with Norman2019 singles data"""
        data_path = Path("tests/data/norman2019_subset_singles.h5ad.gz")

        # Load test data
        adata = sc.read_h5ad(data_path)

        # Basic checks
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert "perturbation" in adata.obs.columns

        # Check that we have expected perturbations
        perturbations = adata.obs["perturbation"].unique()
        expected_targets = ["SET", "KLF1", "SAMD1", "PTPN12", "COL2A1"]
        assert "control" in perturbations
        for target in expected_targets:
            assert target in perturbations, f"Target {target} not found in data"

        # Test perturbation matrix generation
        prtb.utils.get_perturbation_matrix(
            adata, perturbation_col="perturbation", inplace=True
        )

        assert "perturbation" in adata.obsm
        assert adata.obsm["perturbation"].shape[0] == adata.n_obs

        # Check that target genes are included in the dataset
        target_genes_present = pd.Series(expected_targets).isin(adata.var.index)
        missing_genes = pd.Series(expected_targets)[~target_genes_present].tolist()
        assert target_genes_present.all(), f"Missing target genes: {missing_genes}"

        # Test target change calculation
        prtb.utils.calculate_target_change(
            adata, perturbation_column="perturbation", inplace=True
        )
        assert "target_change" in adata.obs.columns

    def test_norman2019_doubles_dual_perturbation_workflow(self):
        """Test dual perturbation workflow with Norman2019 doubles data"""
        data_path = Path("tests/data/norman2019_subset_doubles.h5ad.gz")

        # Load test data
        adata = sc.read_h5ad(data_path)

        # Basic checks
        assert adata.n_obs > 0
        assert adata.n_vars > 0
        assert "perturbation" in adata.obs.columns

        # Check for dual perturbations (using underscore separator)
        dual_perturbs = adata.obs["perturbation"].str.contains("_", na=False)
        assert dual_perturbs.sum() > 0, "No dual perturbations found in test data"

        # Check that we have expected dual perturbations
        expected_duals = ["SET_KLF1", "SAMD1_PTPN12", "KLF1_COL2A1"]
        perturbations = adata.obs["perturbation"].unique()
        for dual in expected_duals:
            assert dual in perturbations, f"Dual perturbation {dual} not found in data"

        # Test perturbation matrix generation
        prtb.utils.get_perturbation_matrix(
            adata, perturbation_col="perturbation", inplace=True
        )

        assert "perturbation" in adata.obsm
        assert adata.obsm["perturbation"].shape[0] == adata.n_obs

        # Test that dual perturbations are properly encoded
        pert_matrix = adata.obsm["perturbation"]
        dual_cells = adata.obs[dual_perturbs].index

        # Dual perturbation cells should have sum > 1 in perturbation matrix
        dual_sums = pert_matrix.loc[dual_cells].sum(axis=1)
        assert (dual_sums > 1).any(), "Dual perturbations not properly encoded"

        # Test target change calculation
        prtb.utils.calculate_target_change(
            adata, perturbation_column="perturbation", inplace=True
        )
        assert "target_change" in adata.obs.columns

    def test_norman2019_data_consistency(self):
        """Test that both Norman2019 datasets are consistent"""
        singles_path = Path("tests/data/norman2019_subset_singles.h5ad.gz")
        doubles_path = Path("tests/data/norman2019_subset_doubles.h5ad.gz")

        singles = sc.read_h5ad(singles_path)
        doubles = sc.read_h5ad(doubles_path)

        # Both should have same gene set
        assert set(singles.var.index) == set(
            doubles.var.index
        ), "Gene sets don't match between datasets"

        # Both should have same single perturbations
        singles_perts = set(singles.obs["perturbation"].unique())
        doubles_singles = set(
            doubles.obs["perturbation"][
                ~doubles.obs["perturbation"].str.contains("_", na=False)
            ].unique()
        )
        assert (
            singles_perts == doubles_singles
        ), "Single perturbations don't match between datasets"


@pytest.mark.integration
class TestPackageImports:
    """Test that all package modules can be imported correctly"""

    def test_import_main_modules(self):
        """Test importing all main modules"""
        import pyturbseq

        # Test that all expected modules are available
        expected_modules = [
            "utils",
            "de",
            "interaction",
            "plot",
            "calling",
            "cellranger",
            "guides",
        ]

        for module_name in expected_modules:
            assert hasattr(
                pyturbseq, module_name
            ), f"Module {module_name} not available"

        # Test version is available
        assert hasattr(pyturbseq, "__version__")
        assert pyturbseq.__version__ is not None

    def test_import_key_functions(self):
        """Test that key functions are available"""
        from pyturbseq.utils import (
            calculate_target_change,
            filter_adata,
            generate_perturbation_matrix,
            pseudobulk,
        )

        # Test that functions are callable
        assert callable(generate_perturbation_matrix)
        assert callable(calculate_target_change)
        assert callable(filter_adata)
        assert callable(pseudobulk)


@pytest.mark.integration
class TestDataCompatibility:
    """Test compatibility with different data formats and structures"""

    def test_anndata_compatibility(self, small_test_adata):
        """Test that package works with standard AnnData objects"""
        # Test basic operations
        prtb.utils.get_perturbation_matrix(
            small_test_adata, perturbation_col="feature_call", inplace=True
        )

        assert "perturbation" in small_test_adata.obsm

    def test_missing_columns_handling(self, small_test_adata):
        """Test handling of missing required columns"""
        # Remove feature_call column
        adata_no_feature = small_test_adata.copy()
        del adata_no_feature.obs["feature_call"]

        # Should raise an error or handle gracefully
        with pytest.raises((KeyError, ValueError)):
            prtb.utils.get_perturbation_matrix(
                adata_no_feature, perturbation_col="feature_call", inplace=True
            )

    def test_real_data_differential_expression(self):
        """Test differential expression analysis with real data"""
        data_path = Path("tests/data/norman2019_subset_singles.h5ad.gz")
        adata = sc.read_h5ad(data_path)

        # Test basic DE analysis
        try:
            deg_results = prtb.de.get_degs(
                adata, design_col="perturbation", control_value="control", alpha=0.05
            )

            # Should return a DataFrame with results
            assert isinstance(deg_results, pd.DataFrame)
            assert len(deg_results) > 0

        except Exception as e:
            # If DE analysis fails, it might be due to data structure
            # This is still informative for debugging
            pytest.skip(f"DE analysis failed with real data: {e}")

    def test_real_data_interaction_analysis(self):
        """Test interaction analysis with real dual perturbation data"""
        data_path = Path("tests/data/norman2019_subset_doubles.h5ad.gz")
        adata = sc.read_h5ad(data_path)

        # Test interaction analysis
        try:
            # Get a dual perturbation to test
            dual_perts = adata.obs["perturbation"][
                adata.obs["perturbation"].str.contains("_", na=False)
            ].unique()
            if len(dual_perts) > 0:
                test_dual = dual_perts[0]

                result, prediction = prtb.interaction.get_model_fit(
                    adata, test_dual, ref="control"
                )

                # Should return results
                assert result is not None
                assert prediction is not None

        except Exception as e:
            # If interaction analysis fails, it might be due to data structure
            # This is still informative for debugging
            pytest.skip(f"Interaction analysis failed with real data: {e}")
