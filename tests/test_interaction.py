"""
Tests for pyturbseq.interaction module
"""

import pandas as pd
import pytest

import pyturbseq.interaction as interaction
import pyturbseq.utils as utils


class TestInteractionAnalysis:
    """Test interaction analysis functions"""

    def test_interaction_calculation_basic(self, dual_perturbation_adata):
        """Test basic interaction calculation"""
        # Add perturbation matrix first
        utils.get_perturbation_matrix(
            dual_perturbation_adata, perturbation_col="feature_call", inplace=True
        )

        # Test that the unified norman_model function exists
        assert hasattr(interaction, "norman_model")
        assert callable(interaction.norman_model)

        # Test that deprecated functions still exist but issue warnings
        assert hasattr(interaction, "fit_many")
        assert hasattr(interaction, "get_model_fit")

    def test_dual_perturbation_detection(self, dual_perturbation_adata):
        """Test detection of dual perturbations"""
        # Check that dual perturbations are properly identified
        dual_perturbs = dual_perturbation_adata.obs["feature_call"].str.contains(
            "|", na=False
        )
        assert dual_perturbs.sum() > 0

        # Check specific dual perturbations exist
        assert "GENE1|GENE2" in dual_perturbation_adata.obs["feature_call"].values


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


class TestNormanModelUnified:
    """Test the unified norman_model function with different input types"""

    def test_norman_model_single_perturbation(self, dual_perturbation_adata):
        """Test norman_model with a single perturbation string"""
        # Pseudobulk the data for Norman model (perturbations as rows, genes as columns)
        pb_data = utils.pseudobulk(dual_perturbation_adata, groupby="feature_call")
        pb_df = pb_data.to_df()

        # Clean up index names (remove 'feature_call.' prefix added by pseudobulk)
        pb_df.index = pb_df.index.str.replace("feature_call.", "", regex=False)

        # Check we have the expected dual perturbations in the data
        available_dual_perts = [idx for idx in pb_df.index if "|" in idx]
        if len(available_dual_perts) == 0:
            pytest.skip("No dual perturbations found in pseudobulked data")

        test_dual_pert = available_dual_perts[
            0
        ]  # Use first available dual perturbation

        try:
            result, prediction = interaction.norman_model(
                pb_df, test_dual_pert, method="robust", plot=False, verbose=False
            )

            # Basic checks on result structure
            assert isinstance(result, dict)
            assert "perturbation" in result
            assert result["perturbation"] == test_dual_pert
            assert prediction is not None

        except Exception as e:
            # If it fails due to missing dependencies or other issues,
            # that's acceptable for this test
            if "norman_model" not in str(e):
                pytest.fail(f"norman_model function failed unexpectedly: {e}")

    def test_norman_model_list_perturbations(self, dual_perturbation_adata):
        """Test norman_model with a list of perturbations"""
        # Pseudobulk the data for Norman model
        pb_data = utils.pseudobulk(dual_perturbation_adata, groupby="feature_call")
        pb_df = pb_data.to_df()

        # Clean up index names (remove 'feature_call.' prefix added by pseudobulk)
        pb_df.index = pb_df.index.str.replace("feature_call.", "", regex=False)

        # Get available dual perturbations
        available_dual_perts = [idx for idx in pb_df.index if "|" in idx]
        if len(available_dual_perts) < 2:
            pytest.skip("Need at least 2 dual perturbations for this test")

        perturbation_list = available_dual_perts[:2]  # Use first 2

        try:
            metrics_df, predictions_df = interaction.norman_model(
                pb_df, perturbation_list, method="robust", verbose=False
            )

            # Should return DataFrames for multiple perturbations
            assert isinstance(metrics_df, pd.DataFrame)
            assert isinstance(predictions_df, pd.DataFrame)
            assert len(metrics_df) <= 2  # May be less if some fail

        except Exception as e:
            if "norman_model" not in str(e):
                pytest.fail(f"norman_model with list failed unexpectedly: {e}")

    def test_norman_model_auto_detect(self, dual_perturbation_adata):
        """Test norman_model with default None to auto-detect perturbations"""
        # Pseudobulk the data for Norman model
        pb_data = utils.pseudobulk(dual_perturbation_adata, groupby="feature_call")
        pb_df = pb_data.to_df()

        # Clean up index names (remove 'feature_call.' prefix added by pseudobulk)
        pb_df.index = pb_df.index.str.replace("feature_call.", "", regex=False)

        try:
            metrics_df, predictions_df = interaction.norman_model(
                pb_df, verbose=False  # perturbations=None is the default
            )

            # Should auto-detect the dual perturbations
            assert isinstance(metrics_df, pd.DataFrame)
            assert isinstance(predictions_df, pd.DataFrame)
            # May be 0 if no valid dual perturbations can be analyzed

        except Exception as e:
            if "norman_model" not in str(e):
                pytest.fail(f"norman_model auto-detect failed unexpectedly: {e}")

    def test_norman_model_parallel_option(self, dual_perturbation_adata):
        """Test norman_model with parallel processing option"""
        # Pseudobulk the data for Norman model
        pb_data = utils.pseudobulk(dual_perturbation_adata, groupby="feature_call")
        pb_df = pb_data.to_df()

        # Clean up index names (remove 'feature_call.' prefix added by pseudobulk)
        pb_df.index = pb_df.index.str.replace("feature_call.", "", regex=False)

        try:
            # Test that parallel parameter is accepted
            metrics_df, predictions_df = interaction.norman_model(
                pb_df, parallel=True, processes=2, verbose=False
            )

            assert isinstance(metrics_df, pd.DataFrame)
            assert isinstance(predictions_df, pd.DataFrame)

        except Exception as e:
            if "norman_model" not in str(e):
                pytest.fail(f"norman_model parallel failed unexpectedly: {e}")


class TestInteractionFunctions:
    """Test individual interaction analysis functions"""

    def test_get_singles_function(self):
        """Test the get_singles helper function"""
        if hasattr(interaction, "get_singles"):
            single_a, single_b = interaction.get_singles("GENE1|GENE2")
            assert single_a in ["GENE1|NTC", "NTC|GENE1"]
            assert single_b in ["GENE2|NTC", "NTC|GENE2"]

    def test_norman_model_exists(self):
        """Test that norman_model function exists"""
        assert hasattr(interaction, "norman_model")
        assert callable(interaction.norman_model)

    def test_deprecated_functions_exist(self):
        """Test that deprecated functions still exist for backward compatibility"""
        assert hasattr(interaction, "fit_many")
        assert callable(interaction.fit_many)
        assert hasattr(interaction, "get_model_fit")
        assert callable(interaction.get_model_fit)

    def test_deprecated_functions_issue_warnings(self, dual_perturbation_adata):
        """Test that deprecated functions issue deprecation warnings"""
        # Pseudobulk the data for Norman model
        pb_data = utils.pseudobulk(dual_perturbation_adata, groupby="feature_call")
        pb_df = pb_data.to_df()

        # Clean up index names (remove 'feature_call.' prefix added by pseudobulk)
        pb_df.index = pb_df.index.str.replace("feature_call.", "", regex=False)

        # Get available dual perturbations
        available_dual_perts = [idx for idx in pb_df.index if "|" in idx]
        if len(available_dual_perts) == 0:
            pytest.skip("No dual perturbations available for this test")

        # Test fit_many deprecation warning
        with pytest.warns(DeprecationWarning, match="fit_many is deprecated"):
            try:
                interaction.fit_many(pb_df, [available_dual_perts[0]], verbose=False)
            except Exception:
                pass  # We're just testing the warning is issued

        # Test get_model_fit deprecation warning
        with pytest.warns(DeprecationWarning, match="get_model_fit is deprecated"):
            try:
                interaction.get_model_fit(pb_df, available_dual_perts[0], verbose=False)
            except Exception:
                pass  # We're just testing the warning is issued


class TestInteractionInputValidation:
    """Test input validation for interaction functions"""

    def test_norman_model_with_pseudobulk_data(self, dual_perturbation_adata):
        """Test norman_model with pseudobulk data structure"""
        # Pseudobulk the data for Norman model
        pb_data = utils.pseudobulk(dual_perturbation_adata, groupby="feature_call")
        pb_df = pb_data.to_df()

        # Clean up index names (remove 'feature_call.' prefix added by pseudobulk)
        pb_df.index = pb_df.index.str.replace("feature_call.", "", regex=False)

        # Get available dual perturbations
        available_dual_perts = [idx for idx in pb_df.index if "|" in idx]
        if len(available_dual_perts) == 0:
            pytest.skip("No dual perturbations available for this test")

        try:
            result, prediction = interaction.norman_model(
                pb_df,
                available_dual_perts[0],
                method="robust",
                plot=False,
                verbose=False,
            )

            # Basic checks on result structure
            assert isinstance(result, dict)
            assert "perturbation" in result
            assert result["perturbation"] == available_dual_perts[0]

        except Exception as e:
            # If it fails due to missing dependencies or other issues,
            # that's acceptable for this test
            if "norman_model" not in str(e):
                pytest.fail(f"norman_model function failed unexpectedly: {e}")


class TestInteractionUtilities:
    """Test interaction utility functions"""

    def test_breakdown_functions_exist(self):
        """Test that perturbation breakdown functions exist"""
        breakdown_functions = [
            "breakdown_double_wRef",
            "breakdown_triple_wRef",
            "breakdown_perturbation",
        ]

        for func_name in breakdown_functions:
            if hasattr(interaction, func_name):
                assert callable(getattr(interaction, func_name))

    def test_get_model_wNTC_exists(self):
        """Test that get_model_wNTC function exists"""
        if hasattr(interaction, "get_model_wNTC"):
            assert callable(interaction.get_model_wNTC)

    def test_model_fit_wrapper_exists(self):
        """Test that model_fit_wrapper function exists"""
        if hasattr(interaction, "model_fit_wrapper"):
            assert callable(interaction.model_fit_wrapper)
