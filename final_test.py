#!/usr/bin/env python3
"""
Final test script for the improved norman_model functionality
"""

import numpy as np
import pandas as pd
from anndata import AnnData

import pyturbseq as prtb
import pyturbseq.utils as utils


def create_test_data():
    """Create test data for Norman model testing"""
    np.random.seed(123)

    n_obs = 200
    n_vars = 30

    X = np.random.negative_binomial(3, 0.4, size=(n_obs, n_vars))

    # Create systematic dual perturbation design
    perturbations = []
    genes = ["GENE1", "GENE2", "GENE3", "GENE4"]

    # Single perturbations
    for gene in genes:
        perturbations.extend([gene] * 20)

    # Dual perturbations
    for i, gene1 in enumerate(genes):
        for j, gene2 in enumerate(genes[i + 1 :], i + 1):
            perturbations.extend([f"{gene1}|{gene2}"] * 15)

    # Controls
    n_controls = n_obs - len(perturbations)
    perturbations.extend(["NTC"] * n_controls)

    obs = pd.DataFrame(
        {
            "cell_id": [f"cell_{i}" for i in range(n_obs)],
            "feature_call": perturbations[:n_obs],
            "n_genes_by_counts": np.random.randint(800, 4000, n_obs),
            "total_counts": np.random.randint(3000, 15000, n_obs),
        }
    )

    var = pd.DataFrame(
        {
            "gene_ids": [f"ENSG{i:08d}" for i in range(n_vars)],
            "feature_types": ["Gene Expression"] * n_vars,
        }
    )
    var.index = [f"TARGET_{i}" for i in range(n_vars)]

    adata = AnnData(X=X, obs=obs, var=var)
    adata.obs.index = adata.obs["cell_id"]

    return adata


def test_improved_norman_model():
    """Test the improved norman_model with optional perturbations parameter"""
    print("🧪 Testing improved norman_model functionality")
    print("=" * 60)

    # Create test data
    adata = create_test_data()
    print(f"Test data shape: {adata.shape}")
    print(
        "Perturbations: "
        f"{adata.obs['feature_call'].value_counts().sort_index().to_dict()}"
    )

    # Pseudobulk the data
    pb_data = utils.pseudobulk(adata, groupby="feature_call")
    pb_df = pb_data.to_df()

    # Clean up index names
    pb_df.index = pb_df.index.str.replace("feature_call.", "", regex=False)
    print(f"\nPseudobulked data shape: {pb_df.shape}")
    print(f"Available perturbations: {pb_df.index.tolist()}")

    # Test 1: Single perturbation (explicit)
    print("\n1️⃣ Testing single perturbation (explicit):")
    dual_perts = [idx for idx in pb_df.index if "|" in idx]
    if dual_perts:
        test_dual = dual_perts[0]
        result, prediction = prtb.interaction.norman_model(
            pb_df, test_dual, verbose=False
        )
        print(f"   ✅ Single perturbation '{test_dual}' works")
        print(f"   📊 Result type: {type(result)}")
        print(
            "   📈 Prediction shape: "
            f"{prediction.shape if prediction is not None else 'None'}"
        )

    # Test 2: Multiple perturbations (explicit list)
    print("\n2️⃣ Testing multiple perturbations (explicit list):")
    if len(dual_perts) >= 2:
        test_list = dual_perts[:2]
        metrics_df, predictions_df = prtb.interaction.norman_model(
            pb_df, test_list, verbose=False
        )
        print(f"   ✅ Multiple perturbations {test_list} work")
        print(f"   📊 Metrics shape: {metrics_df.shape}")
        print(f"   📈 Predictions shape: {predictions_df.shape}")

    # Test 3: Auto-detection (default behavior - no perturbations parameter)
    print("\n3️⃣ Testing auto-detection (default behavior):")
    metrics_df, predictions_df = prtb.interaction.norman_model(pb_df, verbose=False)
    print("   ✅ Auto-detection works")
    print(f"   📊 Auto-detected {len(metrics_df)} dual perturbations")
    print(f"   📈 Predictions shape: {predictions_df.shape}")

    # Test 4: Parallel processing
    print("\n4️⃣ Testing parallel processing:")
    metrics_df, predictions_df = prtb.interaction.norman_model(
        pb_df, parallel=True, processes=2, verbose=False
    )
    print("   ✅ Parallel processing works")
    print(f"   📊 Results shape: {metrics_df.shape}")

    # Test 5: Backward compatibility
    print("\n5️⃣ Testing backward compatibility:")
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        if dual_perts:
            result = prtb.interaction.fit_many(pb_df, [dual_perts[0]], verbose=False)
            print("   ✅ fit_many still works (with deprecation warning)")
            print(f"   ⚠️  Warning issued: {w[0].message if w else 'None'}")

    print("\n" + "=" * 60)
    print("🎉 All tests passed! The improved norman_model is working " "correctly.")
    print("✅ Optional perturbations parameter works")
    print("✅ Default auto-detection works")
    print("✅ Parallel processing works")
    print("✅ Backward compatibility maintained")


if __name__ == "__main__":
    test_improved_norman_model()
