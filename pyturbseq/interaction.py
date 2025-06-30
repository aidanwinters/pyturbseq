import concurrent.futures

import numpy as np
import pandas as pd
import scanpy as sc
from dcor import distance_correlation
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from tqdm import tqdm


def norman_model(
    data,
    double,
    method="robust",
    targets=None,
    delim="|",
    ref="NTC",
    plot=True,
    verbose=True,
):
    """
    Tom Norman's approach rewritten by Aidan Winters
    Assumes no observations but many features.
    Assumes that perturbation is index of observations (this is the case for pseudobulked to perturbation)
    Assumes data is pd DataFrame

    Args:
        data: DataFrame with perturbations as index and genes as columns
        double: String representing dual perturbation (e.g., "GENE1|GENE2")
        method: "robust" for TheilSenRegressor or "linear" for LinearRegression
        targets: List of target genes to analyze (default: all genes)
        delim: Delimiter for splitting dual perturbations
        ref: Reference/control value (not used in current implementation)
        plot: Whether to generate plots (placeholder for future)
        verbose: Whether to print progress

    Returns:
        tuple: (metrics_dict, predictions_array)
    """
    # if data is anndata then make it df
    if type(data) == sc.AnnData:
        print("Found AnnData, densifying to df. This may take a while... ")
        data = data.to_df()

    A, B = double.split(delim)
    if verbose:
        print("Fitting model for", double)
        print("\tA:", A, "B:", B)

    # confirm that A, B, and double are in data.index
    if pd.Series([A, B, double]).isin(data.index).sum() != 3:
        print("Error: not all perturbations in data")
        return None, None

    # confirm all targets are in data
    if targets is None:
        targets = data.columns
    else:
        targets = [t for t in targets if t in data.columns]

    singlesX = data.loc[[A, B], targets].T
    aX = data.loc[A, targets].T
    bX = data.loc[B, targets].T
    doubleX = data.loc[double, targets].T

    if method == "robust":
        regr = TheilSenRegressor(
            fit_intercept=False, max_subpopulation=1e5, max_iter=1000, random_state=1000
        )
    else:
        regr = LinearRegression(fit_intercept=False)

    X = singlesX
    y = doubleX

    regr.fit(X, y)

    Z = regr.predict(X)

    out = {}
    out["perturbation"] = double
    out["a"] = A
    out["b"] = B

    # Get corrs
    out["corr_a"] = spearmanr(aX, doubleX)[0]
    out["corr_b"] = spearmanr(bX, doubleX)[0]
    out["corr_sum"] = spearmanr(aX + bX, doubleX)[0]
    out["corr_a_b"] = spearmanr(aX, bX)[0]

    # how well correlated is the fit to the double perturbation
    out["fit_spearmanr"] = spearmanr(Z, doubleX)[0]
    out["fit_pearsonr"] = pearsonr(Z, doubleX)[0]

    # other distance metrics
    out["fit_cosine_dist"] = distance.cosine(Z, doubleX)

    out["coef_a"] = regr.coef_[0]
    out["coef_b"] = regr.coef_[1]
    out["coef_ratio"] = regr.coef_[0] / regr.coef_[1]
    out["coef_difference"] = regr.coef_[0] - regr.coef_[1]
    out["coef_sum"] = regr.coef_[0] + regr.coef_[1]
    out["log2_ratio_coefs"] = np.log2(abs(regr.coef_[0]) / abs(regr.coef_[1]))
    out["log10_ratio_coefs"] = np.log10(abs(regr.coef_[0]) / abs(regr.coef_[1]))
    out["abs_log10_ratio_coefs"] = abs(
        np.log10(abs(regr.coef_[0]) / abs(regr.coef_[1]))
    )

    out["coef_norm"] = np.mean([np.abs(out["coef_a"]), np.abs(out["coef_b"])])
    out["coef_norm2"] = np.sqrt(out["coef_a"] ** 2 + out["coef_b"] ** 2)
    out["score"] = regr.score(X, y)

    # get residual
    out["median_abs_residual"] = np.median(abs(doubleX - Z))
    out["rss"] = np.sum((doubleX - Z) ** 2)

    # Tom's metrics
    out["dcor_AnB_AB"] = distance_correlation(
        singlesX, doubleX
    )  # distance correlation between [A,B] and AB (the double perturbation)
    out["dcor_A_B"] = distance_correlation(
        aX, bX
    )  # distance correlation between A and B
    out["dcor_AnB_fit"] = distance_correlation(
        singlesX, Z
    )  # distance correlation between the [A, B] and predicted AB
    out["dcor_AB_fit"] = distance_correlation(
        doubleX, Z
    )  # distance correlation between AB and predicted AB
    out["dcor_A"] = distance_correlation(
        aX, doubleX
    )  # distance correlation between A and predicted AB
    out["dcor_B"] = distance_correlation(
        bX, doubleX
    )  # distance correlation between B and predicted AB
    out["dcor_A_fit"] = distance_correlation(
        aX, Z
    )  # distance correlation between A and predicted AB
    out["dcor_B_fit"] = distance_correlation(
        bX, Z
    )  # distance correlation between B and predicted AB
    out["min_dcor"] = min(out["dcor_A"], out["dcor_B"])
    out["max_dcor"] = max(out["dcor_A"], out["dcor_B"])
    out["dcor_ratio"] = out["min_dcor"] / out["max_dcor"]

    return out, Z


def fit_many(
    data,
    perturbations=None,
    delim="|",
    ref="NTC",
    parallel=False,
    processes=4,
    **kwargs,
):
    """
    Unified function to fit Norman model to multiple perturbations.

    Args:
        data: DataFrame with perturbations as index and genes as columns
        perturbations:
            - None: Run on all dual perturbations (containing delim, excluding ref)
            - str: Run on single perturbation
            - list: Run on specified list of perturbations
        delim: Delimiter for identifying dual perturbations
        ref: Reference/control value to exclude from analysis
        parallel: Whether to use parallel processing
        processes: Number of processes for parallel execution
        **kwargs: Additional arguments passed to norman_model

    Returns:
        tuple: (metrics_DataFrame, predictions_DataFrame)
    """
    # if data is anndata then make it df
    if type(data) == sc.AnnData:
        print("Found AnnData, densifying to df. This may take a while... ")
        data = data.to_df()

    # Determine which perturbations to analyze
    if perturbations is None:
        # Run on all dual perturbations (exclude control)
        all_perts = data.index.tolist()
        perturbations = [p for p in all_perts if delim in p and ref not in p]
        print(f"Auto-detected {len(perturbations)} dual perturbations to analyze")
    elif isinstance(perturbations, str):
        # Single perturbation
        perturbations = [perturbations]
    elif isinstance(perturbations, list):
        # List of perturbations provided
        pass
    else:
        raise ValueError("perturbations must be None, str, or list")

    # Validate that all perturbations exist in data
    missing_perts = [p for p in perturbations if p not in data.index]
    if missing_perts:
        print(
            f"Warning: {len(missing_perts)} perturbations not found in data: {missing_perts[:5]}..."
        )
        perturbations = [p for p in perturbations if p in data.index]

    if len(perturbations) == 0:
        raise ValueError("No valid perturbations found to analyze")

    print(f"Analyzing {len(perturbations)} perturbations...")

    # Run analysis
    if parallel and len(perturbations) > 1:
        return _fit_many_parallel(data, perturbations, processes, **kwargs)
    else:
        return _fit_many_sequential(data, perturbations, **kwargs)


def _fit_many_sequential(data, perturbations, **kwargs):
    """Sequential execution of norman_model for multiple perturbations."""
    results = []
    predictions = []

    for pert in tqdm(perturbations, desc="Fitting models"):
        out, Z = norman_model(data, pert, verbose=False, **kwargs)
        if out is not None:
            results.append(out)
            predictions.append(Z)

    # Convert to DataFrames
    metrics_df = pd.DataFrame(results).set_index("perturbation")
    predictions_df = pd.DataFrame(
        predictions, index=[r["perturbation"] for r in results], columns=data.columns
    )

    return metrics_df, predictions_df


def _fit_many_parallel(data, perturbations, processes, **kwargs):
    """Parallel execution of norman_model for multiple perturbations."""
    results = []
    predictions = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        # Submit all jobs
        futures = [
            executor.submit(norman_model, data, pert, verbose=False, **kwargs)
            for pert in perturbations
        ]

        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(perturbations),
            desc="Fitting models",
        ):
            out, Z = future.result()
            if out is not None:
                results.append(out)
                predictions.append(Z)

    # Convert to DataFrames
    metrics_df = pd.DataFrame(results).set_index("perturbation")
    predictions_df = pd.DataFrame(
        predictions, index=[r["perturbation"] for r in results], columns=data.columns
    )

    return metrics_df, predictions_df


# Legacy compatibility - keep for backwards compatibility but deprecate
def get_model_fit(*args, **kwargs):
    """Deprecated: Use norman_model instead."""
    import warnings

    warnings.warn(
        "get_model_fit is deprecated. Use norman_model instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return norman_model(*args, **kwargs)
