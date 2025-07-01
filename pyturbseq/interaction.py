import concurrent.futures
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import scanpy as sc
from anndata import AnnData
from dcor import distance_correlation
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from tqdm import tqdm


def norman_model(
    data: Union[pd.DataFrame, AnnData],
    perturbations: Optional[Union[str, List[str]]] = None,
    method: str = "robust",
    targets: Optional[List[str]] = None,
    delim: str = "|",
    ref: str = "NTC",
    parallel: bool = False,
    processes: int = 4,
    plot: bool = True,
    verbose: bool = True,
    **kwargs,
) -> Union[Tuple[Dict[str, Any], np.ndarray], Tuple[pd.DataFrame, pd.DataFrame]]:
    """Tom Norman's approach for analyzing genetic interactions in perturbation data.

    This function can handle single perturbations or multiple perturbations with automatic
    detection and parallel processing support.

    Args:
        data: DataFrame with perturbations as index and genes as columns, or AnnData object
        perturbations:
            - str: Single dual perturbation to analyze (e.g., "GENE1|GENE2")
            - list: List of perturbations to analyze
            - None (default): Auto-detect all dual perturbations (containing delim, excluding ref)
        method: "robust" for TheilSenRegressor or "linear" for LinearRegression
        targets: List of target genes to analyze (default: all genes)
        delim: Delimiter for identifying/splitting dual perturbations
        ref: Reference/control value to exclude from auto-detection (default: "NTC")
        parallel: Whether to use parallel processing (only for multiple perturbations)
        processes: Number of processes for parallel execution
        plot: Whether to generate plots (placeholder for future, only used for single perturbations)
        verbose: Whether to print progress
        **kwargs: Additional arguments (for backwards compatibility)

    Returns:
        For single perturbation (str):
            tuple: (metrics_dict, predictions_array)
        For multiple perturbations (list or None):
            tuple: (metrics_DataFrame, predictions_DataFrame)
    """
    # Convert AnnData to DataFrame if needed
    if type(data) == sc.AnnData:
        if verbose:
            print("Found AnnData, densifying to df. This may take a while... ")
        data = data.to_df()

    # Handle single perturbation case
    if isinstance(perturbations, str):
        return _norman_model_single(
            data, perturbations, method, targets, delim, ref, plot, verbose
        )

    # Handle multiple perturbations case
    return _norman_model_multiple(
        data, perturbations, method, targets, delim, ref, parallel, processes, verbose
    )


def _norman_model_single(
    data: pd.DataFrame,
    double: str,
    method: str = "robust",
    targets: Optional[List[str]] = None,
    delim: str = "|",
    ref: str = "NTC",
    plot: bool = True,
    verbose: bool = True,
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """Internal function to handle single perturbation analysis.

    This is the original norman_model functionality.

    Args:
        data: DataFrame with perturbations as index and genes as columns.
        double: Single dual perturbation to analyze.
        method: "robust" for TheilSenRegressor or "linear" for LinearRegression.
        targets: List of target genes to analyze.
        delim: Delimiter for splitting dual perturbations.
        ref: Reference/control value.
        plot: Whether to generate plots.
        verbose: Whether to print progress.

    Returns:
        Tuple of (metrics_dict, predictions_array) or (None, None) if error.
    """
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

    singlesX = data.loc[[A, B], targets].T.astype(np.float64)
    aX = data.loc[A, targets].T.astype(np.float64)
    bX = data.loc[B, targets].T.astype(np.float64)
    doubleX = data.loc[double, targets].T.astype(np.float64)

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
    out["dcor_A_B"] = distance_correlation(aX, bX)
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


def _norman_model_multiple(
    data: pd.DataFrame,
    perturbations: Optional[List[str]] = None,
    method: str = "robust",
    targets: Optional[List[str]] = None,
    delim: str = "|",
    ref: str = "NTC",
    parallel: bool = False,
    processes: int = 4,
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Internal function to handle multiple perturbations analysis.

    This incorporates the original fit_many functionality.

    Args:
        data: DataFrame with perturbations as index and genes as columns.
        perturbations: List of perturbations to analyze or None for auto-detection.
        method: "robust" for TheilSenRegressor or "linear" for LinearRegression.
        targets: List of target genes to analyze.
        delim: Delimiter for identifying/splitting dual perturbations.
        ref: Reference/control value to exclude from auto-detection.
        parallel: Whether to use parallel processing.
        processes: Number of processes for parallel execution.
        verbose: Whether to print progress.

    Returns:
        Tuple of (metrics_DataFrame, predictions_DataFrame).
    """
    # Determine which perturbations to analyze
    if perturbations is None:
        # Run on all dual perturbations (exclude control)
        all_perts = data.index.tolist()
        perturbations = [p for p in all_perts if delim in p and ref not in p]
        if verbose:
            print(f"Auto-detected {len(perturbations)} dual perturbations to analyze")
    elif isinstance(perturbations, list):
        # List of perturbations provided
        pass
    else:
        raise ValueError("perturbations must be None, str, or list")

    # Validate that all perturbations exist in data
    missing_perts = [p for p in perturbations if p not in data.index]
    if missing_perts:
        if verbose:
            print(
                f"Warning: {len(missing_perts)} perturbations not found in data: {missing_perts[:5]}..."
            )
        perturbations = [p for p in perturbations if p in data.index]

    # Handle case where no valid perturbations are found
    if len(perturbations) == 0:
        if verbose:
            print("No valid perturbations found to analyze - returning empty results")
        # Return empty DataFrames with proper structure
        metrics_df = pd.DataFrame(columns=["perturbation", "a", "b"]).set_index(
            "perturbation"
        )
        predictions_df = pd.DataFrame(index=[], columns=data.columns)
        return metrics_df, predictions_df

    if verbose:
        print(f"Analyzing {len(perturbations)} perturbations...")

    # Run analysis
    if parallel and len(perturbations) > 1:
        return _fit_many_parallel(
            data, perturbations, processes, method, targets, delim, ref, verbose
        )
    else:
        return _fit_many_sequential(
            data, perturbations, method, targets, delim, ref, verbose
        )


def _fit_many_sequential(
    data: pd.DataFrame,
    perturbations: List[str],
    method: str,
    targets: Optional[List[str]],
    delim: str,
    ref: str,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Sequential execution of norman_model for multiple perturbations.

    Args:
        data: DataFrame with perturbations as index and genes as columns.
        perturbations: List of perturbations to analyze.
        method: "robust" for TheilSenRegressor or "linear" for LinearRegression.
        targets: List of target genes to analyze.
        delim: Delimiter for splitting dual perturbations.
        ref: Reference/control value.
        verbose: Whether to print progress.

    Returns:
        Tuple of (metrics_DataFrame, predictions_DataFrame).
    """
    results = []
    predictions = []

    for pert in tqdm(perturbations, desc="Fitting models", disable=not verbose):
        out, Z = _norman_model_single(
            data, pert, method, targets, delim, ref, plot=False, verbose=False
        )
        if out is not None:
            results.append(out)
            predictions.append(Z)

    # Handle case where no results were found
    if len(results) == 0:
        # Return empty DataFrames with proper structure
        metrics_df = pd.DataFrame(columns=["perturbation", "a", "b"]).set_index(
            "perturbation"
        )
        predictions_df = pd.DataFrame(index=[], columns=data.columns)
        return metrics_df, predictions_df

    # Convert to DataFrames
    metrics_df = pd.DataFrame(results).set_index("perturbation")
    predictions_df = pd.DataFrame(
        predictions, index=[r["perturbation"] for r in results], columns=data.columns
    )

    return metrics_df, predictions_df


def _fit_many_parallel(
    data: pd.DataFrame,
    perturbations: List[str],
    processes: int,
    method: str,
    targets: Optional[List[str]],
    delim: str,
    ref: str,
    verbose: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Parallel execution of norman_model for multiple perturbations.

    Args:
        data: DataFrame with perturbations as index and genes as columns.
        perturbations: List of perturbations to analyze.
        processes: Number of processes for parallel execution.
        method: "robust" for TheilSenRegressor or "linear" for LinearRegression.
        targets: List of target genes to analyze.
        delim: Delimiter for splitting dual perturbations.
        ref: Reference/control value.
        verbose: Whether to print progress.

    Returns:
        Tuple of (metrics_DataFrame, predictions_DataFrame).
    """
    results = []
    predictions = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        # Submit all jobs
        futures = [
            executor.submit(
                _norman_model_single,
                data,
                pert,
                method,
                targets,
                delim,
                ref,
                False,
                False,
            )
            for pert in perturbations
        ]

        # Process results as they complete
        for future in tqdm(
            concurrent.futures.as_completed(futures),
            total=len(perturbations),
            desc="Fitting models",
            disable=not verbose,
        ):
            out, Z = future.result()
            if out is not None:
                results.append(out)
                predictions.append(Z)

    # Handle case where no results were found
    if len(results) == 0:
        # Return empty DataFrames with proper structure
        metrics_df = pd.DataFrame(columns=["perturbation", "a", "b"]).set_index(
            "perturbation"
        )
        predictions_df = pd.DataFrame(index=[], columns=data.columns)
        return metrics_df, predictions_df

    # Convert to DataFrames
    metrics_df = pd.DataFrame(results).set_index("perturbation")
    predictions_df = pd.DataFrame(
        predictions, index=[r["perturbation"] for r in results], columns=data.columns
    )

    return metrics_df, predictions_df


# Deprecated compatibility functions
def fit_many(
    *args, **kwargs
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[Dict[str, Any], np.ndarray]]:
    """Deprecated: Use norman_model instead.

    This function is kept for backwards compatibility but will be removed
    in a future version. Use norman_model with a list of perturbations instead.

    Returns:
        Same as norman_model function.
    """
    import warnings

    warnings.warn(
        "fit_many is deprecated. Use norman_model with perturbations as a list or None instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    # Convert to new norman_model call
    if len(args) >= 2:
        data, perturbations = args[0], args[1]
        return norman_model(data, perturbations, *args[2:], **kwargs)
    else:
        return norman_model(*args, **kwargs)


def get_model_fit(
    *args, **kwargs
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[Dict[str, Any], np.ndarray]]:
    """Deprecated: Use norman_model instead.

    Returns:
        Same as norman_model function.
    """
    import warnings

    warnings.warn(
        "get_model_fit is deprecated. Use norman_model instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return norman_model(*args, **kwargs)
