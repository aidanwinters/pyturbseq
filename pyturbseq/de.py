from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Iterable, Optional, List
from tqdm import tqdm
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from pydeseq2.default_inference import DefaultInference
from statsmodels.stats.multitest import multipletests
import numpy as np
import multiprocessing
import math

from joblib import Parallel, delayed


def get_degs(
    adata: "AnnData",
    design_col: str,
    covariate_cols: Optional[List[str]] = None,
    ref_val: Optional[str] = None,
    alpha: float = 0.05,
    n_cpus: int = 16,
    quiet: bool = False,
) -> pd.DataFrame:
    """Run DESeq2 differential expression for a single comparison.

    Args:
        adata: AnnData object containing counts.
        design_col: Column in ``adata.obs`` describing the experimental design.
        covariate_cols: Optional list of additional covariates.
        ref_val: Reference value for ``design_col``.
        alpha: Adjusted p-value cutoff for significance.
        n_cpus: Number of CPUs used by pydeseq2.
        quiet: Suppress verbose output when ``True``.
    Returns:
        DataFrame of DESeq2 results for the contrast.
    """

    ref_level = [design_col, ref_val] if ref_val is not None else None

    inference = DefaultInference(n_cpus=n_cpus)

    design = design_col if covariate_cols is None else [design_col] + covariate_cols

    dds = DeseqDataSet(
        counts=pd.DataFrame(
            adata.X.toarray() if type(adata.X) is not np.ndarray else adata.X,
            index=adata.obs.index, columns=adata.var.index
        ),
        metadata=adata.obs,
        design_factors=design,
        inference=inference,
        min_replicates=math.inf, 
        min_mu=1e-6,
        ref_level=ref_level,
        refit_cooks=True,
        quiet=quiet,  # Passing the quiet argument
    )

    dds.deseq2()  # Passing the quiet argument

    design_col_categories = adata.obs[design_col].unique()
    #drop ref_val
    if ref_val is not None:
        design_col_categories = design_col_categories[design_col_categories != ref_val]
    contrast = [design_col] + list(design_col_categories) + [ref_val] 
    stat_res = DeseqStats(dds, contrast=contrast, quiet=quiet, inference=inference)
    stat_res.summary()

    df = stat_res.results_df
    # df['padj_bh'] = multipletests(df['pvalue'], method='fdr_bh')[1]
    df['significant'] = df['padj'] < 0.05

    return df

def get_all_degs(
    adata: "AnnData",
    design_col: str,
    reference: str,
    conditions: Optional[Iterable[str]] = None,
    parallel: bool = True,
    n_cpus: int = 8,
    max_workers: int = 4,
    quiet: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """Run DESeq2 for multiple conditions in parallel.

    Args:
        adata: AnnData object with count data.
        design_col: Column used as the design factor.
        reference: Reference condition within ``design_col``.
        conditions: Specific conditions to test. If ``None`` all non-reference
            values are tested.
        parallel: Whether to run in parallel using joblib.
        n_cpus: Number of CPUs per DESeq2 task.
        max_workers: Maximum number of concurrent tasks.
        quiet: Suppress verbose output.
        **kwargs: Additional keyword arguments passed to :func:`get_degs`.
    Returns:
        Concatenated DataFrame of DESeq2 results for each condition.
    """

    vp = print if not quiet else lambda *a, **k: None

    if conditions is None:
        conditions = list(set(adata.obs[design_col]) - {reference})


    def get_deg_worker(condition):
        try:
            df = get_degs(
                adata[adata.obs[design_col].isin([condition, reference])],
                design_col,
                ref_val=reference,
                n_cpus=n_cpus,
                quiet=quiet,
                **kwargs
                )
            df['condition'] = condition
            return df
        except Exception as e:
            print(f"Exception in DESeq2 execution: {e}")
            return pd.DataFrame()
    try:
        if parallel:
            available_cpus = multiprocessing.cpu_count()
            n_cpus = min(n_cpus, available_cpus // max_workers)
            vp(f"Running DESeq2 in parallel with {max_workers} workers and {n_cpus} per worker...")
            dfs = Parallel(n_jobs=max_workers)(delayed(get_deg_worker)(condition) for condition in tqdm(conditions, disable=quiet))
        else:
            vp(f"Running DESeq2 synchronously...")
            dfs = [get_deg_worker(condition) for condition in conditions]
        # return pd.concat(dfs)
        res = pd.concat(dfs).reset_index().rename(columns={'index': 'gene'})
        return res
        
    except KeyboardInterrupt:
        print("Cancellation requested by user. Shutting down...")
        executor.shutdown(wait=False)
        raise 
