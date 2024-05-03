from concurrent.futures import ProcessPoolExecutor, as_completed
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


def get_degs(adata, design_col, covariate_cols=None, ref_val=None, alpha=0.05, n_cpus=16, quiet=False):
    """
    Run DESeq2 analysis on single-cell RNA sequencing data.
    
    Parameters:
    adata (AnnData): AnnData object containing the single-cell RNA-seq data.
    design_col (str): Column name in adata.obs that contains the design matrix.
    ref_val (str, optional): Reference value for the design matrix. Defaults to None.
    n_cpus (int, optional): Number of CPUs to use for DESeq2. Defaults to 16.
    quiet (bool, optional): Flag to suppress DESeq2 output. Defaults to True.

    Returns:
    pd.DataFrame: DataFrame containing DESeq2 results.
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

def get_all_degs(adata, design_col, reference, conditions=None, parallel=True, n_cpus=8, max_workers=4, quiet=False, **kwargs):
    """
    Run DESeq2 analysis in parallel for multiple conditions.

    Parameters:
    adata (AnnData): AnnData object containing the single-cell RNA-seq data.
    design_col (str): Column name in adata.obs that contains the design matrix.
    reference (str): Reference condition for the differential expression test.
    conditions (list, optional): List of conditions to test against the reference. Defaults to None.
    n_cpus (int, optional): Number of CPUs to use for each DESeq2 task. Defaults to 8.
    max_workers (int, optional): Maximum number of parallel tasks. Defaults to 4.
    quiet (bool, optional): Flag to suppress DESeq2 output. Defaults to False.

    Returns:
    pd.DataFrame: Concatenated DataFrame containing results for all conditions.
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
        return pd.concat(dfs)
    except KeyboardInterrupt:
        print("Cancellation requested by user. Shutting down...")
        executor.shutdown(wait=False)
        raise 