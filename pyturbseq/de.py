from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from statsmodels.stats.multitest import multipletests
import numpy as np
import multiprocessing

from joblib import Parallel, delayed


def get_degs(adata, design_col, ref_val=None, n_cpus=16, quiet=False):
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
    dds = DeseqDataSet(
        counts=pd.DataFrame(
            adata.X.toarray() if type(adata.X) is not np.ndarray else adata.X,
            index=adata.obs.index, columns=adata.var.index
        ),
        metadata=adata.obs,
        design_factors=design_col,
        n_cpus=n_cpus,
        quiet=quiet,  # Passing the quiet argument
    )

    dds.deseq2()  # Passing the quiet argument
    # Setting up contrast for DESeq2
    if ref_val is None:
        contrast = [design_col] + list(adata.obs[design_col].unique())
    else:
        design_vals = adata.obs[design_col].unique()
        alt_val = [x for x in design_vals if x != ref_val]
        if len(alt_val) > 1:
            raise ValueError(f"More than one alternative value for {design_col} in adata. This is currently not supported.")
        contrast = [design_col, alt_val[0], ref_val]

    # Running the statistical analysis
    stat_res = DeseqStats(dds, contrast=contrast, n_cpus=n_cpus, quiet=quiet)
    stat_res.summary()

    df = stat_res.results_df
    df['padj_bh'] = multipletests(df['pvalue'], method='fdr_bh')[1]

    return df

def get_all_degs(adata, design_col, reference, conditions=None, parallel=True, n_cpus=8, max_workers=4, quiet=False):
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
            return get_degs(
                adata[adata.obs[design_col].isin([condition, reference])],
                design_col,
                ref_val=reference,
                n_cpus=n_cpus,
                quiet=quiet
            )
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