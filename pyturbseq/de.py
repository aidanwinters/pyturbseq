from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import pandas as pd
import numpy as np

from statsmodels.stats.multitest import multipletests

from adjustText import adjust_text

def get_degs(adata, design_col, ref_val=None, n_cpus=16, quiet=True):
    """
    This function runs DESeq2 on an AnnData object. It returns a dataframe with the results of the DESeq2 analysis.
    Args:
        adata: AnnData object
        design_col: column in adata.obs that contains the design matrix
        ref_val: 
            reference value for the design matrix. If None, all values in design_col are used as contrasts and listed in alphabetical order. 
            This should be set to syncronize logFC values across multiple DEG runs or for convenience of interpreting. 
            When set, positive logFC values indicate that the reference value is higher than alternative value. More than 2 values in contrast is currently not supported. 
        n_cpus: number of cpus to use for DESeq2
        quiet: whether to print DESeq2 output
    """
    dds = DeseqDataSet(
        counts=pd.DataFrame(

            adata.X.toarray() if type(adata.X) is not np.ndarray else adata.X,
            index=adata.obs.index, columns=adata.var.index
        ),
        metadata=adata.obs,
        design_factors=design_col,
        n_cpus=n_cpus,
        quiet=quiet,
    )
    try:
        dds.deseq2()
    except Exception as e:
        print(f"Exception:" + str(e))
        return pd.DataFrame()

    if ref_val is None:
        contrast = [design_col] + list(adata.obs[design_col].unique()) 
    else:
        print(f"Using {ref_val} as reference value for {design_col}")
        design_vals = adata.obs[design_col].unique()
        alt_val = [x for x in design_vals if x != ref_val] #check to make sure that this is a single calue
        if len(alt_val) > 1:
            raise ValueError(f"More than one alternative value for {design_col} in adata. This is currently not supported.")
        else: 
            alt_val = alt_val[0]
        contrast = [design_col, ref_val, alt_val]
    stat_res = DeseqStats(
        dds, contrast=contrast , n_cpus=n_cpus, quiet=quiet
    )
    stat_res.summary()
    df = stat_res.results_df
    df['padj_bh'] = multipletests(df['pvalue'], method='fdr_bh')[1]
    return df

def get_all_degs(adata, design_col, reference, conditions=None, n_cpus=8):

    def get_degs_subset(condition):
        df = get_degs(
            adata[adata.obs[design_col].isin([condition, reference])],
            design_col,
            ref_val=reference,
            n_cpus=n_cpus
        )
        df['condition'] = condition
        return df
    
    if conditions is None: #get all conditions if not specified
        conditions = adata.obs[design_col].unique()
        #remove reference from conditions
        conditions = [x for x in conditions if x != reference]

    #get all dfs
    dfs = [get_degs_subset(condition) for condition in conditions]

    #concatenate
    df = pd.concat(dfs)
    return df