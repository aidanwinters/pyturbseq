from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import pandas as pd
import numpy as np

from statsmodels.stats.multitest import multipletests

from adjustText import adjust_text

def get_degs(adata, design_col, n_cpus=16, quiet=True):
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
    contrast = [design_col] + list(adata.obs[design_col].unique()) 
    stat_res = DeseqStats(
        dds, contrast=contrast , n_cpus=n_cpus, quiet=quiet
    )
    stat_res.summary()

    return stat_res.results_df

def get_all_degs(adata, design_col, reference, conditions=None, n_cpus=8):

    def get_degs_subset(condition):
        df = get_degs(adata[adata.obs[design_col].isin([condition, reference])], design_col, n_cpus=n_cpus)
        df['condition'] = condition
        df['padj_bh'] = multipletests(df['pvalue'], method='fdr_bh')[1]
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