from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
import pandas as pd
import numpy as np

def get_degs(adata, design_col, cov_val, n_workers=1):
        print(f"Running DESeq2...")
        dds = DeseqDataSet(
            counts=pd.DataFrame(
                adata.X.toarray(), index=adata.obs.index, columns=adata.var.index
            ),
            metadata=adata.obs,
            design_factors=design_col,
            n_cpus=1,
        )
        try:
            dds.deseq2()
        except Exception as e:
            print(f"Exception:" + str(e))
            return pd.DataFrame()
        cov_list = list(adata.obs[design_col].unique())
        stat_res = DeseqStats(
            dds, contrast=[design_col, cov_val], n_cpus=n_workers
        )
        stat_res.summary()
        res = stat_res.results_df
        return res