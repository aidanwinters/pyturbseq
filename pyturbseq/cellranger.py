##########################################################################
# 
# Functions for reading and manipulating 10x Single cell data from cellranger outs
#
##########################################################################
import scanpy as sc
import re
import os
import pandas as pd
import math

def parse_h5(
    CR_out,
    pattern=r"opl(?P<opl>\d+).*lane(?P<lane>\d+)",
    add_guide_calls=True,
    quiet=False,
    ):
    """
    Read in 10x data from a CRISPR screen.
    Args: 
        CR_out: path to the output directory of the CRISPR screen
        pattern: regex pattern to extract metadata from the file name
        add_guide_calls: whether to add guide calls to the adata.obs
    """
    vp = print if not quiet else lambda *a, **k: None

    rna_file = os.path.join(CR_out , "filtered_feature_bc_matrix.h5")

    vp(f"Reading {rna_file}")
    adata = sc.read_10x_h5(rna_file, gex_only=False)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    
    ##add each capture group to adata.obs
    match = re.search(pattern, rna_file)
    if match is None:
        raise ValueError(f"Could not extract metadata from {adata_path}")
    else:
        for key, value in match.groupdict().items():
            vp(f"Adding {key} = {value}")
            adata.obs[key] = value

    ##Read in guide calls: 
    if add_guide_calls:
        vp("Adding guide calls")
        guide_call_file = os.path.join(CR_out, "crispr_analysis/protospacer_calls_per_cell.csv")
        vp(f"Reading {guide_call_file}")
        guide_calls = pd.read_csv(guide_call_file, index_col=0)

        cbcs = adata.obs.index
        inds = guide_calls.index.intersection(cbcs)
        guide_calls = guide_calls.loc[inds, :]

        adata.obs = adata.obs.join(guide_calls)

    return(adata)

def parse_umi(x):
    """
    Parse the output of the CRISPR analysis from cellranger pipeline to get the total number of UMIs and the max UMI
    """
    if isinstance(x, float):
        return {
            'CR_total_umi': x,
            'CR_max_umi': x,
            'CR_ratio_2nd_1st': math.nan
        }
    split = [int(i) for i in x.split('|')]
    split.sort(reverse=True)
    m = {
        'CR_total_umi': sum(split),
        'CR_max_umi': split[0],
        'CR_ratio_2nd_1st': math.nan if len(split) == 1 else split[1]/split[0]
    }
    return m

def add_CR_umi_metrics(adata):
    """
    Add the CRISPR UMI metrics to the adata.obs
    """
    df = pd.DataFrame([parse_umi(x) for x in adata.obs['num_umis']], index=adata.obs.index)
    #merge by index but keep index as index
    adata.obs = adata.obs.merge(df, left_index=True, right_index=True)
    return adata


def add_CR_sgRNA(
    adata,
    sgRNA_analysis_out,
    calls_file="protospacer_calls_per_cell.csv",
    library_ref_file=None,
    quiet=False,
    ):
    """
    Uses the cellranger calls and merges them with anndata along with some metrics
    """
    calls = pd.read_csv(sgRNA_analysis_out + calls_file, index_col=0)
    inds = calls.index.intersection(adata.obs.index)
    if not quiet: print(f'Found sgRNA information for {len(inds)}/{adata.obs.shape[0]} ({round(len(inds) / adata.obs.shape[0] * 100, 2)}%) of cell barcodes')
    calls = calls.loc[inds, :]

    #merge with anndata obs
    for col in calls.columns:
        adata.obs.loc[inds, col] = calls[col]
    return adata