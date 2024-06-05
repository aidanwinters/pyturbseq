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

from .utils import add_pattern_to_adata


def parse_CR_h5(
    cellranger_h5_path,
    guide_call_csv=None,
    pattern=None,
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

    # rna_file = os.path.join(CR_out , "filtered_feature_bc_matrix.h5")
    if not os.path.exists(cellranger_h5_path):
        raise ValueError(f"Cellranger h5 file provided but file does not exist: {cellranger_h5_path}")

    vp(f"Reading {cellranger_h5_path}")
    adata = sc.read_10x_h5(cellranger_h5_path, gex_only=False)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()

    if pattern is not None:
        add_pattern_to_adata(adata, cellranger_h5_path, pattern, strict=True, quiet=quiet)

    ##Read in guide calls: 
    if guide_call_csv is not None:
        vp("Adding guide calls")
        # guide_call_file = os.path.join(CR_out, "crispr_analysis/protospacer_calls_per_cell.csv")
        #confirm that calls_file exists
        if not os.path.exists(guide_call_csv):
            raise ValueError(f"Guide call CSV provided but file does not exist: {guide_call_csv}")

        vp(f"Reading {guide_call_csv}")
        guide_calls = pd.read_csv(guide_call_csv, index_col=0)

        cbcs = adata.obs.index
        inds = guide_calls.index.intersection(cbcs)
        guide_calls = guide_calls.loc[inds, :]

        vp(f"Found sgRNA information for {len(inds)}/{adata.obs.shape[0]} ({round(len(inds) / adata.obs.shape[0] * 100, 2)}%) of cell barcodes")

        adata.obs = adata.obs.join(guide_calls)

    vp(f"Finished reading adata with {adata.n_obs} cells and {adata.n_vars} genes")

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
    calls_file="protospacer_calls_per_cell.csv",
    library_ref_file=None,
    inplace=True,
    quiet=False,
    ):
    """
    Uses the cellranger calls and merges them with anndata along with some metrics
    """
    #confirm that calls_file exists
    if not os.path.exists(calls_file):
        raise ValueError(f"{calls_file} does not exist")
    calls = pd.read_csv(calls_file, index_col=0)
    inds = calls.index.intersection(adata.obs.index)
    if not quiet: print(f'Found sgRNA information for {len(inds)}/{adata.obs.shape[0]} ({round(len(inds) / adata.obs.shape[0] * 100, 2)}%) of cell barcodes')
    calls = calls.loc[inds, :]

    #merge with anndata obs
    if not inplace: 
        adata = adata.copy()
        
    for col in calls.columns:
        adata.obs.loc[inds, col] = calls[col]

    if not inplace:
        return adata


def parse_CR_flex_metrics(df):
    df = df.copy()
    #formatted like so: 40.00%
    pct_fmt = df['Metric Value'].str.endswith('%', na=False)
    if pct_fmt.any():
        df.loc[pct_fmt, 'Metric Value'] = df.loc[pct_fmt, 'Metric Value'].str.rstrip('%').astype(float) / 100
    
    pct_paren_fmt = df['Metric Value'].str.endswith('%)', na=False)
    if pct_paren_fmt.any():
        #get the number btwn the parentheses and drop the %
        df.loc[pct_paren_fmt, 'Metric Value'] = df.loc[pct_paren_fmt, 'Metric Value'].str.extract(r'\((.*)%\)', expand=False).astype(float) / 100

    #formatted like so: 20,000 (20.00%)
    comma_fmt = df['Metric Value'].str.contains(',', na=False)
    if comma_fmt.any():
        df.loc[comma_fmt, 'Metric Value'] = df.loc[comma_fmt, 'Metric Value'].str.replace(',', '').astype(float)

    #convert everything to float
    df['Metric Value'] = df['Metric Value'].astype(float)
    return df