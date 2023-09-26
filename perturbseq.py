import scanpy as sc
import re
import pandas as pd
import os
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def parse_h5(
    CR_out,
    pattern=r"opl(?P<opl>\d+).*lane(?P<lane>\d+)",
    add_guide_calls=True,
    ):
    """
    Read in 10x data from a CRISPR screen.
    Args: 
        CR_out: path to the output directory of the CRISPR screen
        pattern: regex pattern to extract metadata from the file name
        add_guide_calls: whether to add guide calls to the adata.obs
    """

    rna_file = os.path.join(CR_out , "filtered_feature_bc_matrix.h5")
    

    print(f"Reading {rna_file}")
    adata = sc.read_10x_h5(rna_file, gex_only=False)
    adata.var_names_make_unique()
    adata.obs_names_make_unique()
    
    ##add each capture group to adata.obs
    match = re.search(pattern, rna_file)
    for key, value in match.groupdict().items():
        print(f"Adding {key} = {value}")
        adata.obs[key] = value

    ##Read in guide calls: 
    if add_guide_calls:
        print("Adding guide calls")
        guide_call_file = os.path.join(CR_out, "crispr_analysis/protospacer_calls_per_cell.csv")
        print(f"Reading {guide_call_file}")
        guide_calls = pd.read_csv(guide_call_file, index_col=0)

        cbcs = adata.obs.index
        inds = guide_calls.index.intersection(cbcs)
        guide_calls = guide_calls.loc[inds, :]

        adata.obs = adata.obs.join(guide_calls)

    return(adata)



## add some of our own metrics using outputs from CR
#split 'num_umis' column on '|' and sum all the values
def parse_umi(x):
    # if x is float:
    if isinstance(x, float):
        return {
            'CR_total_umi': x,
            'CR_max_umi': x,
            'CR_ratio_2nd_1st': math.nan
        }

    split = [int(i) for i in x.split('|')]
    #sort split
    split.sort(reverse=True)
    m = {
        'CR_total_umi': sum(split),
        'CR_max_umi': split[0],
        'CR_ratio_2nd_1st': math.nan if len(split) == 1 else split[1]/split[0]
    }
    return m


########################################################################################################################
########################################################################################################################
############# GUIDE CALLING FUNCTIONS ##################################################################################
########################################################################################################################
########################################################################################################################

def add_CR_umi_metrics(adata):
    df = pd.DataFrame([parse_umi(x) for x in adata.obs['num_umis']], index=adata.obs.index)
    #merge by index but keep index as index
    adata.obs = adata.obs.merge(df, left_index=True, right_index=True)
    return adata

##random pivot proportion function
#convert above into function
def get_pct_count(df, col1, col2):
    vc = df[[col1, col2]].value_counts().reset_index()
    vc = vc.pivot(index=col1, columns=col2, values='count')
    vc = vc.div(vc.sum(axis=1), axis=0)*100
    return vc


### function to take threshold mapping file and binarize guide matrix
def binarize_guides(adata, threshold_df=None, threshold_file=None, threshold_col='UMI_threshold', inplace=False):

    if threshold_df is None and threshold_file is None:
        print('Must provide either threshold_df or threshold_file')
        return None
    elif threshold_df is None:
        threshold_df = pd.read_csv(threshold_file, index_col=0)
        threshold_df.columns = [threshold_col] #required because 

    overlap = adata.var.index[adata.var.index.isin(threshold_df.index)]
    # if no overlap then exit:
    print(f"Found {len(overlap)} overlapping features")
    if len(overlap) == 0:
        print('No overlap between adata and threshold_df')
        return None
    

    #synchornize so only feautres in threshold_df are kept
    adata = adata[:, overlap]
    #set order of threshold df as well 
    threshold_df = threshold_df.loc[overlap, :]
    thresholds = threshold_df[threshold_col]

    binX = np.greater_equal(adata.X.toarray(), thresholds.values)
    # if inplace: 
    #     print('Updating X in place')
    #     adata.X = csr_matrix(binX.astype(int))
    # else:
    print('Creating new X matrix')
    adata = adata.copy()
    adata.X = csr_matrix(binX.astype(int))
    return adata


def check_calls(guide_call_matrix, expected_max_proportion=0.2):
    """
    Function to check if a given guide is enriched above expected
    Args: 
        guide_call_matrix (ad.AnnData): AnnData object with guide calls in adata.obs['guide']
        expected_max_proportion (float): expected proportion of cells that should have a given guide
    """

    #for now only check is if a given guide is enriched above expected
    prop_calls = guide_call_matrix.X.toarray().sum(axis=0)/guide_call_matrix.shape[0]
    over = np.where(prop_calls > expected_max_proportion)
    # prop_calls.index[over]
    flagged_guides = guide_call_matrix.var.index[over].values
    print(f"Found {len(flagged_guides)} guides that are assigned above expected")
    if len(flagged_guides) > 0:
        print(f'These guide(s) are enriched above expected:')
        for i in flagged_guides:
            print("\t" + i)
    return flagged_guides


def plot_guide_cutoff(adata, feat, thresh, ax=None, x_log=True, y_log=True):
    vals = adata[:,adata.var['gene_ids'] ==  feat].X.toarray().flatten()

    if ax is None:
        fig, ax = plt.subplots()

    x,t = (np.log10(vals+1), np.log10(thresh+1)) if x_log else (vals, thresh)
    sns.histplot(x, bins=30,  ax=ax)
    ax.axvline(t, linestyle='--', color='red')
    ax.set_yscale('log', base=10)
    ax.set_title(feat + "\nthreshold: " + str(thresh))
    if x_log: 
        ax.set_xlabel('log10(UMI+1)')
    else:
        ax.set_xlabel('UMI')
    ylab = 'Number of cells'
    if y_log:
        ylab = ylab + ' (log10)'
    ax.set_ylabel(ylab)

def plot_many_guide_cutoffs(adata, features, thresholds, ncol=4, **kwargs):

    nrow = int(np.ceil(len(features)/ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(ncol*5, nrow*5))
    ax = ax.flatten()
    # rand_feats = np.random.choice(thresholds.index, 10)

    for i, (f, t) in enumerate(zip(features, thresholds)):
        plot_guide_cutoff(adata, f, t, ax=ax[i], **kwargs)

    fig.tight_layout()
