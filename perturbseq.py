import scanpy as sc
import re
import pandas as pd
import os
import math
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from adpbulk import ADPBulk


import anndata as ad

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


########################################################################################################################

def filter_adata(adata, obs_filters=None, var_filters=None):
    
    if obs_filters is not None:
        for f in obs_filters:
            adata = adata[adata.obs.query(f).index, :]
        
    if var_filters is not None:
        for f in var_filters:
            adata = adata[:, adata.var.query(f).index]

    return adata

########################################################################################################################
########################################################################################################################
############# PSEUDO BULK and ZSCORE FUNCTIONS #########################################################################
########################################################################################################################
########################################################################################################################

def zscore(adata, ref_col='perturbation',ref_val='NTC|NTC', scale_factor = None,):
    
    ##check if csr matrix
    if isinstance(adata.X, np.ndarray):
        arr = adata.X
    else:
        arr = adata.X.toarray()

    total_counts = arr.sum(axis=1)
    if scale_factor is None:

        median_count = np.median(total_counts)
    else: 
        median_count = scale_factor

    scaling_factors = median_count / total_counts
    scaling_factors = scaling_factors[:, np.newaxis] #reshape to be a column vector
    arr = arr * scaling_factors
    ref_inds = np.where(adata.obs[ref_col] == ref_val)[0]

    #exit if not ref_inds
    if len(ref_inds) == 0:
        
        raise ValueError(f"ref_col '{ref_col}' and ref_val '{ref_val}' yielded no results")

    mean = arr[ref_inds,].mean(axis=0)
    stdev = arr[ref_inds,].std(axis=0)
    # stdev = np.std(adata[ref_inds,:].X, axis=0)
    return np.array(np.divide((arr - mean), stdev))

def zscore_cov(
        adata, 
        covariates=None,
        **kwargs):
    
    #get median

    #get mapping of index val to row val
    # Create a dictionary mapping index to row number
    index_to_row = {index: row for row, index in enumerate(adata.obs.index)}
    normalized_array = np.empty_like(adata.X.toarray())
    print(normalized_array.shape)

    #first split the adata into groups based on covariates
    if covariates is not None:
        #iterate on each groupby for anndata and apply pseudobulk 
        g = adata.obs.groupby(covariates)
        meta = pd.DataFrame(g.groups.keys(), columns=covariates)
        print(f"Splitting into {len(meta)} groups based on covariates: {covariates}")

        mapping = g.groups.items()
        for key, inds in mapping:
            print(key)
            rows = [index_to_row[index] for index in inds]
            normalized_array[rows,] = zscore(adata[inds,], **kwargs)
        # arr = np.vstack([zscore(adata[inds,], **kwargs) for key, inds in mapping])

        ##append all the inds together: 
        # inds = [inds for key, inds in mapping]

        return normalized_array

        # out = {key: }


        # for key, inds in mapping:
        #     adata[inds,].X = out[key]

        return adata

    else:
        #if covariates are none then we just apply pseudobulk to the whole matrix (ie single sample)
        return zscore(adata, **kwargs)


def pseudobulk(adata, groupby, **kwargs):
    """
    Function to apply pseudobulk to anndata object
    Args:
        adata (ad.AnnData): AnnData object with guide calls in adata.obs['guide']
        groupby (str): column in adata.obs to group by
        **kwargs: arguments to pass to pseudobulk function
    """
    adpb = ADPBulk(adata, groupby=groupby, **kwargs)
    pseudobulk_matrix = adpb.fit_transform()
    sample_meta = adpb.get_meta().set_index('SampleName')
    adata = sc.AnnData(pseudobulk_matrix, obs=sample_meta, var=adata.var)

    return adata


########################################################################################################################
########################################################################################################################
############# KNOCKDOWN FUNCTIONS ######################################################################################
########################################################################################################################
########################################################################################################################

def plot_kd(adata, gene):
    gene_vals = adata[:,gene].X.toarray().flatten()
    ##plot AR for AR KD vs NTC|NTC
    gene_inds = (adata.obs['perturbation'].str.contains(gene + '\|')) | (adata.obs['perturbation'].str.contains('\|' + gene))
    NTC_inds = adata.obs['perturbation'] == 'NTC|NTC'
    print(f"Number of obs in NTC|NTC: {np.sum(NTC_inds)}")
    print(f"Number of obs in {gene} KD: {np.sum(gene_inds)}")


    plt.hist(gene_vals[NTC_inds], label='NTC', alpha=0.5, bins=30)
    plt.hist(gene_vals[gene_inds], label=gene + ' KD', alpha=0.5, bins=30)
    #add mean line for each group
    plt.axvline(gene_vals[NTC_inds].mean(), color='blue')
    plt.axvline(gene_vals[gene_inds].mean(), color='orange')
    plt.legend()
    plt.show()


def percent_knocked_down_per_cell(adata, perturbation_column, reference_label):
    """
    Compute the "percent knocked down" for each cell against a reference.
    
    Parameters:
    - adata: anndata.AnnData object containing expression data
    - perturbation_column: column in adata.obs indicating the perturbation/knockdown
    - reference_label: label of the reference population in perturbation_column
    
    Returns:
    - An AnnData object with an additional column in obs containing the "percent knocked down" for each cell.
    """
    
    # Get mean expression values for reference population
    reference_means = adata[adata.obs[perturbation_column] == reference_label, :].X.toarray().mean(axis=0)
    
    # Placeholder for percent knocked down values
    percent_kd_values = []
    reference_mean_values = []
    
    # Loop through cells
    for cell, perturb in zip(adata.X.toarray(), adata.obs[perturbation_column]):
        if perturb == reference_label or perturb not in adata.var_names:
            percent_kd_values.append(None)
            reference_mean_values.append(None)
            continue

        # print('here')
        gene_idx = adata.var_names.get_loc(perturb)
        
        percent_value = (1 - ((cell[gene_idx]+1) / (reference_means[gene_idx] + 1))) * 100
        percent_kd_values.append(percent_value)
        reference_mean_values.append(reference_means[gene_idx])
    
    # Add to adata
    adata.obs['percent_kd'] = percent_kd_values
    adata.obs['reference_mean'] = reference_mean_values
    
    return adata



#####################################################################################################################
#####################################################################################################################
##################################################################################################################### 
############# LIBRARY QC ############################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
from scipy.spatial import distance
from tqdm import tqdm

def hamming_dist(a, b):
    return distance.hamming(list(a), list(b))

# get all hamming distances for pairs of guides
def hamming_dist_matrix(ref):
    """
    This method take a list of strings, assumed to all be the same length, and pairwise haming distance
    """
    dists = np.zeros((len(ref), len(ref)))

    #get triangle upper indices
    triu_indices = np.triu_indices(len(ref), k=0)

    # iterate over trianlge indices
    for i, j in tqdm(zip(*triu_indices)):
        d = hamming_dist(ref[i], ref[j])
        dists[i, j] = d
        dists[j, i] = d

    return dists



