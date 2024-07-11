##########################################################################
# 
# Functions for manipulation and filtering of anndata objects and other data structures
#
##########################################################################

##IMPORTS
# Data manipulation and computation
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, issparse
from scipy.spatial.distance import pdist, cdist
from scipy.cluster.hierarchy import linkage, leaves_list

# Visualization
import seaborn as sns
import matplotlib.pyplot as plt

# Single-cell analysis
import scanpy as sc
from anndata import AnnData
from adpbulk import ADPBulk

# Parallel processing and warnings
from joblib import Parallel, delayed
import warnings

# Progress bar
from tqdm import tqdm

# Metrics
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve

# Regular expressions
import re

import warnings
from typing import Optional, Dict, Union, List

########################################################################################################################
########### Basic STRING parsing utils #################################################################################
########################################################################################################################
def split_sort_trim(label, delim='|', delim2='_'):
    #if not string then print
    if type(label) != str:
        return None
    vals = np.sort([x.split(delim2)[0] for x in label.split(delim)])
    return delim.join(vals)

def split_compare(label, delim='|', delim2='_', expected_num=2):
    if type(label) != str:
        return None
    vals = [x.split(delim2)[0] for x in label.split(delim)]
    # if vals are all the same return val[0]
    if len(vals) != expected_num:
        return None
    elif len(set(vals)) == 1:
        return vals[0]
    else: 
        None    

def split_sort_paste(l, split_delim='_', paste_delim='|'):
    #if type is not series make it so
    if type(l) != pd.Series:
        l = pd.Series(l)

    l = l.str.split(split_delim).str[0]
    return paste_delim.join(np.sort(l.values))

def add_pattern_to_adata(adata, search_string, pattern, strict=True, quiet=True):
    vp = print if not quiet else lambda *a, **k: None
    ##add each capture group to adata.obs
    match = re.search(pattern, search_string)
    if (match is None) and strict:
        raise ValueError(f"In strict mode and could not extract metadata from {search_string}")
    else:
        for key, value in match.groupdict().items():
            vp(f"Adding {key} = {value}")
            adata.obs[key] = value



########################################################################################################################
########################################################################################################################
############# ADATA FILTERING FUNCTIONS ################################################################################
########################################################################################################################

def filter_adata(adata, obs_filters=None, var_filters=None, copy=True):
    """
    Filter an anndata object based on obs and var filters.
    """

    if obs_filters is not None:
        obs_filter = np.all([adata.obs.eval(f) for f in obs_filters], axis=0)
        adata = adata[obs_filter, :]
        
    if var_filters is not None:
        var_filter = np.all([adata.var.eval(f) for f in var_filters], axis=0)
        adata = adata[:, var_filter]

    return adata

def filter_to_feature_type(
    adata,
    feature_type='Gene Expression'
    ):
    """
    Updates an anndata object to only include the GEX feature type in its .X slot. 
    Optionally adds the removed features to metadata
    """
    return adata[:, adata.var['feature_types'] == feature_type].copy()

def split_by_feature_type(
    adata,
    copy=True,
    ):
    """
    Updates an anndata object to only include the GEX feature type in its .X slot. 
    Optionally adds the removed features to metadata
    """
    out = {}
    for ftype in adata.var['feature_types'].unique():
        out[ftype] = adata[:, adata.var['feature_types'] == ftype]
        if copy:
            out[ftype] = out[ftype].copy()
    return out




########################################################################################################################
#read in the feature call column, split all of them by delimiter 
def generate_perturbation_matrix(
    adata,
    perturbation_col = 'feature_call',
    delim = '|',
    reference_value = 'NTC',
    feature_list = None,
    keep_ref = False,
    set_ref_1 = False,
    return_boolean=True,
    # sparse = True,
    verbose = True,
    ):

    #if there is no feature list, split all the features in the column and build one
    if feature_list is None:
        #get all the features but not nan
        labels = adata.obs[perturbation_col].dropna()
        feature_list = labels.str.split(delim).explode().unique()
        if verbose:
            print(f"Found {len(feature_list)} unique features.")

    if reference_value not in feature_list:
        raise ValueError(f"Trying to pass 'reference_value' of '{reference_value}' to 'generate_perturbation_matrix' but not found in feature list")

    #create a matrix of zeros with the shape of the number of cells and number of features
    perturbation_matrix = np.zeros((adata.shape[0], len(feature_list)))
    #build dicitonary mapping feature to columns index
    feature_dict = dict(zip(feature_list, range(len(feature_list))))

    #for each cell, split the feature call column by the delimiter and add 1 to the index of the feature in the feature list
    counter = 0
    for i, cell in enumerate(adata.obs[perturbation_col].str.split(delim)):
        try:
            perturbation_matrix[i, [feature_dict[feature] for feature in cell]] = 1
        except:
            counter += 1

    if not keep_ref:
        ##remove featutre ref from matrix
        perturbation_matrix = np.delete(perturbation_matrix, feature_dict[reference_value], axis=1)
        feature_list = feature_list[feature_list != reference_value]
    elif set_ref_1 and keep_ref:
        perturbation_matrix[:, feature_dict[reference_value]] = 1

    if return_boolean:
        perturbation_matrix = perturbation_matrix.astype(bool) 

    # if sparse:
    #     return csr_matrix(perturbation_matrix)
    return pd.DataFrame(
        perturbation_matrix,
        index=adata.obs.index,
        columns=feature_list)

def get_perturbation_matrix(
        adata, 
        perturbation_col = 'feature_call',
        inplace = True,    
        **kwargs      
        ):
    """
    Add a perturbation matrix to an anndata object. 
    Args:
        adata: anndata object
        perturbation_col: column in adata.obs that contains the perturbation information
        feature_list: list of features to include in the perturbation matrix. If None, all features in the perturbation column will be included.
        inplace: whether to add the perturbation matrix to the adata object or return it
    Returns:
        adata object with perturbation matrix in adata.layers['perturbations']
    """
    pm = generate_perturbation_matrix(
            adata,
            perturbation_col = perturbation_col,
            **kwargs
            )

    if inplace:
        adata.obsm['perturbation'] = pm.loc[adata.obs.index, :].copy()
        # cols = pm.columns.tolist()
        # adata.uns['perturbation_var'] = dict(zip(cols, range(len(cols))))
    else:
        return pm.loc[adata.obs.index, :]


def cluster_df(df, cluster_rows=True, cluster_cols=True, method='average'):
    """
    Reorders a DataFrame based on hierarchical clustering.
    
    Parameters:
    - df: The input DataFrame
    - cluster_rows: Whether to cluster and reorder the rows
    - cluster_cols: Whether to cluster and reorder the columns
    - method: Linkage algorithm to use for clustering (e.g., 'average', 'single', 'complete')
    
    Returns:
    - DataFrame reordered based on hierarchical clustering.
    """
    
    if cluster_cols:
        # Compute pairwise distances for columns and cluster
        col_linkage = linkage(pdist(df.T), method=method)
        # Extract column order from dendrogram
        col_order = leaves_list(col_linkage)
        df = df[df.columns[col_order]]
    
    if cluster_rows:
        # Compute pairwise distances for rows and cluster
        row_linkage = linkage(pdist(df), method=method)
        # Extract row order from dendrogram
        row_order = leaves_list(row_linkage)
        df = df.iloc[row_order]
    return df

def cells_not_normalized(adata):
    sums = np.array(adata.X.sum(axis=1)).flatten()
    dev = np.std(sums)
    return True if dev > 1 else False

def _get_target_change_single_perturbation(adata, gene, perturbed_bool, ref_bool):
    """
    Compute the "percent change" for each cell against a reference.
    
    Parameters:
    - adata: anndata.AnnData object containing expression data. assumed to be transformed as desired
    - perturbed_bool: boolean array indicating which cells are perturbed
    - gene: gene name
    - ref_mean: reference mean expression value
    
    Returns:
    - A list of "percent knocked down" for each cell.
    """

    warnings.warn("This function is deprecated. Please use _get_target_change_single_perturbation_indexed instead.", DeprecationWarning)

    if gene not in adata.var_names:
        out = {}
        out['target_gene'] = np.nan
        out['target_gene_expression'] = np.nan
        out['target_reference_mean'] = np.nan
        out['target_reference_std'] = np.nan
        out['target_pct_change'] = np.nan
        out['target_zscore'] = np.nan
        out['target_log2fc'] = np.nan
        return out

    target_gene_expression = adata[:, gene].X.flatten()
    reference_target_mean = float(np.mean(target_gene_expression[ref_bool]))
    reference_target_std = float(np.std(target_gene_expression[ref_bool]))

    out = {}
    out['target_gene'] = gene
    out['target_gene_expression'] = target_gene_expression[perturbed_bool]
    out['target_reference_mean'] = reference_target_mean
    out['target_reference_std'] = reference_target_std
    out['target_pct_change'] = ((target_gene_expression[perturbed_bool] - reference_target_mean) / reference_target_mean) * 100
    out['target_zscore'] = (target_gene_expression[perturbed_bool] - reference_target_mean) / reference_target_std
    out['target_log2fc'] = np.log2((target_gene_expression[perturbed_bool] + 1) / (reference_target_mean+1))

    return out
def _get_target_change_single_perturbation_indexed(adata, gene, perturbed_inds, ref_inds):
    """
    Compute the "percent change" for each cell against a reference.
    
    Parameters:
    - adata: anndata.AnnData object containing expression data. assumed to be transformed as desired
    - perturbed_bool: boolean array indicating which cells are perturbed
    - gene: gene name
    - ref_mean: reference mean expression value
    
    Returns:
    - A list of "percent knocked down" for each cell.
    """

    if gene not in adata.var_names:
        out = {}
        out['target_gene'] = np.nan
        out['target_gene_expression'] = np.nan
        out['target_reference_mean'] = np.nan
        out['target_reference_std'] = np.nan
        out['target_pct_change'] = np.nan
        out['target_zscore'] = np.nan
        out['target_log2fc'] = np.nan
        return out

    # target_gene_expression = adata[:, gene].X.flatten()
    ref_gex = adata[ref_inds, gene].X.flatten()
    reference_target_mean = float(np.mean(ref_gex))
    reference_target_std = float(np.std(ref_gex))

    ptrb_gex = adata[perturbed_inds, gene].X.flatten()
    out = {}
    out['target_gene'] = gene
    out['target_gene_expression'] = ptrb_gex
    out['target_reference_mean'] = reference_target_mean
    out['target_reference_std'] = reference_target_std
    out['target_pct_change'] = ((ptrb_gex - reference_target_mean) / reference_target_mean) * 100
    out['target_zscore'] = (ptrb_gex - reference_target_mean) / reference_target_std
    out['target_log2fc'] = np.log2((ptrb_gex + 1) / (reference_target_mean+1))

    return out


def unroll_target_change(adata, value):
    if ('perturbation' not in adata.obsm) | (value not in adata.obsm):
        raise ValueError(f"Could not find 'perturbation' or '{value}' in adata.obsm. Please run calculate_target_change first.")
    return adata.obsm[value].values[adata.obsm['perturbation'].values]
    
def calculate_target_change(
    adata: AnnData,
    perturbation_column,
    reference_value: Optional[str] = None,
    perturbation_gene_map: Optional[Dict[str, str]] = None,
    groupby: Optional[Union[str, List[str]]] = None,
    check_norm: bool = True,
    quiet: bool = False,
    inplace: bool = True,
    collapse_into_obs: bool = True,
    **kwargs,
) -> Optional[AnnData]:
    """
    Calculate the "percent change" for each cell against a reference.

    Parameters:
    -----------
    adata : AnnData
        Annotated data matrix.
    perturbation_column : str
        Column in `adata.obs` indicating the perturbations.
    reference_value : str, optional
        Value in `perturbation_column` to use as the reference.
    perturbation_gene_map : dict, optional
        Mapping from perturbation identifiers to gene names.
    groupby : str or list of str, optional
        Columns in `adata.obs` to group by for calculations.
    check_norm : bool, optional
        Whether to check and normalize data to counts per cell if necessary.
    quiet : bool, optional
        Suppress output messages.
    inplace : bool, optional
        Whether to modify the `adata` object in place.
    collapse_into_obs : bool, optional
        Whether to collapse the computed metrics into `adata.obs`.

    Returns:
    --------
    adata : AnnData, optional
        Updated AnnData object with calculated metrics if `inplace=False`.

    Example:
    --------
    >>> calculate_target_change(adata, 'perturbation', reference_value='NegCtrl', quiet=False, inplace=True)

    Notes:
    ------
    This function calculates the percent change in gene expression for each cell 
    against a reference value (e.g., control cells). The results can be stored 
    in the `adata` object either in place or returned as a new object.
    """

    duplicated_genes = adata.var.index.duplicated()
    if sum(duplicated_genes) > 0:
        raise ValueError(f"Duplicated gene names found in adata.var.index. Please remove duplicated gene names before running this function. \nDuplicated gene names found: {list(adata.var.index[duplicated_genes])}")

    duplicated_obs = adata.obs.index.duplicated()
    if sum(duplicated_obs) > 0:
        raise ValueError(f"Observation names are not unique. To make them unique, call `.obs_names_make_unique` before running target change.")

    final_adata = adata if inplace else adata.copy()

    if not quiet: print(f"Computing percent change for '{perturbation_column}' across {final_adata.shape[0]} cells...")

    if check_norm:
        if not quiet: print('\tChecking if data is normalized to counts per cell...')
        if cells_not_normalized(final_adata):
            if not quiet: print('\t\tData is not normalized to counts per cell. Normalizing...')
            final_adata.layers['temp_counts'] = final_adata.X.copy()
            sc.pp.normalize_total(final_adata)
        else:
            if not quiet: print('\t\tData already normalized to counts per cell.')

    if not quiet: print('\tComputing percent change for each cell...')

    if issparse(final_adata.X):
        final_adata.X = final_adata.X.toarray()

    if not quiet: print(f"\tGenerating perturbation matrix from '{perturbation_column}' column...")
    get_perturbation_matrix(final_adata, perturbation_column, reference_value=reference_value, inplace=True, verbose=not quiet, **kwargs)
    pm = final_adata.obsm['perturbation']

    if not quiet: print(f"\tFound {pm.shape[1]} unique perturbations in {perturbation_column} column.")

    if perturbation_gene_map is not None:
        pm.columns = [perturbation_gene_map[x] for x in pm.columns]

    check = [x in final_adata.var_names for x in pm.columns]
    if sum(check) == 0:
        raise ValueError(f"No perturbations found in adata.var_names. Please check the perturbation_gene_map or perturbation_column.")
    elif sum(check) != len(check):
        if not quiet: print(f"\tMissing {len(check) - sum(check)} perturbations not found in adata.var_names.")

    metrics = ['target_pct_change', 'target_log2fc', 'target_zscore', 'target_gene_expression', 'target_reference_mean', 'target_reference_std'] 
    for m in metrics:
        final_adata.obsm[m] = pd.DataFrame(index=final_adata.obs.index, columns=pm.columns).fillna(0)
    final_adata.obsm['target_gene'] = pd.DataFrame(index=final_adata.obs.index, columns=pm.columns)
    metrics += ['target_gene']

    if groupby is not None:
        groups = final_adata.obs.groupby(groupby).groups
    else:
        groups = {None: final_adata.obs.index}
    
    for group, group_idx in tqdm(groups.items(), desc='Groups', disable=quiet):
        ref_bool = (pm.loc[group_idx].sum(axis=1) == 0).values # get the non perturbed cells for this grouping
        ref_idx = group_idx[ref_bool]
        for target in tqdm(pm.columns, desc='Perturbations', leave=True, total=pm.shape[1], disable=quiet):
            out = _get_target_change_single_perturbation_indexed(final_adata, target, group_idx, ref_idx)
            for m in metrics:
                final_adata.obsm[m].loc[group_idx, target] = out[m]

    if not quiet: print('Target change calculation done. Storing in AnnData...')

    if not collapse_into_obs:
        if not quiet: print(f"\t'collapse_into_obs' is {collapse_into_obs}. Keeping metrics in .obsm...")
    elif sum(pm.sum(axis=1) > 1) == 0: #if only a single perturbation and specified to collapse into .obs
        if not quiet: print(f"\tNo cells with more than 1 perturbation. Adding metrics to .obs...")
        inds = ~(pm.sum(axis=1) == 0)
        for m in metrics:
            final_adata.obs.loc[inds, m] = final_adata.obsm[m].values[pm.values]
    else:
        if not quiet: print(f"\tMultiple perturbations found. Keeping metrics in .obsm...")

    if 'temp_counts' in final_adata.layers:
        final_adata.X = final_adata.layers['temp_counts']
        del final_adata.layers['temp_counts']

    if not inplace:
        return final_adata

############################################################################################################
##### Perturbation Similarity Analysis  #####
############################################################################################################

from scipy.spatial.distance import pdist, squareform
from hdbscan import HDBSCAN

def calculate_adjacency(adata, metric='correlation', inplace=True):
    """
    Get adjacency matrix from adata.
    Args:
        adata (AnnData): AnnData object
        metric (str): metric to use for adjacency matrix, this is passed directly to scipy.spatial.distance.pdist. 
    """
    if inplace:
        adata.obsm['adjacency'] = squareform(pdist(adata.X, metric=metric))
        #if metric is correlation, then  convert to 1 - correlation
        # if metric == 'correlation':
        #     adata.obsm['adjacency'] = 1 - adata.obsm['adjacency']
    else:
        return squareform(pdist(adata.X.T, metric=metric))

def cluster_adjacency(adata, method='leiden', inplace=True, **kwargs):
    """
    Cluster adjacency matrix.
    Args:
        adata (AnnData): AnnData object, assumes .obsm['adjacency'] exists
        method (str): clustering method, either 'hdbscan' or 'leiden'
    """
    #if adjacency matrix does not exist, calculate it
    if 'adjacency' not in adata.obsm.keys():
        print("No adjacency found at .obsm['adjacency']. Calculating adjacency matrix with default params...")
        calculate_adjacency(adata, inplace=True)

    if method == 'hdbscan':
        clusterer = HDBSCAN(metric='precomputed',
                            # min_samples=1,
                            **kwargs
                            )
        clusterer.fit(adata.obsm['adjacency'])
        labels = clusterer.labels_
        if inplace:
            adata.obs['adjacency_cluster'] = labels
            #in this case, set unassigned (rows with -1 from hdbscan) to None
            adata.obs['adjacency_cluster'] = adata.obs['adjacency_cluster'].astype('Int64').astype('str')
            adata.obs['adjacency_cluster'] = adata.obs['adjacency_cluster'].replace('-1', None)
        else:
            return labels
    # elif method == 'leiden': #currently does not work unless adjacency is sparse (ie, 0,1 matrix)
    #     sc.tl.leiden(
    #         adata,
    #         adjacency='adjacency',
    #         key_added='adjacency_cluster'
    #         )
    else: 
        raise ValueError('method must be either hdbscan or leiden')



########################################################################################################################
########################################################################################################################
############# PSEUDO BULK and ZSCORE FUNCTIONS #########################################################################
########################################################################################################################
########################################################################################################################

def _zscore(adata, ref_col='perturbation', ref_val='NTC|NTC', scale_factor = None,):
    
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

def zscore(
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
            normalized_array[rows,] = _zscore(adata[inds,], **kwargs)
        # arr = np.vstack([zscore(adata[inds,], **kwargs) for key, inds in mapping])

        ##append all the inds together: 
        # inds = [inds for key, inds in mapping]

        return normalized_array

    else:
        #if covariates are none then we just apply pseudobulk to the whole matrix (ie single sample)
        return _zscore(adata, **kwargs)

def pseudobulk(adata, groupby, **kwargs):
    """
    Function to apply pseudobulk to anndata object
    Args:
        adata (sc.AnnData): AnnData object with guide calls in adata.obs['guide']
        groupby (str): column in adata.obs to group by
        **kwargs: arguments to pass to pseudobulk function
    """
    adpb = ADPBulk(adata, groupby=groupby, **kwargs)
    pseudobulk_matrix = adpb.fit_transform()
    sample_meta = adpb.get_meta().set_index('SampleName')
    adata = sc.AnnData(pseudobulk_matrix, obs=sample_meta, var=adata.var)

    return adata

##############################################################################################################################
############## DOWNSAMPLE AND SUBSAMPLE FUNCTIONS ############################################################################
##############################################################################################################################

import numpy as np
import random
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.sparse import issparse
from anndata import AnnData
from joblib import Parallel, delayed
import warnings
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt

def subsample_on_covariate(adata: AnnData, column: str, num_cells: int = None, copy: bool = True, seed=999) -> AnnData:
    """
    Subsamples an AnnData object so that all labels in the specified column have the same number of samples.
    
    Parameters:
        adata (AnnData): The AnnData object containing single-cell data.
        column (str): The column name in adata.obs to subsample on.
        num_cells (int): The number of cells to subsample per label (default: None, uses minimum label count).
        copy (bool): Whether to return a copy of the subsampled AnnData object (default: True).
        
    Returns:
        AnnData: The subsampled AnnData object.
    
    Example usage:
        adata = sc.read_h5ad('path_to_your_data.h5ad')
        subsampled_adata = subsample_on_covariate(adata, 'cell_type')
    """
    # Determine the minimum number of samples for any label
    min_count = adata.obs[column].value_counts().min()
    
    if num_cells:
        min_count = min(min_count, num_cells)
    
    # Subsample the data
    indices = adata.obs.groupby(column).apply(lambda x: x.sample(min_count, random_state=seed)).index.get_level_values(1)
    
    if copy:
        return adata[indices].copy()
    else:
        return adata[indices]


def subsample_on_multiple_covariates(adata, columns, num_cells=None, min_cols=None, copy=True, seed=999):
    """
    Subsamples the AnnData object based on multiple covariates.
    
    Parameters:
    - adata: AnnData object
    - columns: list of columns to subsample on
    - num_cells: target number of cells to subsample per group combination
    - min_cols: the columns to calculate the minimum count for each group within this column
    - copy: whether to return a copy of the subsampled AnnData object
    
    Returns:
    - Subsampled AnnData object
    """

    #check to make sure index names are unique
    assert len(adata.obs.index) == len(set(adata.obs.index)), 'Index names are not unique'
    
    # Initialize DataFrame to track group sizes
    group_sizes = pd.DataFrame(adata.obs, columns=columns)
    
    # Calculate group sizes for combinations of covariates
    group_sizes = group_sizes.groupby(columns).size().reset_index(name='count')
    
     # If a minimum column is specified, calculate the minimum count for each group within this column
    if min_cols:
        #get the covariate that are not in min_cols
        other_cols = [col for col in columns if col not in min_cols]
        min_counts = group_sizes.groupby(other_cols).agg({'count': 'min'}).reset_index()
        group_sizes = pd.merge(group_sizes[group_sizes.columns[:-1]], min_counts, on=other_cols, how='left')
    else:
        # Determine the minimum number of samples for any label
        min_count = group_sizes['count'].min()
        if num_cells:
            min_count = min(min_count, num_cells)

        group_sizes['count'] = group_sizes['count'].apply(lambda x: min(x, min_count))
    
    # Sample indices from each group
    np.random.seed(seed)
    inds = []
    for _, row in group_sizes.iterrows():
        filter_condition = (adata.obs[columns] == row[columns]).all(axis=1)
        eligible_indices = adata.obs[filter_condition].index
        sampled_indices = np.random.choice(eligible_indices, int(row['count']), replace=False)
        inds.extend(sampled_indices)
    
    if copy:
        return adata[inds, :].copy()
    else:
        return adata[inds, :]

def _calculate_similarity(matrix_a, matrix_b, metric):
    """
    Calculate the pairwise similarity between two matrices.
    
    Parameters:
        matrix_a (np.ndarray): First data matrix.
        matrix_b (np.ndarray): Second data matrix.
        metric (str): The similarity metric to use.
        
    Returns:
        np.ndarray: Flattened array of pairwise distances.
    """
    distances = cdist(matrix_a, matrix_b, metric)
    return distances[np.triu_indices_from(distances, k=1)]
    
def calculate_label_similarity(
    adata: AnnData,
    label_column: str,
    metric: str = 'euclidean',
    subset: int = None,
    group_subset: bool = True,
    verbose: bool = True,
    n_jobs: int = 5,
    subsample: bool = True
):
    """
    Evaluate the similarity of labeling within single cells in an AnnData object.
    
    Parameters:
        adata (AnnData): The AnnData object containing single-cell data.
        label_column (str): The name of the column in adata.obs that contains the labels.
        metric (str): The similarity metric to use (default: 'euclidean').
        subset (int): The number of cells to use as a random subset for the calculation (default: None).
        group_subset (int): The number of unique labels to compare for across-label similarity (default: None).
        verbose (bool): Whether to print verbose output (default: False).
        n_jobs (int): The number of parallel jobs to run (default: 1).
        subsample (bool): Whether to subsample the data to have equal representation of labels (default: True).
        
    Returns:
        pd.DataFrame: DataFrame containing pairwise similarity results.
    
    Example usage:
        adata = sc.read_h5ad('path_to_your_data.h5ad')
        similarity_results = calculate_label_similarity(adata, 'cell_type', metric='euclidean', subset=100, group_subset=True, verbose=True, n_jobs=4, subsample=True)
        print(similarity_results)
    """
    if verbose:
        print(f"Evaluating labeling similarity using metric: {metric}")

    # Ensure the label_column exists
    if label_column not in adata.obs.columns:
        raise ValueError(f"{label_column} not found in adata.obs")

    # Subsample the data if requested
    if subsample:
        adata = subsample_on_covariate(adata, label_column)
        if verbose:
            print(f"\tSubsampled data to have equal representation of labels. Total cells: {adata.n_obs}. Cells per label: {adata.n_obs / len(adata.obs[label_column].unique())}")

    # Extract labels and data matrix
    labels = adata.obs[label_column]
    data = adata.X

    # If data is sparse, convert to dense array
    if issparse(data):
        data = data.toarray()

    # If subset is specified, randomly sample subset number of cells
    if subset:
        if subset > adata.n_obs:
            warnings.warn(f"Desired subset size ({subset}) is greater than the number of cells in the dataset ({adata.n_obs}). Continuing WITHOUT subsetting...")
        else:
            indices = random.sample(range(adata.n_obs), subset)
            labels = labels.iloc[indices]
            data = data[indices]
            if verbose:
                print(f"\tSubsampled data to {subset} cells")

    unique_labels = labels.unique()

    if verbose:
        print(f"\tTotal unique groups: {len(unique_labels)}")
        print(f"\tTotal cells: {data.shape[0]}")

    # Prepare pairs for comparison
    within_label_pairs = [(unique_labels[I], unique_labels[I]) for I in range(len(unique_labels))]

    # Randomly sample groups if group_subset is specified
    if group_subset:
        num_groups = np.ceil(np.sqrt(len(unique_labels) * 2)).astype(int)
        unique_labels = np.random.choice(unique_labels, num_groups, replace=False)
        if verbose:
            print(f"\tSubsetting # of across groups to approx. match # of within group comparisons")
            print(f"\tTotal across group comparisons: {len(unique_labels) * (len(unique_labels) - 1) / 2}")
        
    across_label_pairs = [(unique_labels[I], unique_labels[j]) for I in range(len(unique_labels)) for j in range(I + 1, len(unique_labels))]

    all_pairs = within_label_pairs + across_label_pairs
    all_pairs_within_label = ['within'] * len(within_label_pairs) + ['across'] * len(across_label_pairs)
    
    if verbose: 
        print(f"\tTotal comparisons: {len(all_pairs)}")

    results = Parallel(n_jobs=n_jobs)(
        delayed(_calculate_similarity)(data[labels == l1], data[labels == l2], metric)
        for l1, l2 in all_pairs
    )

    df = pd.DataFrame({"similarity": np.concatenate(results)})
    df['label1'] = np.concatenate([[l1] * len(results[I]) for I, (l1, l2) in enumerate(all_pairs)])
    df['label2'] = np.concatenate([[l2] * len(results[I]) for I, (l1, l2) in enumerate(all_pairs)])
    df['within'] = df['label1'] == df['label2']
    return df

def get_average_precision_score(res, *args, **kwargs):
    """
    Calculate the average precision score for the labeling similarity in an AnnData object.
    
    Parameters:
        res (pd.DataFrame): DataFrame containing similarity results.
        
    Returns:
        float: The average precision score.
    
    Example usage:
        avg_prec_score = get_average_precision_score(similarity_results)
        print(f"Average Precision Score: {avg_prec_score:.2f}")
    """
    return average_precision_score(~res['within'], res['similarity'])
