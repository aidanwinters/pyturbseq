##########################################################################
# 
# Functions for manipulation and filtering of anndata objects and other data structures
#
##########################################################################
import scanpy as sc
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
import numpy as np
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list
import seaborn as sns
import matplotlib.pyplot as plt
from adpbulk import ADPBulk

########################################################################################################################
########################################################################################################################
############# ADATA FILTERING FUNCTIONS ################################################################################
########################################################################################################################

def filter_adata(adata, obs_filters=None, var_filters=None, copy=True):
    if obs_filters is not None:
        for f in obs_filters:
            adata = adata[adata.obs.query(f).index, :]
        
    if var_filters is not None:
        for f in var_filters:
            adata = adata[:, adata.var.query(f).index]

    if copy:
        return adata.copy()
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
    # feature_type='Gene Expression'
    ):
    """
    Updates an anndata object to only include the GEX feature type in its .X slot. 
    Optionally adds the removed features to metadata
    """
    out = {}
    for ftype in adata.var['feature_types'].unique():
        out[ftype] = adata[:, adata.var['feature_types'] == ftype].copy()
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
    print(feature_dict)

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

    if gene not in adata.var_names:
        out = {}
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
    out['target_gene_expression'] = target_gene_expression[perturbed_bool]
    out['target_reference_mean'] = reference_target_mean
    out['target_reference_std'] = reference_target_std
    out['target_pct_change'] = ((target_gene_expression[perturbed_bool] - reference_target_mean) / reference_target_mean) * 100
    out['target_zscore'] = (target_gene_expression[perturbed_bool] - reference_target_mean) / reference_target_std
    out['target_log2fc'] = np.log2((target_gene_expression[perturbed_bool] + 1) / (reference_target_mean+1))

    return out

def calculate_target_change(
    adata,
    perturbation_column=None,
    reference_value=None,
    perturbation_gene_map=None,
    check_norm=True,
    quiet=False,
    ):
    """
    Compute the "percent change" for each cell against a reference.
    
    Parameters:
    - adata: anndata.AnnData object containing expression data. assumed to be transformed as desired
    - perturbation_column: column in adata.obs indicating the perturbation/knockdown. If passed, then perturbation matrix is regenerated. Else, looks for adata.obsm['perturbation'].
    - reference_label: label of the reference population in perturbation_column
    - perturbation_gene_map: dictionary mapping perturbations to genes. If None, then perturbation_column is assumed to be gene names.
    - check_norm: if True, checks if data is normalized to counts per cell. If not, normalizes.
    - quiet: if False, prints progress
    
    Returns:
    - An AnnData object with an additional column in obs containing the "percent knocked down" for each cell.
    """

    final_adata = adata    
    if not quiet: print(f"Computing percent change for '{perturbation_column}' across {adata.shape[0]} cells...")

    #check inputs: 
    duplicated_genes = adata.var.index.duplicated()
    if sum(duplicated_genes) > 0:
        raise ValueError(f"Duplicated gene names found in adata.var.index. Please remove duplicated gene names before running this function. \nGene names found: {list(adata.var.index[duplicated_genes])}")

    ##check to see if data is normalized to counts per cell
    if check_norm:
        if not quiet: print('\tChecking if data is normalized to counts per cell...')
        if cells_not_normalized(adata):
            if not quiet: print('\tData is not normalized to counts per cell. Normalizing...')
            adata = adata.copy()
            sc.pp.normalize_total(adata)
        # Loop through cells
    if not quiet: print('\tComputing percent change for each cell...')

    #convert to numpy if sparse
    if type(adata.X) == csr_matrix:
        adata.X = adata.X.toarray()

    #if no perturbation matrix, create one
    if perturbation_column is not None:
        if not quiet: print(f"\tGenerating perturbation matrix from '{perturbation_column}' column...")
        pm = get_perturbation_matrix(adata, perturbation_column, reference_value=reference_value, inplace=False, verbose=not quiet)
    elif 'perturbation' in adata.obsm.keys():
        pm = adata.obsm['perturbation']
    else: 
        raise ValueError("No perturbation matrix found in adata.obsm. Please provide a perturbation_column or run get_perturbation_matrix first.")

    if not quiet: print(f"\tFound {pm.shape[1]} unique perturbations in {perturbation_column} column.")

    #check that the gene a perturbation maps to is actually in adata
    if perturbation_gene_map is not None:
        #for now we assume all the perturbations are in the perturbation_gene_map
        pm.columns = [perturbation_gene_map[x] for x in pm.columns]
    
    check = [x in adata.var_names for x in pm.columns]
    if sum(check) == 0:
        raise ValueError(f"No perturbations found in adata.var_names. Please check the perturbation_gene_map or perturbation_column.")
    elif sum(check) != len(check):
        if not quiet: print(f"\tMissing {len(check) - sum(check)} perturbations not found in adata.var_names.")
    

    #reference labels are where pm row sums are 0
    ref_bool = (pm.sum(axis=1) == 0).values
    
    zscore_matr = np.zeros((adata.shape[0], pm.shape[1]))
    pct_change_matr = np.zeros((adata.shape[0], pm.shape[1]))
    log2fc_matr = np.zeros((adata.shape[0], pm.shape[1]))
    target_gex_matr = np.zeros((adata.shape[0], pm.shape[1]))
    reference_means = np.zeros(pm.shape[1])
    reference_stds = np.zeros(pm.shape[1])
    for i, (prtb, prtb_bool) in tqdm(enumerate(pm.items()), total=pm.shape[1], disable=quiet):
        prtb_bool = prtb_bool.values
        out = _get_target_change_single_perturbation(adata, prtb, prtb_bool, ref_bool)
        pct_change_matr[prtb_bool, i] = out['target_pct_change']
        zscore_matr[prtb_bool, i] = out['target_zscore']
        log2fc_matr[prtb_bool, i] = out['target_log2fc']
        target_gex_matr[prtb_bool, i] = out['target_gene_expression']
        reference_means[i] = out['target_reference_mean']
        reference_stds[i] = out['target_reference_std']

    #if cells got more than 1 perturbation, then set these as .obs else 
    if sum(pm.sum(axis=1) > 1) > 0:
        if not quiet: print(f"Cells with more than 1 perturbation found. Adding to .obsm...")
        final_adata.uns['target_reference_mean'] = reference_means
        final_adata.uns['target_reference_std'] = reference_stds
        final_adata.obsm['target_pct_change'] = pd.DataFrame(pct_change_matr, index=final_adata.obs.index, columns=pm.columns)
        final_adata.obsm['target_zscore'] = pd.DataFrame(zscore_matr, index=final_adata.obs.index, columns=pm.columns)
        final_adata.obsm['target_log2fc'] = pd.DataFrame(log2fc_matr, index=final_adata.obs.index, columns=pm.columns)
        final_adata.obsm['target_gene_expression'] = pd.DataFrame(target_gex_matr, index=final_adata.obs.index, columns=pm.columns)
    else:
        if not quiet: print(f"No cells with more than 1 perturbation. Adding to .obs...")
        final_adata.obs['target_reference_mean'] = reference_means[np.argmax(pm.values, axis=1)]
        final_adata.obs['target_reference_std'] = reference_stds[np.argmax(pm.values, axis=1)]

        # pm = pm.stack()
        final_adata.obs.loc[~ref_bool, 'target_pct_change'] = pct_change_matr[pm.values]
        final_adata.obs.loc[~ref_bool, 'target_zscore'] = zscore_matr[pm.values]
        final_adata.obs.loc[~ref_bool, 'target_log2fc'] = log2fc_matr[pm.values]
        final_adata.obs.loc[~ref_bool, 'target_gene_expression'] = target_gex_matr[pm.values]

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

def zscore(adata, ref_col='perturbation', ref_val='NTC|NTC', scale_factor = None,):
    
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

    else:
        #if covariates are none then we just apply pseudobulk to the whole matrix (ie single sample)
        return zscore(adata, **kwargs)


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
