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

def filter_adata(adata, obs_filters=None, var_filters=None):
    if obs_filters is not None:
        for f in obs_filters:
            adata = adata[adata.obs.query(f).index, :]
        
    if var_filters is not None:
        for f in var_filters:
            adata = adata[:, adata.var.query(f).index]

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
    feature_list = None,
    set_ref_1 = None,
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

    if set_ref_1 is not None:
        #automatically set reference to 1
        #this is done because techincally every cell also is part reference
        #however, some workflows may rely on only seeing a 1 here when when the cell is annotating as reference for 1 of N conditions
        if set_ref_1 not in feature_dict.keys():
            raise ValueError(f"Trying to pass 'set_ref_1' of '{set_ref_1}' to 'generate_perturbation_matrix' but not found in feature list")
        perturbation_matrix[:, feature_dict[set_ref_1]] = 1

    #ensure perturbation matrix is in the same order as adata.X
    # using feature_Dict
    #get the order of the features in the perturbation matrix

    # #split and append all
    # #put perturbation matrix in same order as adata.X
    # if inplace:
    #     adata.layers['perturbations'] = csr_matrix(perturbation_matrix)

    #print num null 

    # if sparse:
    #     return csr_matrix(perturbation_matrix)
    return pd.DataFrame(perturbation_matrix, index=adata.obs.index, columns=feature_list)

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
        adata.obsm['perturbation'] = pm.loc[adata.obs.index, :].values
        cols = pm.columns.tolist()
        adata.uns['perturbation_var'] = dict(zip(cols, range(len(cols))))
    else:
        return pm.loc[adata.obs.index, :]

def split_sort_trim(label, delim='|', delim2='_'):
    #if not string then print
    if type(label) != str:
        return None
    vals = [x.split(delim2)[0] for x in label.split(delim)]
    vals.sort()
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


def calculate_target_change(
    adata,
    perturbation_column,
    reference_label,
    perturbation_gene_map=None,
    check_norm=True,
    quiet=False,
    ):
    """
    Compute the "percent change" for each cell against a reference.
    
    Parameters:
    - adata: anndata.AnnData object containing expression data. assumed to be transformed as desired
    - perturbation_column: column in adata.obs indicating the perturbation/knockdown
    - reference_label: label of the reference population in perturbation_column
    
    Returns:
    - An AnnData object with an additional column in obs containing the "percent knocked down" for each cell.
    """

    final_adata = adata
    
    if not quiet: print(f"Computing percent change for '{perturbation_column}' across {adata.shape[0]} cells...")

    ##check to see if data is normalized to counts per cell
    if check_norm:
        if not quiet: print('\tChecking if data is normalized to counts per cell...')
        sums = np.array(adata.X.sum(axis=1)).flatten()
        dev = np.std(sums[0:10])
        if dev > 1:
            print('Warning: data does not appear to be normalized to counts per cell. Normalizing with sc.pp.normalize_total(). To disable this behavior set check_norm=False.')
            #copy adata to preserve original object
            adata = adata.copy()
            sc.pp.normalize_total(adata)
    

        # Loop through cells
    if not quiet: print('\tComputing percent change for each cell...')

    perturbed_inds = adata.obs[perturbation_column] != reference_label
    padata = adata[perturbed_inds, :] #subset to perturbed cells

    # translate perturbation labels to gene names if necessary
    if perturbation_gene_map is not None:
        target_genes = np.array([perturbation_gene_map[x] for x in padata.obs[perturbation_column]])
    else: 
        target_genes = padata.obs[perturbation_column].values #if no mapping, its assumed that the perturbation directly maps to gene
    
    target_gene_set = list(set(target_genes))
    original_length = len(target_gene_set)
    if not quiet: print(f"\tFound {original_length} unique perturbations in {perturbation_column} column.")
    target_gene_set = [x for x in target_gene_set if x in adata.var_names] #remove genes not in adata.var_names
    if not quiet: print(f"\tRemoved {original_length - len(target_gene_set)} perturbations not found in adata.var_names.")

    # padata.obs['target_gene'] = target_genes
    target_genes_filter = [x in target_gene_set for x in target_genes]
    target_genes = target_genes[target_genes_filter]
    padata = padata[target_genes_filter, target_gene_set] #subset to only target gene and perturbations with measured target genes

    # Get mean expression values for reference population
    if not quiet: print(f'\tComputing mean expression values for reference population: {reference_label}....')
    reference_target_means = adata[~perturbed_inds, target_gene_set].X.toarray().mean(axis=0) #compute per gene mean expression for reference population
    reference_target_stds  = adata[~perturbed_inds, target_gene_set].X.toarray().std(axis=0) #compute per gene stdev for reference population

    # Placeholder for percent knocked down values
    percent_kds = []
    zscores = []
    reference_means = []
    reference_stds = []
    target_gene_expression = []

    if not quiet : print(f'\tComputing percent change for {padata.shape[0]} perturbed cells...')
    for cell, perturb in tqdm(zip(padata.X.toarray(), target_genes), disable=quiet):
        gene_idx = padata.var_names.get_loc(perturb) # get index of target gene

        percent_value = ((cell[gene_idx]) - (reference_target_means[gene_idx])) / (reference_target_means[gene_idx]) * 100
        percent_kds.append(percent_value)

        zscore_value = (cell[gene_idx] - reference_target_means[gene_idx]) / reference_target_stds[gene_idx]
        zscores.append(zscore_value)

        reference_means.append(reference_target_means[gene_idx])
        reference_stds.append(reference_target_stds[gene_idx])
        target_gene_expression.append(cell[gene_idx])
    
    # Add to adata
    metrics = ['target_pct_change', 'target_zscore', 'target_reference_mean', 'target_reference_std', 'target_gene_expression']
    for m in metrics:
        final_adata.obs[m] = np.nan

    final_adata.obs.loc[padata.obs.index, 'target_gene'] = target_genes
    final_adata.obs.loc[padata.obs.index, 'target_pct_change'] = percent_kds
    final_adata.obs.loc[padata.obs.index, 'target_zscore'] = zscores
    final_adata.obs.loc[padata.obs.index, 'target_reference_mean'] = reference_means
    final_adata.obs.loc[padata.obs.index, 'target_reference_std'] = reference_stds
    final_adata.obs.loc[padata.obs.index, 'target_gene_expression'] = target_gene_expression
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
