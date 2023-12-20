import scanpy as sc
import pandas as pd
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, leaves_list

def filter_to_feature_type(
    adata,
    feature_type='Gene Expression'
    ):
    """
    Updates an anndata object to only include the GEX feature type in its .X slot. 
    Optionally adds the removed features to metadata
    """
    ## Subset 
    return adata[:, adata.var['feature_types'] == feature_type].copy()

def split_by_feature_type(
    adata,
    # feature_type='Gene Expression'
    ):
    """
    Updates an anndata object to only include the GEX feature type in its .X slot. 
    Optionally adds the removed features to metadata
    """
    ## Subset 
    out = {}
    for ftype in adata.var['feature_types'].unique():
        out[ftype] = adata[:, adata.var['feature_types'] == ftype].copy()

    return out

#read in the feature call column, split all of them by delimiter 
def generate_perturbation_matrix(
    adata,
    perturbation_col = 'feature_call',
    delim = '|',
    feature_list = None,
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

    adata.obsm['perturbation'] = pm.loc[adata.obs.index, :].values
    cols = pm.columns.tolist()
    adata.uns['perturbation_var'] = dict(zip(cols, range(len(cols))))
    return adata

def split_sort_trim(label, delim='|', delim2='_'):
    #if not string then print
    if type(label) != str:
        return None
    vals = [x.split(delim2)[0] for x in label.split(delim)]
    vals.sort()
    return delim.join(vals)

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

        percent_value = ((cell[gene_idx]+1) - (reference_target_means[gene_idx] + 1)) / (reference_target_means[gene_idx] + 1) * 100
        percent_kds.append(percent_value)

        zscore_value = (cell[gene_idx] - reference_target_means[gene_idx]) / reference_target_stds[gene_idx]
        zscores.append(zscore_value)

        reference_means.append(reference_target_means[gene_idx])
        reference_stds.append(reference_target_stds[gene_idx])
        target_gene_expression.append(cell[gene_idx])
    
    # Add to adata
    final_adata.obs.loc[padata.obs.index, 'target_gene'] = target_genes
    final_adata.obs.loc[padata.obs.index, 'target_pct_change'] = percent_kds
    final_adata.obs.loc[padata.obs.index, 'target_zscore'] = zscores
    final_adata.obs.loc[padata.obs.index, 'target_reference_mean'] = reference_means
    final_adata.obs.loc[padata.obs.index, 'target_reference_std'] = reference_stds
    final_adata.obs.loc[padata.obs.index, 'target_gene_expression'] = target_gene_expression
    return final_adata