from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd

def filter_to_feature_type(
    adata,
    feature_type='Gene Expression'
    ):
    """
    Updates an anndata object to only include the GEX feature type in its .X slot. 
    Optionally adds the removed features to metadata
    """
    ## Subset 
    return adata[:, adata.var['feature_types'] == feature_type]

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
        out[ftype] = adata[:, adata.var['feature_types'] == ftype]

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
    else: 
        return pd.DataFrame(perturbation_matrix, index=adata.obs.index, columns=feature_list)

def add_perturbation_matrix(adata):
    """
    Add a perturbation matrix to an anndata object. 
    """

    adata.obsm['perturbation'] = pm.loc[guides.obs.index, :].values
    guides.uns['perturbation_var'] = pm.columns.tolist()
    guides

    if inplace:
        adata.layers['perturbations'] = generate_perturbation_matrix(
            adata,
            perturbation_col = perturbation_col,
            feature_list = feature_list,
            inplace = inplace
            )
    else:
        return generate_perturbation_matrix(
            adata,
            perturbation_col = perturbation_col,
            feature_list = feature_list,
            inplace = inplace
            )

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
