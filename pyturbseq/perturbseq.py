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

def assign_guides(guides, max_ratio_2nd_1st=0.35, min_total_counts=10):
    """ 
    Assumes that guides is an AnnData object with counts at guides.X
    """

    matr = guides.X.toarray()
    total_counts = matr.sum(axis=1)
    #sort matr within each row
    matr_sort = np.sort(matr, axis=1)
    ratio_2nd_1st = matr_sort[:, -2] / matr_sort[:, -1]

    #get argmax for each row
    argmax = np.argmax(matr, axis=1)
    assigned = guides.var.index[argmax].values


    #set any that don't pass filter to none
    assigned[(ratio_2nd_1st > max_ratio_2nd_1st) | (total_counts < min_total_counts)] = None

    #print how many guides did not pass thresholds
    print(f"{(ratio_2nd_1st > max_ratio_2nd_1st).sum()} guides did not pass ratio filter")
    print(f"{(total_counts < min_total_counts).sum()} guides did not pass total counts filter")
    #print total that are None
    print(f"{(assigned == None).sum()} cells did not pass thresholds")

    guides.obs['assigned_perturbation'] = assigned
    guides.obs['guide_ratio_2nd_1st'] = ratio_2nd_1st
    guides.obs['guide_total_counts'] = total_counts

    return guides


### FEATURE CALLING FUNCTIONS


import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

def CLR_transform(df):
    '''
    Implements the Centered Log-Ratio (CLR) transformation often used in compositional data analysis.
    
    Args:
    - df (pd.DataFrame): The input data frame containing features to be transformed.
    
    Returns:
    - pd.DataFrame: A data frame with the CLR-transformed features.
    
    Notes:
    - Reference: "Visualizing and interpreting single-cell gene expression datasets with similarity weighted nonnegative embedding" (https://doi.org/10.1038/nmeth.4380)
    - The function first applies the log transform (after adding 1 to handle zeros). 
      Then, for each feature, it subtracts the mean value of that feature, thus "centering" the log-transformed values.
    '''
    logn1 = np.log(df + 1)
    T_clr = logn1.sub(logn1.mean(axis=0), axis=1)
    return T_clr

def get_gm(col, n_components=2):
    '''
    Fits a Gaussian Mixture model to a given feature/column and assigns cluster labels.
    
    Args:
    - col (np.array): The input column/feature to cluster.
    - n_components (int): Number of mixture components to use (default=2).
    
    Returns:
    - tuple: Cluster labels assigned to each data point and the maximum probabilities of cluster membership.
    '''
    
    # Reshaping column to a 2D array, required for GaussianMixture input
    col = col.reshape(-1, 1)
    
    # Fitting the Gaussian Mixture model
    gm = GaussianMixture(n_components=n_components, random_state=0).fit(col)
    gm_assigned = gm.predict(col)

    # Reorder cluster labels so that they are consistent with the order of mean values of clusters
    mapping = {}
    classes = set(gm_assigned)
    class_means = [(col[gm_assigned == c].mean(), c) for c in classes]
    ordered = sorted(class_means)
    mapping = {x[1]: i for i, x in enumerate(ordered)}
    gm_assigned = np.array([mapping[x] for x in gm_assigned])

    max_probability = gm.predict_proba(col).max(axis=1)
    return (gm_assigned, max_probability)

def assign_hto_per_column_mixtureModel(hto_df, filter_on_prob=None):
    '''
    Assigns labels to each data point in the provided dataframe based on Gaussian Mixture clustering results.
    
    Args:
    - hto_df (pd.DataFrame): The input data frame containing features to be clustered.
    - filter_on_prob (float, optional): If provided, it may be used to filter results based on probability thresholds.
    
    Returns:
    - tuple: A data frame summarizing cluster assignments and two arrays with cluster labels and max probabilities.
    '''
    
    # Apply CLR transform to the dataframe
    clr = CLR_transform(hto_df)

    # Fit Gaussian Mixture to each column in the transformed dataframe
    n_components = 2
    gms = [get_gm(clr[c].values, n_components=n_components) for c in hto_df.columns]
    gm_assigned = np.array([x[0] for x in gms]).T
    max_probability = np.array([x[1] for x in gms]).T

    # Define a helper function to determine cluster assignment based on Gaussian Mixture results
    def assign(x, cols):
        if sum(x) > 1:
            return 'multiplet'
        elif sum(x) < 1:
            return 'unassigned'
        else:
            return cols[x == 1].values[0]

    # Use the helper function to determine cluster assignment for each data point
    trt = [assign(x, hto_df.columns) for x in gm_assigned]

    # Create a summary dataframe
    df = pd.DataFrame({
        'treatment': trt,
        'HTO_max_prob': max_probability.max(axis=1),
        'ratio_max_prob_to_total': max_probability.max(axis=1) / max_probability.sum(axis=1),
        'total_HTO_counts': hto_df.sum(axis=1),
    })

    return df, gm_assigned, max_probability


def assign_hto_mixtureModel(
    hto_df,
    n_components=2,
    filter_on_prob=None,
    per_column=False,
    ):

    print(f'Fitting Gaussian Mixture Model....')
    clr = CLR_transform(hto_df)

    gm = GaussianMixture(n_components=n_components, random_state=0).fit(clr.values)
    gm_assigned = gm.predict(clr.values)

    max_probability = gm.predict_proba(clr.values).max(axis=1)


    ## Perform mapping by assigning the maximum CLR to each predicted class
    mapping = {}
    for c in set(gm_assigned):
        mapping[c] = clr.loc[gm_assigned == c, clr.columns].mean(axis=0).idxmax()
    
    trt = pd.Series([mapping[x] for x in gm_assigned]) # Get treatment cal
    if filter_on_prob is not None: 
        print(f'\t Filtering below GM prediction probability {filter_on_prob}')
        trt[max_probability <= filter_on_prob] =  None 

    df = pd.DataFrame({
        'total_HTO_counts': hto_df.sum(axis=1),
        'treatment': trt.values,
        'HTO_max_prob': max_probability
    }, index=hto_df.index)
    return df



########################################################################################################################
########################################################################################################################
############# KNOCKDOWN FUNCTIONS ######################################################################################
########################################################################################################################
########################################################################################################################

#deprecated
# def plot_kd(adata, gene, ):
#     gene_vals = adata[:,gene].X.toarray().flatten()
#     ##plot AR for AR KD vs NTC|NTC
#     gene_inds = (adata.obs['perturbation'].str.contains(gene + '\|')) | (adata.obs['perturbation'].str.contains('\|' + gene))
#     NTC_inds = adata.obs['perturbation'] == 'NTC|NTC'
#     print(f"Number of obs in NTC|NTC: {np.sum(NTC_inds)}")
#     print(f"Number of obs in {gene} KD: {np.sum(gene_inds)}")


#     plt.hist(gene_vals[NTC_inds], label='NTC', alpha=0.5, bins=30)
#     plt.hist(gene_vals[gene_inds], label=gene + ' KD', alpha=0.5, bins=30)
#     #add mean line for each group
#     plt.axvline(gene_vals[NTC_inds].mean(), color='blue')
#     plt.axvline(gene_vals[gene_inds].mean(), color='orange')
#     plt.legend()
#     plt.show()


# def percent_knocked_down_per_cell(adata, perturbation_column, reference_label):
#     """
#     Compute the "percent knocked down" for each cell against a reference.
    
#     Parameters:
#     - adata: anndata.AnnData object containing expression data
#     - perturbation_column: column in adata.obs indicating the perturbation/knockdown
#     - reference_label: label of the reference population in perturbation_column
    
#     Returns:
#     - An AnnData object with an additional column in obs containing the "percent knocked down" for each cell.
#     """
    
#     # Get mean expression values for reference population
#     reference_means = adata[adata.obs[perturbation_column] == reference_label, :].X.toarray().mean(axis=0)
    
#     # Placeholder for percent knocked down values
#     percent_kd_values = []
#     reference_mean_values = []
    
#     # Loop through cells
#     for cell, perturb in zip(adata.X.toarray(), adata.obs[perturbation_column]):
#         if perturb == reference_label or perturb not in adata.var_names:
#             percent_kd_values.append(None)
#             reference_mean_values.append(None)
#             continue

#         # print('here')
#         gene_idx = adata.var_names.get_loc(perturb)
        
#         percent_value = (1 - ((cell[gene_idx]+1) / (reference_means[gene_idx] + 1))) * 100
#         percent_kd_values.append(percent_value)
#         reference_mean_values.append(reference_means[gene_idx])
    
#     # Add to adata
#     adata.obs['percent_kd'] = percent_kd_values
#     adata.obs['reference_mean'] = reference_mean_values
    
#     return adata



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



