##########################################################################
# 
# Feature Calling functions (ie guide, HTO, etc)
#
##########################################################################
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
from tqdm import tqdm

from scipy.sparse import csr_matrix


def gm(counts, n_components=2, prob_threshold=0.5, subset=False, subset_minimum=50, nonzero=False, seed=99, **kwargs):
    """
    Fits a Gaussian Mixture Model to the input data.
    Args:
        x: numpy array of data to fit
        n_components: number of components to fit. Default 2
        subset: fraction of data to fit on, this speeds up computation. Default 0.2
        subset_minimum: minimum number of cells to fit on, if subset is too small. Default 50
        nonzero: whether to subset the data to only include nonzero values. Default True
        seed: random seed. Default 0
    """
    
    counts = counts.reshape(-1, 1)

    if nonzero:
        counts = counts[counts > 0].reshape(-1,1)

    counts = np.log10(counts + 1)

    if counts.shape[0] < 10:
        print(f"too few cells ({counts.shape[0]}) to run GMM. Returning -1")
        #return preds of -1
        return np.repeat(-1, counts.shape[0]), np.repeat(-1, counts.shape[0])

    if subset: #optionally subset the data to fit on only a fraction, this speeds up computation
        s = min(int(counts.shape[0]*subset), subset_minimum)
        # print(f"subsetting to {subset}. {int(dat_in.shape[0]*subset)} cells of {dat_in.shape[0]}")
        counts = counts[np.random.choice(counts.shape[0], size=s, replace=False), :]

    try:
        gmm = GaussianMixture(n_components=n_components, random_state=seed, covariance_type="tied", n_init=3, **kwargs)
        pred = gmm.fit(counts)
    except:
        print(f"failed to fit GMM. Returning -1")
        #return preds of -1
        return np.repeat(-1, counts.shape[0]), np.repeat(-1, counts.shape[0])

    #get probability per class
    probs = gmm.predict_proba(counts)
    #get probability for the 'postitive'
    means = gmm.means_.flatten()
    positive = np.argmax(means)
    probs_positive = probs[:,positive]

    return probs_positive > prob_threshold #return confident (ie above threshold) positive calls

def call_features(features, n_jobs=1, inplace=True, quiet=True, **kwargs):
    """
    Accepts an anndata object with adata.X containing the counts of each guide.
    In parallel, fits a GMM to each guide and returns the predicted class for each guide.
    Args:
        features: anndata object with adata.X containing the counts of each guide
    Returns:
        anndata object with adata.X containing the predicted class for each guide
    """
    vp = print if not quiet else lambda *a, **k: None

    lil = features.X.T.tolil()

    vp(f'Running GMM with {n_jobs} workers...')
    #add tqdm to results call
    results = Parallel(n_jobs=n_jobs)(delayed(gm)(lst.toarray(), **kwargs) for lst in tqdm(lil, disable=quiet))
    called = csr_matrix(results).T.astype('uint8')

    if not inplace:
        vp(f"Creating copy AnnData object with guide calls...")
        features = features.copy()

    vp(f"Updating AnnData object with guide calls...")
    features.layers['calls'] = called
    features.obs['num_features'] = called.toarray().sum(axis=1).flatten()
    features.obs['feature_call'] = ['|'.join(features.var_names.values[called[x,:].toarray().flatten() == 1]) for x in range(features.shape[0])]
    features.obs['feature_umi'] = ['|'.join(features.var_names.values[features.X[x,:].toarray().flatten() == 1]) for x in range(features.X.shape[0])]
    if not inplace:
        return features

########################################################################################################################
########################################################################################################################
############# GUIDE CALLING FUNCTIONS ##################################################################################
########################################################################################################################
########################################################################################################################

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

def CLR(df):
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
    clr = CLR(hto_df)

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
    n_components=None,
    filter_on_prob=None,
    per_column=False,
    ):

    print(f'Fitting Gaussian Mixture Model....')
    clr = CLR(hto_df)

    if n_components is None:
        print(f'\t No n_components provided, using {hto_df.shape[1]}')
        n_components = hto_df.shape[1]

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
