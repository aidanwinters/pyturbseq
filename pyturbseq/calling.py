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
import os

from sklearn.mixture import GaussianMixture
from joblib import Parallel, delayed
from tqdm import tqdm

from scipy.sparse import csr_matrix, issparse



## Functions for feature calling

def gm(counts, n_components=2, prob_threshold=0.5, subset=False, subset_minimum=50, nonzero=False, calling_min_count=1, seed=99, **kwargs):
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

    if counts.max() < calling_min_count:
        print(f"max count ({counts.max()}) is less than calling_min_count ({calling_min_count}). Returning no calls.")
        #return preds of -1
        return np.repeat(0, counts.shape[0])

    counts = np.log10(counts + 1)

    if counts.shape[0] < 10:
        print(f"too few cells ({counts.shape[0]}) to run GMM. Returning no calls.")
        #return preds of -1
        return np.repeat(-1, counts.shape[0]), np.repeat(-1, counts.shape[0])

    if subset: #optionally subset the data to fit on only a fraction, this speeds up computation
        s = min(int(counts.shape[0]*subset), subset_minimum)
        # print(f"subsetting to {subset}. {int(dat_in.shape[0]*subset)} cells of {dat_in.shape[0]}")
        counts = counts[np.random.choice(counts.shape[0], size=s, replace=False), :]

    try:
        gmm = GaussianMixture(n_components=n_components, random_state=seed, covariance_type="tied", n_init=5, **kwargs)
        pred = gmm.fit(counts)
    except Exception:
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

def call_features(features, feature_type=None, feature_key=None, min_feature_umi=1, n_jobs=1, inplace=True, quiet=True, **kwargs):
    """
    Accepts an anndata object with adata.X containing the counts of each guide.
    In parallel, fits a GMM to each guide and returns the predicted class for each guide.
    Args:
        features: anndata object with adata.X containing the counts of each guide

    Returns:
        anndata object with adata.X containing the predicted class for each guide
    """
    vp = print if not quiet else lambda *a, **k: None

    if feature_type is not None:
        #confirm feature type is in var['feature_type']
        vp(f"Subsetting features to {feature_type}...")
        assert feature_type in features.var['feature_types'].unique(), f"feature_type {feature_type} not found in var['feature_types']"
        feature_list = features.var.index[features.var['feature_types'] == feature_type]
        vp(f"Found {len(feature_list)} features of type {feature_type}")
        lil = features[:,features.var.index[features.var['feature_types'] == feature_type]].X.T.tolil()
    else:
        feature_list = features.var.index
        lil = features.X.T.tolil()

    vp(f'Running GMM with {n_jobs} workers...')
    #add tqdm to results call
    results = Parallel(n_jobs=n_jobs)(delayed(gm)(lst.toarray(), **kwargs) for lst in tqdm(lil, disable=quiet))
    called = csr_matrix(results).T.astype('uint8')

    # Remove calls for features that do not pass threshold (indicating issue with that guide)

    if not inplace:
        vp(f"Creating copy AnnData object with guide calls...")
        features = features.copy()

    vp(f"Updating AnnData object with guide calls...")
    if feature_key:
        features.obsm[f'{feature_key}_calls'] = called
        features.uns[f'{feature_key}'] = feature_list.values
        features.obs[f'num_{feature_key}'] = called.toarray().sum(axis=1).flatten()
        features.obs[f'{feature_key}_call'] = ['|'.join(feature_list[called[x,:].toarray().flatten() == 1]) for x in range(features.shape[0])]
    else:
        features.obsm['calls'] = called
        features.uns['features'] = feature_list.values
        features.obs['num_features'] = called.toarray().sum(axis=1).flatten()
        features.obs['feature_call'] = ['|'.join(feature_list[called[x,:].toarray().flatten() == 1]) for x in range(features.shape[0])]
    # features.obs['feature_umi'] = ['|'.join(features[x,feature_list].X.toarray().astype('int').astype('str').flatten()) for x in range(features.X.shape[0])]
    if not inplace:
        return features

def calculate_feature_call_metrics(
    features,
    feature_type=None,
    inplace=True,
    topN = [1,2],
    quiet=False):
    vp = print if not quiet else lambda *a, **k: None

    if feature_type is not None:
        features_subset = features[:,features.var.index[features.var['feature_types'] == feature_type]]
    else:
        features_subset = features

    if not inplace:
        vp(f"Creating copy AnnData object with guide calls...")
        features = features.copy()

    features.obs['total_feature_counts'] = features_subset.X.sum(axis=1)
    features.obs['log1p_total_feature_counts'] = np.log1p(features_subset.X.sum(axis=1))
    features.obs['log10_total_feature_counts'] = np.log10(features_subset.X.sum(axis=1)+1)


    X = features_subset.X.toarray()
    X.sort(axis=1)
    features.obs['ratio_2nd_1st_feature'] = (X[:,-2]) / (X[:,-1])
    features.obs['log2_ratio_2nd_1st_feature'] = np.log2(features.obs['ratio_2nd_1st_feature'])

    if (topN is not None) and (len(topN) > 0):
        for n in topN:
            topN = X[:,-n:]
            prop_topN = topN.sum(axis=1) / X.sum(axis=1)
            features.obs[f'pct_top{n}_features'] = prop_topN * 100

    if not inplace:
        return features


########################################################################################################################
## Functions for parsing calls
########################################################################################################################
def parse_dual_guide_df(
    calls,
    position_annotation: list = None,
    call_sep: str = '|',
    collapse_same_target: bool = True,
    sort_perturbation: bool = True,
    perturbation_name: str = 'perturbation',
    position_extraction = lambda x: x.split('_')[-1], #default is last underscore
    perturbation_extraction = lambda x: x.split('_')[0],
    library_reference: [pd.DataFrame, str] = None,
    feature_key=None
    ):
    """
    Parse dual guide calls into a single perturbation annotation.
    This was built to parse calls in the format of 'sgRNA_A|sgRNA_B' into a single perturbation annotation.

    Args:
    calls (pd.DataFrame): DataFrame of calls with columns 'feature_call' and 'num_features'.
    position_annotation (list): List of two strings indicating the position annotation of the sgRNAs (recommended). If None, then position extraction is used to get position annotations.
    call_sep (str): Separator between calls.
    collapse_same_target (bool): Collapse dual perturbations that target the same gene.
    sort_perturbation (bool): Sort the perturbations in the annotation.
    perturbation_name (str): Name of the new perturbation annotation.
    position_extraction (function): Function to extract position annotation from the call.
    perturbation_extraction (function): Function to extract perturbation annotation from the call.
    library_reference (pd.DataFrame or str): Reference library to check for perturbations in.
    """

    num_feature_col = f'num_{feature_key}' if feature_key else 'num_features'
    feature_call_col = f'{feature_key}_call' if feature_key else 'feature_call' 

    cols = [num_feature_col, feature_call_col]
    #check if cols are in 
    if not all([x in calls.columns for x in cols]):
        raise ValueError(f"Missing columns {cols} in calls. Did you load the sgRNA calls?")

    dual_calls = calls.loc[calls[num_feature_col] == 2, cols].copy()
    duals_split = dual_calls[feature_call_col].str.split(call_sep, expand=True)

    if position_annotation is None:
        print("Extracting position annotation from calls...")
        positions = duals_split.apply(lambda row: [position_extraction(x) for x in row], axis=1)
        position_annotation = list(set([x for y in positions for x in y]))
        print(f"Detected position annotation: {position_annotation}")

    #check that position_annotation is len 2
    if len(position_annotation) != 2:
        raise ValueError(f"Position annotation must be length 2, got {len(position_annotation)}")


    position_annot_srt = sorted(position_annotation)
    #sort by the position letter to get into the same order as the annotation
    sgRNA_wPos_cols = [f'sgRNA_fullID_{x}' for x in position_annot_srt]
    sgRNA_cols= [f'sgRNA_{x}' for x in position_annot_srt]
    dual_calls[sgRNA_wPos_cols] = duals_split.apply(lambda row: sorted(row, key=lambda x: position_extraction(x)), axis=1).tolist()
    for c_wPos, c in zip(sgRNA_wPos_cols, sgRNA_cols):
        dual_calls[c] = dual_calls[c_wPos].apply(lambda x: perturbation_extraction(x))

    dual_calls[f"{perturbation_name}_fullID"] = dual_calls[sgRNA_wPos_cols].apply(lambda row: '|'.join(row), axis=1)

    ## adding annotations
    pos1, pos2 = position_annot_srt

    #check if recombined (ie doubled up on the same position)
    same_position = duals_split.apply(lambda row: position_extraction(row[0]) == position_extraction(row[1]), axis=1)
    dual_calls.loc[same_position, f'{perturbation_name}_status'] = 'same_position'

    if library_reference:
        #check if isa dataframe
        if isinstance(library_reference, pd.DataFrame):
            ref = library_reference
        elif (isinstance(library_reference, str) and os.path.exists(library_reference)):
            ref = pd.read_csv(library_reference)
        else: 
            raise ValueError(f"library_reference must be a dataframe or a path to a csv file, got {type(library_reference)}")
        # confirm the columns of ref match sgRNA_cols
        if not all([x in ref.columns for x in sgRNA_cols]):
            raise ValueError(f"Missing columns {sgRNA_cols} in library reference")

        print(f"Parsing library reference with {ref.shape[0]} sgRNA pairs...")
        ref['as_str'] = ref[sgRNA_cols].apply(lambda row: '|'.join(row), axis=1).to_list()
        dual_calls.loc[dual_calls[f"{perturbation_name}_fullID"].isin(ref['as_str']), f'{perturbation_name}_status'] = 'in_library'

    if collapse_same_target & all(dual_calls[f'sgRNA_{pos1}'] == dual_calls[f'sgRNA_{pos2}']):
        #check to see if all dual perturbations have the same target and, if indicated, collapse
        print("Dual guide same target found, collapsing to same target..")
        dual_calls[perturbation_name] = dual_calls[f'sgRNA_{pos1}']
    elif sort_perturbation:
        #this assumes that gene name is stored before the underscore
        print(f"Sorting sgRNA for '{perturbation_name}' annotation...")
        sgRNAs = dual_calls[sgRNA_cols].apply(lambda row: sorted(row, key=lambda x: str(x)), axis=1).to_list()
        dual_calls[perturbation_name] = [f"{x[0]}|{x[1]}" for x in sgRNAs]
    else:
        sgRNAs = dual_calls[sgRNA_cols].values.tolist()
        dual_calls[perturbation_name] = [f"{x[0]}|{x[1]}" for x in sgRNAs]
    
    # merge with original calls (drop any existing columns that have been recalculated)
    dual_calls = dual_calls.drop(columns=cols)
    out = calls.drop(columns=dual_calls.columns, errors='ignore').merge(dual_calls, how='left', left_index=True, right_index=True)
    return out

def parse_dual_guide(
    adata: sc.AnnData,
    inplace: bool = False,
    **kwargs):
    """
    Parse dual guide calls into a single perturbation annotation. A wrapper AnnData function for parse_dual_guide_df.
    """
    adata = adata.copy() if not inplace else adata
    adata.obs = parse_dual_guide_df(adata.obs, **kwargs)
    
    if not inplace:
        return adata



########################################################################################################################
## Functions for HTO calling
########################################################################################################################
def CLR(x):
    '''
    Implements the Centered Log-Ratio (CLR) transformation often used in compositional data analysis.
    
    Args:
    - 
    
    Returns:
    -  A numpy array with the CLR-transformed features.
    
    Notes:
    - Reference: "Visualizing and interpreting single-cell gene expression datasets with similarity weighted nonnegative embedding" (https://doi.org/10.1038/nmeth.4380)
    - The function first applies the log transform (after adding 1 to handle zeros). 
      Then, for each feature, it subtracts the mean value of that feature, thus "centering" the log-transformed values.
    '''
    log1p = np.log1p(x)
    #center each column at 0
    T_clr = log1p - log1p.mean(axis=0)
    return T_clr

def _multivariate_clr_gm(
    x,
    n_components=None,
    filter_on_prob=None,
    per_column=False,
    ):

    print(f'Fitting Gaussian Mixture Model....')
    clr = CLR(x)

    if n_components is None:
        print(f'\t No n_components provided, using {x.shape[1]}')
        n_components = x.shape[1]

    gm = GaussianMixture(n_components=n_components, random_state=0).fit(clr)
    gm_assigned = gm.predict(clr)
    max_probability = gm.predict_proba(clr).max(axis=1)
    return clr, gm_assigned, max_probability

def call_hto(
    counts: sc.AnnData,
    features=None,
    feature_type=None,
    rename=None,
    probability_threshold=None,
    inplace=False,
    ):
    """
    """
    if (features is None) & (feature_type is None):
        features = counts.var.index
    elif feature_type:
        print(f"Filtering features by type: {feature_type}")
        features = counts.var.query(f"feature_types == '{feature_type}'").index

    print(f"Found {len(features)} features to use for HTO calling.")
    x = counts[:, features].X
    #if x is sparse, convert to dense
    if issparse(x):
        x = x.toarray()   

    clr, assigned, probs = _multivariate_clr_gm(x, n_components=len(features))

    #map the assignment to the max feature
    #get mean of each clr column for each group and assign each cell to the group with the highest mean
    clr_means = pd.DataFrame(clr, columns=features).groupby(assigned).mean()
    mapping = clr_means.T.idxmax(axis=0).to_dict()

    print(f"Assigned {counts.shape[0]} cells to {len(mapping)} HTOs.")
    
    #confirm that nothing maps to the same group
    if len(set(mapping.values())) != len(mapping):
        raise ValueError("Assignement failed. GMM was not able to confidently assign a distinct component to each HTO.")

    df = pd.DataFrame({
        'HTO_total_counts': np.sum(x, axis=1),
        'HTO': [mapping[x] for x in assigned],
        'HTO_max_probability': probs
    }, index=counts.obs.index)

    if probability_threshold:
        print(f"Filtering HTO calls with probability > {probability_threshold}")
        df['HTO'] = np.where(df['HTO_max_probability'] > probability_threshold, df['HTO'], None)

    if rename:
        df[rename] = df['HTO']

    counts = counts.copy() if not inplace else counts

    for col in df.columns:
        counts.obs[col] = df[col]
    
    if not inplace: 
        return counts


########################################################################################################################
## Calling QC functions
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


########################################################################################################################
########################################################################################################################
############# DEPRECATED CALLING FUNCTIONS ##################################################################################
########################################################################################################################
########################################################################################################################
# def assign_guides(guides, max_ratio_2nd_1st=0.35, min_total_counts=10):
#     """ 
#     Assumes that guides is an AnnData object with counts at guides.X
#     """

#     matr = guides.X.toarray()
#     total_counts = matr.sum(axis=1)
#     #sort matr within each row
#     matr_sort = np.sort(matr, axis=1)
#     ratio_2nd_1st = matr_sort[:, -2] / matr_sort[:, -1]

#     #get argmax for each row
#     argmax = np.argmax(matr, axis=1)
#     assigned = guides.var.index[argmax].values


#     #set any that don't pass filter to none
#     assigned[(ratio_2nd_1st > max_ratio_2nd_1st) | (total_counts < min_total_counts)] = None

#     #print how many guides did not pass thresholds
#     print(f"{(ratio_2nd_1st > max_ratio_2nd_1st).sum()} guides did not pass ratio filter")
#     print(f"{(total_counts < min_total_counts).sum()} guides did not pass total counts filter")
#     #print total that are None
#     print(f"{(assigned == None).sum()} cells did not pass thresholds")

#     guides.obs['assigned_perturbation'] = assigned
#     guides.obs['guide_ratio_2nd_1st'] = ratio_2nd_1st
#     guides.obs['guide_total_counts'] = total_counts

#     return guides


### FEATURE CALLING FUNCTIONS
# import numpy as np
# import pandas as pd
# from sklearn.mixture import GaussianMixture

# def CLR(df):
#     '''
#     Implements the Centered Log-Ratio (CLR) transformation often used in compositional data analysis.
    
#     Args:
#     - df (pd.DataFrame): The input data frame containing features to be transformed.
    
#     Returns:
#     - pd.DataFrame: A data frame with the CLR-transformed features.
    
#     Notes:
#     - Reference: "Visualizing and interpreting single-cell gene expression datasets with similarity weighted nonnegative embedding" (https://doi.org/10.1038/nmeth.4380)
#     - The function first applies the log transform (after adding 1 to handle zeros). 
#       Then, for each feature, it subtracts the mean value of that feature, thus "centering" the log-transformed values.
#     '''
#     logn1 = np.log(df + 1)
#     T_clr = logn1.sub(logn1.mean(axis=0), axis=1)
    
#     return T_clr

# def get_gm(col, n_components=2):
#     '''
#     Fits a Gaussian Mixture model to a given feature/column and assigns cluster labels.
    
#     Args:
#     - col (np.array): The input column/feature to cluster.
#     - n_components (int): Number of mixture components to use (default=2).
    
#     Returns:
#     - tuple: Cluster labels assigned to each data point and the maximum probabilities of cluster membership.
#     '''
    
#     # Reshaping column to a 2D array, required for GaussianMixture input
#     col = col.reshape(-1, 1)
    
#     # Fitting the Gaussian Mixture model
#     gm = GaussianMixture(n_components=n_components, random_state=0).fit(col)
#     gm_assigned = gm.predict(col)

#     # Reorder cluster labels so that they are consistent with the order of mean values of clusters
#     mapping = {}
#     classes = set(gm_assigned)
#     class_means = [(col[gm_assigned == c].mean(), c) for c in classes]
#     ordered = sorted(class_means)
#     mapping = {x[1]: i for i, x in enumerate(ordered)}
#     gm_assigned = np.array([mapping[x] for x in gm_assigned])

#     max_probability = gm.predict_proba(col).max(axis=1)
#     return (gm_assigned, max_probability)

# def assign_hto_per_column_mixtureModel(hto_df, filter_on_prob=None):
#     '''
#     Assigns labels to each data point in the provided dataframe based on Gaussian Mixture clustering results.
    
#     Args:
#     - hto_df (pd.DataFrame): The input data frame containing features to be clustered.
#     - filter_on_prob (float, optional): If provided, it may be used to filter results based on probability thresholds.
    
#     Returns:
#     - tuple: A data frame summarizing cluster assignments and two arrays with cluster labels and max probabilities.
#     '''
    
#     # Apply CLR transform to the dataframe
#     clr = CLR(hto_df)

#     # Fit Gaussian Mixture to each column in the transformed dataframe
#     n_components = 2
#     gms = [get_gm(clr[c].values, n_components=n_components) for c in hto_df.columns]
#     gm_assigned = np.array([x[0] for x in gms]).T
#     max_probability = np.array([x[1] for x in gms]).T

#     # Define a helper function to determine cluster assignment based on Gaussian Mixture results
#     def assign(x, cols):
#         if sum(x) > 1:
#             return 'multiplet'
#         elif sum(x) < 1:
#             return 'unassigned'
#         else:
#             return cols[x == 1].values[0]

#     # Use the helper function to determine cluster assignment for each data point
#     trt = [assign(x, hto_df.columns) for x in gm_assigned]

#     # Create a summary dataframe
#     df = pd.DataFrame({
#         'treatment': trt,
#         'HTO_max_prob': max_probability.max(axis=1),
#         'ratio_max_prob_to_total': max_probability.max(axis=1) / max_probability.sum(axis=1),
#         'total_HTO_counts': hto_df.sum(axis=1),
#     })

#     return df, gm_assigned, max_probability


# def assign_HTO(
#     hto_df,
#     n_components=None,
#     filter_on_prob=None,
#     per_column=False,
#     ):

#     print(f'Fitting Gaussian Mixture Model....')
#     clr = CLR(hto_df)

#     if n_components is None:
#         print(f'\t No n_components provided, using {hto_df.shape[1]}')
#         n_components = hto_df.shape[1]

#     gm = GaussianMixture(n_components=n_components, random_state=0).fit(clr.values)
#     gm_assigned = gm.predict(clr.values)

#     max_probability = gm.predict_proba(clr.values).max(axis=1)


#     ## Perform mapping by assigning the maximum CLR to each predicted class
#     mapping = {}
#     for c in set(gm_assigned):
#         mapping[c] = clr.loc[gm_assigned == c, clr.columns].mean(axis=0).idxmax()
    
#     trt = pd.Series([mapping[x] for x in gm_assigned]) # Get treatment cal
#     if filter_on_prob is not None: 
#         print(f'\t Filtering below GM prediction probability {filter_on_prob}')
#         trt[max_probability <= filter_on_prob] =  None 

#     df = pd.DataFrame({
#         'total_HTO_counts': hto_df.sum(axis=1),
#         'HTO': trt.values,
#         'HTO_max_prob': max_probability
#     }, index=hto_df.index)
#     return df
