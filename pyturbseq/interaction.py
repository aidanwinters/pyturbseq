import statsmodels.api as sm
from sklearn.linear_model import LinearRegression, TheilSenRegressor
from scipy.stats import pearsonr, spearmanr
# from dcor import distance_correlation, partial_distance_correlation
from sklearn.metrics import r2_score
from scipy.spatial import distance
import scanpy as sc
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import concurrent.futures
from tqdm import tqdm

from dcor import distance_correlation, partial_distance_correlation



## Function may not be necessary but assumes that a perturbation is a single string 
def get_singles(dual, delim='|', ref='NTC'):
    """Get the single gene perturbation from the dual perturbation"""
    single = dual.split(delim)
    single_a = [single[0], ref]
    single_a.sort()
    #sort 
    single_b = [ref, single[1]]
    single_b.sort()

    return delim.join(single_a), delim.join(single_b)

def get_model_fit(data, double, method = 'robust', targets=None, plot=True, verbose=True):
    """
    Tom Norman's approach
    Assumes no observations but many features.
    Assumes that perturbation is index of observations (this is the case for psueodbulked to perturbation)
    Assumes data is pd DataFrame
    """

    # if data is anndata then make it df
    if type(data) == sc.AnnData:
        print("Found AnnData, densifying to df. This may take a while... ")
        data = data.to_df()


    A,B = get_singles(double)
    #confirm the overlap of replicate_col across each condition
    # perurbations = [singles[0], singles[1], double]

    #confirm all targets are in data
    if targets is None:
        targets = data.columns
    else:
        targets = [t for t in targets if t in data.columns]

    singlesX = data.loc[[A,B], targets].T
    aX = data.loc[A, targets].T
    bX = data.loc[B, targets].T
    doubleX = data.loc[double, targets].T
    
    if method == 'robust':
        regr = TheilSenRegressor(fit_intercept=False,
                    max_subpopulation=1e5,
                    max_iter=1000,
                    random_state=1000)
    else: 
        regr = LinearRegression(fit_intercept=False)

    X = singlesX
    y = doubleX

    regr.fit(X, y)

    Z = regr.predict(X)

    out = {}
    out['perturbation'] = double
    out['a'] = A
    out['b'] = B

    ##Get corrs
    # print(spearmanr(a_vals, ab_vals))
    out['corr_a'] = spearmanr(aX, doubleX)[0]
    out['corr_b'] = spearmanr(bX, doubleX)[0]
    out['corr_sum'] = spearmanr(aX + bX, doubleX)[0]
    out['corr_a_b'] = spearmanr(aX, bX)[0]

    #how well correlated is the fit to the double perturbation
    out['fit_spearmanr'] = spearmanr(Z, doubleX)[0]
    out['fit_pearsonr'] = pearsonr(Z, doubleX)[0]

    #other distance metrics
    out['fit_cosine_dist'] = distance.cosine(Z, doubleX)
    
    out['coef_a'] = regr.coef_[0]
    out['coef_b'] = regr.coef_[1]
    out['coef_ratio'] = regr.coef_[0] / regr.coef_[1]
    out['coef_difference'] = regr.coef_[0] - regr.coef_[1]
    out['coef_sum'] = regr.coef_[0] + regr.coef_[1]
    out['log2_ratio_coefs'] = np.log2(abs(regr.coef_[0]) / abs(regr.coef_[1]))
    out['log10_ratio_coefs'] = np.log10(abs(regr.coef_[0]) / abs(regr.coef_[1]))
    out['abs_log10_ratio_coefs'] = abs(np.log10(abs(regr.coef_[0]) / abs(regr.coef_[1])))

    # out['coef_abs_log_ratio'] = np.log2(abs(regr.coef_[0]/regr.coef_[1]))
    out['coef_norm'] = np.mean([np.abs(out['coef_a']), np.abs(out['coef_b'])])
    out['coef_norm2'] = np.sqrt(out['coef_a']**2 + out['coef_b']**2)
    out['score'] = regr.score(X, y)

    #get residual
    out['median_abs_residual'] = np.median(abs(doubleX - Z))
    out['rss'] = np.sum((doubleX - Z)**2)


    #Tom's metrics
    out['dcor_AnB_AB'] = distance_correlation(singlesX, doubleX) ## distance correlation between [A,B] and AB (the double perturbation)
    out['dcor_A_B'] = distance_correlation(aX, bX) ## distance correlation between A and B
    out['dcor_AnB_fit'] = distance_correlation(singlesX, Z) ## distance correlation between the [A, B] and predicted AB
    out['dcor_AB_fit'] = distance_correlation(doubleX, Z) ## distance correlation between AB and predicted AB
    out['dcor_A'] = distance_correlation(aX, doubleX) ## distance correlation between A and predicted AB
    out['dcor_B'] = distance_correlation(bX, doubleX) ## distance correlation between B and predicted AB
    out['dcor_A_fit'] = distance_correlation(aX, Z) ## distance correlation between A and predicted AB
    out['dcor_B_fit'] = distance_correlation(bX, Z) ## distance correlation between B and predicted AB
    out['min_dcor'] = min(out['dcor_A'], out['dcor_B'])
    out['max_dcor'] = max(out['dcor_A'], out['dcor_B'])
    out['dcor_ratio'] = out['min_dcor']/out['max_dcor']
    
    return out, Z


def fit_many(data, doubles, **kwargs):
    res = pd.DataFrame([get_model_fit(data, d, **kwargs)[0] for d in doubles])
    return res.set_index('perturbation')

def model_fit_wrapper(data, d, kwargs):
    return get_model_fit(data, d, **kwargs)[0]

def fit_many_parallel(data, doubles, processes=4, **kwargs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        # Use executor.map with the top-level function
        results = list(tqdm(executor.map(model_fit_wrapper, 
                                         [data]*len(doubles), 
                                         doubles, 
                                         [kwargs]*len(doubles)),
                            total=len(doubles)))

    # Convert results to DataFrame and set index
    res = pd.DataFrame(results)
    return res.set_index('perturbation')

def get_val(df, row_ind, col_ind):

    #check if these are in the dataframe
    if not row_ind in df.index:
        return np.nan

    if not col_ind in df.columns:
        return np.nan

    else:
        return df.loc[row_ind, col_ind]




##############################################################################################################
    ## Updated Function
##############################################################################################################
    
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.utils import shuffle
from scipy.stats import spearmanr
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from .utils import get_perturbation_matrix

def get_coef_permutation_plot(coefs, perm_coefs, labels, ax=None, show=True):
    """Plot the distribution of coefficients from a permutation test"""
    fig, ax = plt.subplots(1, 3, figsize=(7,2))
    for i, coef in enumerate(coefs):
        pval = np.mean(np.abs(perm_coefs[:, i]) >= np.abs(coef))
        sns.histplot(np.abs(perm_coefs[:, i]), ax=ax[i])
        ax[i].axvline(np.abs(coef), color='red', linestyle='--')
        ax[i].set_title(f"{labels[i]} - p={pval:.2f}")
        #remove xaxis and x ticks
        ax[i].set_xlabel('')
        ax[i].set_xticks([])
    fig.suptitle('Magnitude of actual coefficient to permutated distribution')
    fig.tight_layout(); 
    if show:
        plt.show()


# Parallel permutation test
def run_permutation(X, y, alpha=0.005, l1_ratio=0.5, seed=1000):
    np.random.seed(seed)  # Set the seed for reproducibility
    y_permuted = shuffle(y, random_state=seed)
    if alpha > 0:
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, random_state=seed)
    else:
        model = LinearRegression(fit_intercept=False)
    model.fit(X, y_permuted)
    return model.coef_
    

def get_model(
        adata, double, target,
        perturbation_col='perturbation', 
        reference='NTC', 
        delim='|', 
        quiet=False, 
        plot=False,
        permutation_plot=False, 
        seed=1000, 
        n_permutations=100,
        alpha=0.05,
        n_jobs=None):
    # Prepare data

    adata = adata[:, target].copy()
    if isinstance(adata.X, csr_matrix):
        adata.X = adata.X.toarray()
    ref_val_combined = reference + delim + reference
    y_ref = adata[adata.obs[perturbation_col] == ref_val_combined, :].X.flatten()
    y0 = y_ref.mean()

    singles = get_singles(double, ref=reference)
    perturbations = [singles[0], singles[1], double]
    data = adata[adata.obs[perturbation_col].isin(perturbations), :]
    single_genes = double.split(delim)
    indicators = get_perturbation_matrix(data, perturbation_col=perturbation_col, inplace=False, verbose=False)
    indicators = indicators.loc[:, single_genes]
    indicators[double] = indicators[single_genes[0]] * indicators[single_genes[1]] 

    # Elastic Net fitting
    regr = ElasticNet(
        alpha=0.005, #params from mSWI/SNF paper
        l1_ratio=0.5, #params from mSWI/SNF paper
        fit_intercept=False,
        random_state=seed)

    if alpha == 0:
        regr = LinearRegression(fit_intercept=False)

    X = indicators.values
    y = data.X.flatten() - y0
    #remove any cells that don't have a value for y
    inds = ~np.isnan(y)
    X = X[inds, :]
    y = y[inds]
    regr.fit(X, y)
    Z = regr.predict(X)

    # Collect results
    out = {'perturbation': double, 'a': singles[0], 'b': singles[1], 'target': target, 'reference': ref_val_combined,
           'coef_a': regr.coef_[0], 'coef_b': regr.coef_[1], 'coef_ab': regr.coef_[2],
           'corr_fit': spearmanr(Z.flatten(), y)[0],
           'score': regr.score(X, y)}

    if n_permutations is not None:
        # Permutation test
        original_coefs = regr.coef_
        if n_jobs is not None:
        # Parallel permutation test using joblib
            perm_coefs = Parallel(n_jobs=n_jobs)(delayed(run_permutation)(X, y, 0.005, 0.5, i) for i in range(n_permutations))
            perm_coefs = np.array(perm_coefs)
        else:
            perm_coefs = np.zeros((n_permutations, 3))
            
            for i in tqdm(range(n_permutations)) if not quiet else range(n_permutations):
                perm_coefs[i, :] = run_permutation(X, y, 0.005, 0.5, seed + i)
        # p_values = np.mean(np.abs(perm_coefs) >= np.abs(regr.coef_), axis=0)
        # modify this so min pvalue is based on # of permutaitons
        p_values = np.mean(np.abs(perm_coefs) >= np.abs(regr.coef_), axis=0) + 1/n_permutations
        out['p_value_a']  = p_values[0]
        out['p_value_b']  = p_values[1]
        out['p_value_ab'] = p_values[2]

        out['-log10_pval_ab'] = -np.log10(out['p_value_ab'] + 1/n_permutations)
        out['signif_interaction'] = out['p_value_ab'] < alpha

        #permutation coef plot
        if permutation_plot and not quiet:
            get_coef_permutation_plot(original_coefs, perm_coefs, labels=perturbations, show=False)

    #add mean reference value (ie y0)
    out['ref_mean'] = float(y0)
    out['ref_median'] = float(np.median(y_ref))

    for val, pert in zip(['a', 'b', 'ab'], perturbations):
        inds = data.obs[perturbation_col] == pert
        out[f'n_cells_{val}'] = np.sum(inds)
        out[f'mean_{val}'] = float(np.mean(y[inds]))
        out[f'predicted_mean_{val}'] = float(np.mean(Z[inds]))
        out[f'median_{val}'] = float(np.median(y[inds]))
        out[f'predicted_median_{val}'] = float(np.median(Z[inds]))
        out[f'corr_{val}'] = spearmanr(y[inds], Z[inds])[0]

    #add predictions for ab group when not using the ab coef
    inds = data.obs[perturbation_col] == double
    ab_indicators = indicators.loc[inds, single_genes]
    pred_ab_no_interaction_term = out['coef_a'] * ab_indicators[single_genes[0]] + out['coef_b'] * ab_indicators[single_genes[1]]
    out['predicted_mean_ab_no_interaction_term'] = float(np.mean(pred_ab_no_interaction_term))
    out['predicted_median_ab_no_interaction_term'] = float(np.median(pred_ab_no_interaction_term))
    out['corr_ab_no_interaction_term'] = spearmanr(y[inds], pred_ab_no_interaction_term)[0]

    ##other additional metrics
    out['abs_coef_ab'] = abs(out['coef_ab'])
    out['abs_coef_a'] = abs(out['coef_a'])
    out['abs_coef_b'] = abs(out['coef_b'])
    
    out['direction_interaction'] = np.sign(out['coef_ab'])
    out['direction_interaction_wA'] = np.sign(out['coef_ab'] * out['coef_a']) #negative if there is disagreemtn
    out['direction_interaction_wB'] = np.sign(out['coef_ab'] * out['coef_b']) #negative if there is disagreemt
    out['direction_interaction_wA_wB'] = np.sign(out['coef_ab'] * out['coef_a'] * out['coef_b']) #negative if there is any disagreemt
    out['relative_magnitude_interaction'] = out['abs_coef_ab'] / (out['abs_coef_a'] + out['abs_coef_b'])

    out['predicted_mean_ab_no_interaction'] = out['coef_a'] + out['coef_b']
    out['predicted_sign_ab_no_interaction'] = np.sign(out['predicted_mean_ab_no_interaction'])
    out['interaction_effect'] = out['predicted_sign_ab_no_interaction'] * out['direction_interaction']

    # Plotting (if required)
    if plot and not quiet:
        fig, ax = plt.subplots(1, 2, figsize=(10,5))
        sns.boxplot(x=data.obs[perturbation_col], y=y, ax=ax[0], order=perturbations)
        ax[0].set_title(f"Target: {target}")
        ax[0].set_ylabel("Expression (difference from reference)")
        sns.scatterplot(y=y, x=Z, hue=data.obs[perturbation_col], hue_order = perturbations, ax=ax[1], alpha=0.7)
        ax[1].set_ylabel('Actual'); ax[1].set_xlabel('Fit')
        model_string = f"{round(out['coef_a'],2)}{out['a']} + {round(out['coef_b'],2)}{out['b']} + {round(out['coef_ab'],2)}{out['perturbation']}"
        ax[1].set_title(f"R2: {out['score']:.2f}\n{model_string}")
        fig.tight_layout(); plt.show()

    return out

##############################################################################################################
import statsmodels.api as sm
import statsmodels.formula.api as smf


def get_model_statsmodels(
        double, target,
        adata = None,
        perturbation_col='perturbation', 
        reference='NTC', 
        delim='|', 
        quiet=False, 
        plot=False,
        ):
    
        singles = get_singles(double, ref=reference)
        perturbations = [singles[0], singles[1], double]

        if adata is not None:
                adata = adata[:, target]
                ref_val_combined = reference + delim + reference
                y_ref = adata[adata.obs[perturbation_col] == ref_val_combined, :].X.toarray().flatten()
                y0 = y_ref.mean()

                data = adata[adata.obs[perturbation_col].isin(perturbations), :]
                single_genes = double.split(delim)
                indicators = get_perturbation_matrix(data, perturbation_col=perturbation_col, inplace=False, verbose=False)
                indicators = indicators.loc[:, single_genes]
                # indicators[double] = indicators[single_genes[0]] * indicators[single_genes[1]] 

                df = pd.DataFrame(indicators, columns=single_genes)
                y = data.X.toarray().flatten() - y0
        
                df['y'] = list(y)

                out = {
                        'perturbation': double, 'a': singles[0], 'b': singles[1], 'reference': ref_val_combined,
                        'target': target, 'n_cells': indicators.shape[0],
                        'ref_mean': float(y0),
                        'ref_median': float(np.median(y_ref)),
                        'n_cells': indicators.shape[0]
                }
        else:
                raise ValueError("Must provide adata")



        formula = f'y ~ {single_genes[0]} + {single_genes[1]} + {single_genes[0]}:{single_genes[1]} -1'
        regr = smf.rlm(formula, data=df)
        regr_fit = regr.fit()
        Z  = regr_fit.predict(df)

        for i, val in enumerate(['a', 'b', 'ab']):
                out[f'pval_{val}'] = regr_fit.pvalues[i]
                out[f'tstat_{val}'] = regr_fit.tvalues[i]
                out[f'std_err_{val}'] = regr_fit.bse[i]
                out[f'coef_{val}'] = regr_fit.params[i]

        # out['score'] = regr_fit.rsquared
        out['corr_fit'] = spearmanr(Z.values, y)[0]


        for val, pert in zip(['a', 'b', 'ab'], perturbations):
                inds = data.obs[perturbation_col] == pert
                out[f'n_cells_{val}'] = np.sum(inds)
                out[f'mean_{val}'] = float(np.mean(y[inds]))
                out[f'predicted_mean_{val}'] = float(np.mean(Z[inds]))
                out[f'median_{val}'] = float(np.median(y[inds]))
                out[f'predicted_median_{val}'] = float(np.median(Z[inds]))

        #add predictions for ab group when not using the ab coef
        inds = data.obs[perturbation_col] == double
        ab_indicators = indicators.loc[inds, single_genes]
        pred_ab_no_interaction_term = out['coef_a'] * ab_indicators[single_genes[0]] + out['coef_b'] * ab_indicators[single_genes[1]]
        out['predicted_mean_ab_no_interaction_term'] = float(np.mean(pred_ab_no_interaction_term))
        out['predicted_median_ab_no_interaction_term'] = float(np.median(pred_ab_no_interaction_term))
        out['corr_ab_no_interaction_term'] = spearmanr(y[inds], pred_ab_no_interaction_term)[0]

        ##other additional metrics
        out['abs_coef_ab'] = abs(out['coef_ab'])
        out['abs_coef_a'] = abs(out['coef_a'])
        out['abs_coef_b'] = abs(out['coef_b'])
        
        out['direction_interaction'] = np.sign(out['coef_ab'])
        out['direction_interaction_wA'] = np.sign(out['coef_ab'] * out['coef_a']) #negative if there is disagreemtn
        out['direction_interaction_wB'] = np.sign(out['coef_ab'] * out['coef_b']) #negative if there is disagreemt
        out['direction_interaction_wA_wB'] = np.sign(out['coef_ab'] * out['coef_a'] * out['coef_b']) #negative if there is any disagreemt
        out['relative_magnitude_interaction'] = out['abs_coef_ab'] / (out['abs_coef_a'] + out['abs_coef_b'])

        out['predicted_mean_ab_no_interaction'] = out['coef_a'] + out['coef_b']
        out['predicted_sign_ab_no_interaction'] = np.sign(out['predicted_mean_ab_no_interaction'])
        out['interaction_effect'] = out['predicted_sign_ab_no_interaction'] * out['direction_interaction']


        if plot and not quiet:
                fig, ax = plt.subplots(1, 2, figsize=(10,5))
                sns.boxplot(x=data.obs[perturbation_col], y=y, ax=ax[0], order=perturbations)
                ax[0].set_title(f"Target: {target}")
                ax[0].set_ylabel("Expression (difference from reference)")
                sns.scatterplot(y=y, x=Z, hue=data.obs[perturbation_col], hue_order = perturbations, ax=ax[1], alpha=0.7)
                ax[1].set_ylabel('Actual'); ax[1].set_xlabel('Fit')
                model_string = f"{round(out['coef_a'],2)}{out['a']} + {round(out['coef_b'],2)}{out['b']} + {round(out['coef_ab'],2)}{out['perturbation']}"
                ax[1].set_title(f"Spearman: {out['corr_fit']:.2f} - pval AB {round(out['pval_ab'], 3)}\n{model_string}")
                fig.tight_layout(); plt.show()

        return out