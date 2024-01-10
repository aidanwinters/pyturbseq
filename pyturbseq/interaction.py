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

from dcor import distance_correlation, partial_distance_correlation



## Function may not be necessary but assumes that a perturbation is a single string 
def get_singles(dual, delim='|', ref='NTC'):
    """Get the single gene perturbation from the dual perturbation"""
    single = dual.split(delim)
    single_a = [single[0], ref]
    single_a.sort()
    #sort 
    single_b = [single[1], ref]
    single_b.sort()

    return delim.join(single_a), delim.join(single_b)

def get_model_fit(data, double, method = 'robust', targets=None, plot=True, verbose=True):
    """
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
    
    return out


def fit_many(data, doubles, **kwargs):
    res = pd.DataFrame([get_model_fit(data, d, **kwargs) for d in doubles])
    return res.set_index('perturbation')

def get_val(df, row_ind, col_ind):

    #check if these are in the dataframe
    if not row_ind in df.index:
        return np.nan

    if not col_ind in df.columns:
        return np.nan

    else:
        return df.loc[row_ind, col_ind]
