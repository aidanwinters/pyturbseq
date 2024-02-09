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
def get_singles(dual, delim="|", ref="NTC"):
    """Get the single gene perturbation from the dual perturbation"""
    single = dual.split(delim)
    single_a = [single[0], ref]
    single_a.sort()
    # sort
    single_b = [ref, single[1]]
    single_b.sort()

    return delim.join(single_a), delim.join(single_b)


def get_model_fit(data, double, method="robust", targets=None, plot=True, verbose=True):
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

    A, B = get_singles(double)
    # confirm the overlap of replicate_col across each condition
    # perurbations = [singles[0], singles[1], double]

    # confirm all targets are in data
    if targets is None:
        targets = data.columns
    else:
        targets = [t for t in targets if t in data.columns]

    singlesX = data.loc[[A, B], targets].T
    aX = data.loc[A, targets].T
    bX = data.loc[B, targets].T
    doubleX = data.loc[double, targets].T

    if method == "robust":
        regr = TheilSenRegressor(
            fit_intercept=False, max_subpopulation=1e5, max_iter=1000, random_state=1000
        )
    else:
        regr = LinearRegression(fit_intercept=False)

    X = singlesX
    y = doubleX

    regr.fit(X, y)

    Z = regr.predict(X)

    out = {}
    out["perturbation"] = double
    out["a"] = A
    out["b"] = B

    ##Get corrs
    # print(spearmanr(a_vals, ab_vals))
    out["corr_a"] = spearmanr(aX, doubleX)[0]
    out["corr_b"] = spearmanr(bX, doubleX)[0]
    out["corr_sum"] = spearmanr(aX + bX, doubleX)[0]
    out["corr_a_b"] = spearmanr(aX, bX)[0]

    # how well correlated is the fit to the double perturbation
    out["fit_spearmanr"] = spearmanr(Z, doubleX)[0]
    out["fit_pearsonr"] = pearsonr(Z, doubleX)[0]

    # other distance metrics
    out["fit_cosine_dist"] = distance.cosine(Z, doubleX)

    out["coef_a"] = regr.coef_[0]
    out["coef_b"] = regr.coef_[1]
    out["coef_ratio"] = regr.coef_[0] / regr.coef_[1]
    out["coef_difference"] = regr.coef_[0] - regr.coef_[1]
    out["coef_sum"] = regr.coef_[0] + regr.coef_[1]
    out["log2_ratio_coefs"] = np.log2(abs(regr.coef_[0]) / abs(regr.coef_[1]))
    out["log10_ratio_coefs"] = np.log10(abs(regr.coef_[0]) / abs(regr.coef_[1]))
    out["abs_log10_ratio_coefs"] = abs(
        np.log10(abs(regr.coef_[0]) / abs(regr.coef_[1]))
    )

    # out['coef_abs_log_ratio'] = np.log2(abs(regr.coef_[0]/regr.coef_[1]))
    out["coef_norm"] = np.mean([np.abs(out["coef_a"]), np.abs(out["coef_b"])])
    out["coef_norm2"] = np.sqrt(out["coef_a"] ** 2 + out["coef_b"] ** 2)
    out["score"] = regr.score(X, y)

    # get residual
    out["median_abs_residual"] = np.median(abs(doubleX - Z))
    out["rss"] = np.sum((doubleX - Z) ** 2)

    # Tom's metrics
    out["dcor_AnB_AB"] = distance_correlation(
        singlesX, doubleX
    )  ## distance correlation between [A,B] and AB (the double perturbation)
    out["dcor_A_B"] = distance_correlation(
        aX, bX
    )  ## distance correlation between A and B
    out["dcor_AnB_fit"] = distance_correlation(
        singlesX, Z
    )  ## distance correlation between the [A, B] and predicted AB
    out["dcor_AB_fit"] = distance_correlation(
        doubleX, Z
    )  ## distance correlation between AB and predicted AB
    out["dcor_A"] = distance_correlation(
        aX, doubleX
    )  ## distance correlation between A and predicted AB
    out["dcor_B"] = distance_correlation(
        bX, doubleX
    )  ## distance correlation between B and predicted AB
    out["dcor_A_fit"] = distance_correlation(
        aX, Z
    )  ## distance correlation between A and predicted AB
    out["dcor_B_fit"] = distance_correlation(
        bX, Z
    )  ## distance correlation between B and predicted AB
    out["min_dcor"] = min(out["dcor_A"], out["dcor_B"])
    out["max_dcor"] = max(out["dcor_A"], out["dcor_B"])
    out["dcor_ratio"] = out["min_dcor"] / out["max_dcor"]

    return out, Z


def fit_many(data, doubles, **kwargs):
    res = pd.DataFrame([get_model_fit(data, d, **kwargs)[0] for d in doubles])
    return res.set_index("perturbation")


def model_fit_wrapper(data, d, kwargs):
    return get_model_fit(data, d, **kwargs)[0]


def fit_many_parallel(data, doubles, processes=4, **kwargs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        # Use executor.map with the top-level function
        results = list(
            tqdm(
                executor.map(
                    model_fit_wrapper,
                    [data] * len(doubles),
                    doubles,
                    [kwargs] * len(doubles),
                ),
                total=len(doubles),
            )
        )

    # Convert results to DataFrame and set index
    res = pd.DataFrame(results)
    return res.set_index("perturbation")


def get_val(df, row_ind, col_ind):

    # check if these are in the dataframe
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
    fig, ax = plt.subplots(1, 3, figsize=(7, 2))
    for i, coef in enumerate(coefs):
        pval = np.mean(np.abs(perm_coefs[:, i]) >= np.abs(coef))
        sns.histplot(np.abs(perm_coefs[:, i]), ax=ax[i])
        ax[i].axvline(np.abs(coef), color="red", linestyle="--")
        ax[i].set_title(f"{labels[i]} - p={pval:.2f}")
        # remove xaxis and x ticks
        ax[i].set_xlabel("")
        ax[i].set_xticks([])
    fig.suptitle("Magnitude of actual coefficient to permutated distribution")
    fig.tight_layout()
    if show:
        plt.show()


# Parallel permutation test
def run_permutation(X, y, alpha=0.005, l1_ratio=0.5, seed=1000):
    np.random.seed(seed)  # Set the seed for reproducibility
    y_permuted = shuffle(y, random_state=seed)
    if alpha > 0:
        model = ElasticNet(
            alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, random_state=seed
        )
    else:
        model = LinearRegression(fit_intercept=False)
    model.fit(X, y_permuted)
    return model.coef_


def get_model(
    adata,
    double,
    target,
    perturbation_col="perturbation",
    reference="NTC",
    delim="|",
    quiet=False,
    plot=False,
    permutation_plot=False,
    seed=1000,
    n_permutations=100,
    alpha=0.05,
    n_jobs=None,
):
    """

    Implementation of 2 perturbation model similar to mSW/SNF paper from Cigall's group (REF needed).
    Basic overview:
    1. A + B + AB = y
    2. Elastic net
    3. Permutation test for coefficient significance
    """
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
    indicators = get_perturbation_matrix(
        data, perturbation_col=perturbation_col, inplace=False, verbose=False
    )
    indicators = indicators.loc[:, single_genes]
    indicators[double] = indicators[single_genes[0]] * indicators[single_genes[1]]

    # Elastic Net fitting
    regr = ElasticNet(
        alpha=0.005,  # params from mSWI/SNF paper
        l1_ratio=0.5,  # params from mSWI/SNF paper
        fit_intercept=False,
        random_state=seed,
    )

    if alpha == 0:
        regr = LinearRegression(fit_intercept=False)

    X = indicators.values
    y = data.X.flatten() - y0
    # remove any cells that don't have a value for y
    inds = ~np.isnan(y)
    X = X[inds, :]
    y = y[inds]
    regr.fit(X, y)
    Z = regr.predict(X)

    # Collect results
    out = {
        "perturbation": double,
        "a": singles[0],
        "b": singles[1],
        "target": target,
        "reference": ref_val_combined,
        "coef_a": regr.coef_[0],
        "coef_b": regr.coef_[1],
        "coef_ab": regr.coef_[2],
        "corr_fit": spearmanr(Z.flatten(), y)[0],
        "score": regr.score(X, y),
    }

    if n_permutations is not None:
        # Permutation test
        original_coefs = regr.coef_
        if n_jobs is not None:
            # Parallel permutation test using joblib
            perm_coefs = Parallel(n_jobs=n_jobs)(
                delayed(run_permutation)(X, y, 0.005, 0.5, i)
                for i in range(n_permutations)
            )
            perm_coefs = np.array(perm_coefs)
        else:
            perm_coefs = np.zeros((n_permutations, 3))

            for i in (
                tqdm(range(n_permutations)) if not quiet else range(n_permutations)
            ):
                perm_coefs[i, :] = run_permutation(X, y, 0.005, 0.5, seed + i)
        # p_values = np.mean(np.abs(perm_coefs) >= np.abs(regr.coef_), axis=0)
        # modify this so min pvalue is based on # of permutaitons
        p_values = (
            np.mean(np.abs(perm_coefs) >= np.abs(regr.coef_), axis=0)
            + 1 / n_permutations
        )
        out["p_value_a"] = p_values[0]
        out["p_value_b"] = p_values[1]
        out["p_value_ab"] = p_values[2]

        out["-log10_pval_ab"] = -np.log10(out["p_value_ab"] + 1 / n_permutations)
        out["signif_interaction"] = out["p_value_ab"] < alpha

        # permutation coef plot
        if permutation_plot and not quiet:
            get_coef_permutation_plot(
                original_coefs, perm_coefs, labels=perturbations, show=False
            )

    # add mean reference value (ie y0)
    out["ref_mean"] = float(y0)
    out["ref_median"] = float(np.median(y_ref))

    for val, pert in zip(["a", "b", "ab"], perturbations):
        inds = data.obs[perturbation_col] == pert
        out[f"n_cells_{val}"] = np.sum(inds)
        out[f"mean_{val}"] = float(np.mean(y[inds]))
        out[f"predicted_mean_{val}"] = float(np.mean(Z[inds]))
        out[f"median_{val}"] = float(np.median(y[inds]))
        out[f"predicted_median_{val}"] = float(np.median(Z[inds]))
        out[f"corr_{val}"] = spearmanr(y[inds], Z[inds])[0]

    # add predictions for ab group when not using the ab coef
    inds = data.obs[perturbation_col] == double
    ab_indicators = indicators.loc[inds, single_genes]
    pred_ab_no_interaction_term = (
        out["coef_a"] * ab_indicators[single_genes[0]]
        + out["coef_b"] * ab_indicators[single_genes[1]]
    )
    out["predicted_mean_ab_no_interaction_term"] = float(
        np.mean(pred_ab_no_interaction_term)
    )
    out["predicted_median_ab_no_interaction_term"] = float(
        np.median(pred_ab_no_interaction_term)
    )
    out["corr_ab_no_interaction_term"] = spearmanr(
        y[inds], pred_ab_no_interaction_term
    )[0]

    ##other additional metrics
    out["abs_coef_ab"] = abs(out["coef_ab"])
    out["abs_coef_a"] = abs(out["coef_a"])
    out["abs_coef_b"] = abs(out["coef_b"])

    out["direction_interaction"] = np.sign(out["coef_ab"])
    out["direction_interaction_wA"] = np.sign(
        out["coef_ab"] * out["coef_a"]
    )  # negative if there is disagreemtn
    out["direction_interaction_wB"] = np.sign(
        out["coef_ab"] * out["coef_b"]
    )  # negative if there is disagreemt
    out["direction_interaction_wA_wB"] = np.sign(
        out["coef_ab"] * out["coef_a"] * out["coef_b"]
    )  # negative if there is any disagreemt
    out["relative_magnitude_interaction"] = out["abs_coef_ab"] / (
        out["abs_coef_a"] + out["abs_coef_b"]
    )

    out["predicted_mean_ab_no_interaction"] = out["coef_a"] + out["coef_b"]
    out["predicted_sign_ab_no_interaction"] = np.sign(
        out["predicted_mean_ab_no_interaction"]
    )
    out["interaction_effect"] = (
        out["predicted_sign_ab_no_interaction"] * out["direction_interaction"]
    )

    # Plotting (if required)
    if plot and not quiet:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        sns.boxplot(x=data.obs[perturbation_col], y=y, ax=ax[0], order=perturbations)
        ax[0].set_title(f"Target: {target}")
        ax[0].set_ylabel("Expression (difference from reference)")
        sns.scatterplot(
            y=y,
            x=Z,
            hue=data.obs[perturbation_col],
            hue_order=perturbations,
            ax=ax[1],
            alpha=0.7,
        )
        ax[1].set_ylabel("Actual")
        ax[1].set_xlabel("Fit")
        model_string = f"{round(out['coef_a'],2)}{out['a']} + {round(out['coef_b'],2)}{out['b']} + {round(out['coef_ab'],2)}{out['perturbation']}"
        ax[1].set_title(f"R2: {out['score']:.2f}\n{model_string}")
        fig.tight_layout()
        plt.show()

    return out


##############################################################################################################
## Updated Function for interaction modeling FOR 2 PERTURBATIONS that includes reference cells
# this builds on the statsmodels implementation which added the following:
# - estimation of coefficient significance and p-values via wald test (ie the background test done by statsmodels)
# - use of robust regression (ie TheilSenRegressor) to avoid outliers
# This implementation simply add the reference cells and removes the need to subtract reference mean from the values
# Rationale for this: it seems that with the reference mean subtraction, there was an inflation of pvals
#   for coefficients (including the interaciton term) with lowly expressed genes that had a consistent decrease compared to reference


def estimate_alpha_empirical(y):
    mean_y = np.mean(y)
    var_y = np.var(y, ddof=1)
    if mean_y <= var_y:  # Check for overdispersion
        try:
            dispersion_factor = (var_y - mean_y) / mean_y**2
            alpha = 1 / dispersion_factor
        except ZeroDivisionError:
            raise ValueError("Cannot estimate alpha when mean_y is 0")
    else:
        alpha = None  # assume poisson estimate
    return alpha, mean_y, var_y


def breakdown_double_wRef(input_str, ref="control", delim="|"):
    split = input_str.split(delim)
    # confirm third pos is in split and if so make sure at the end
    split = input_str.split(delim)
    # confirm third pos is in split and if so make sure at the end
    term2pert = {}
    term2pert["ref"] = delim.join([ref] * 2)
    term2pert["a"] = split[0] + delim + ref
    term2pert["b"] = split[1] + delim + ref
    term2pert["ab"] = split[0] + delim + split[1]

    # sort each of them:
    for key, val in term2pert.items():
        term2pert[key] = delim.join(np.sort(val.split(delim)))

    split = [ref] + split

    # invert term2pert
    pert2term = {v: k for k, v in term2pert.items()}

    return term2pert, pert2term, split


def breakdown_triple_wRef(input_str, third_pos="css", ref="control", delim="|"):
    split = input_str.split(delim)
    # confirm third pos is in split and if so make sure at the end
    if third_pos in split:
        split.remove(third_pos)
        split.append(third_pos)
    else:
        raise ValueError(f"third_pos {third_pos} not in split {split}")
    term2pert = {}
    term2pert["ref"] = delim.join([ref] * 3)
    term2pert["a"] = split[0] + delim + ref + delim + ref
    term2pert["b"] = split[1] + delim + ref + delim + ref
    term2pert["c"] = split[2] + delim + ref + delim + ref
    term2pert["ab"] = split[0] + delim + split[1] + delim + ref
    term2pert["ac"] = split[0] + delim + split[2] + delim + ref
    term2pert["bc"] = split[1] + delim + split[2] + delim + ref
    term2pert["abc"] = input_str

    # sort each of them:
    for key, val in term2pert.items():
        term2pert[key] = delim.join(np.sort(val.split(delim)))

        # invert term2pert
    pert2term = {v: k for k, v in term2pert.items()}

    split = [ref] + split
    return term2pert, pert2term, split


def breakdown_perturbation(combo_perturbation, num_perturbations, **kwargs):
    if num_perturbations == 2:
        return breakdown_double_wRef(combo_perturbation, **kwargs)
    elif num_perturbations == 3:
        return breakdown_triple_wRef(combo_perturbation, **kwargs)
    else:
        raise ValueError("Only 2 or 3 perturbations are supported")


# def add_metrics_triple(out, indicators, y, pert2term, term2pert, s):


def get_model_wNTC(
    combo_perturbation,
    target,
    adata=None,
    perturbation_col="perturbation",
    reference="NTC",
    delim="|",
    num_perturbations=None,
    method="robust",
    eps=1e-250,
    quiet=False,
    plot=False,
):
    """
    Model for 2 perturbations using statsmodels pacakge with addition of reference populations
    Basic Overiew:
    1. *Indicator model: ref + A + B + AB = y
    2. Robust linear regression
    3. Get p-values for coefficients via statsmodels package (ie wald test)

    *main difference from implementation with statsmodels is that we add reference indicators to the model and fit to A,B,AB, and ref cells
    """

    vp = print if not quiet else lambda *a, **k: None

    vp("Running interaction model.")
    vp(f"\tCombined perturbation: {combo_perturbation}")

    if adata is None:
        raise ValueError("Must provide adata")

    if num_perturbations is None:
        # only allow2 or 3 perturbations for now
        num_perturbations = len(combo_perturbation.split(delim))

    if num_perturbations not in [2, 3]:
        raise ValueError("Only 2 or 3 perturbations are supported")
    else:
        vp(f"\tFound {num_perturbations} perturbations...")

    ### SETUP perturbation input and target data
    # Prepare data
    term2pert, pert2term, s = breakdown_perturbation(
        combo_perturbation,
        num_perturbations=num_perturbations,
        delim=delim,
        ref=reference,
    )
    perturbations = list(term2pert.values())
    vp("\tModel terms:")
    if not quiet:
        for k, v in term2pert.items():
            print(f"\t\t{k}: {v}")

    # ensure that all perturbations are in adata

    perturb_set = adata.obs[perturbation_col].unique()
    if not all([x in perturb_set for x in perturbations]):
        perturbations = pd.Series(perturbations)
        raise ValueError(
            f"Missing perturbations: {perturbations[~perturbations.isin(perturb_set)].values}"
        )

    data = adata[adata.obs[perturbation_col].isin(perturbations), target]

    # get indicators and convert to integer
    indicators = get_perturbation_matrix(
        data,
        perturbation_col=perturbation_col,
        reference_value=reference,
        inplace=False,
        keep_ref=True,
        set_ref_1=True,
        verbose=not quiet,
    ).astype(int)

    # ensure they are in the expected order
    indicators = indicators.loc[:, s]
    y = data.X.toarray().flatten()  # get the target gene data
    indicators["y"] = list(y)  # add the target gene data to the indicators

    # adding some metadata: original perturbation label for the cell, corresponding term for the cell
    # this meta data won't effect the statsmodel
    indicators[perturbation_col] = data.obs[perturbation_col]
    indicators["term"] = [pert2term[x] for x in indicators[perturbation_col]]

    out = {
        "perturbation": combo_perturbation,
        "reference": term2pert["ref"],
        "target": target,
        "n_cells": indicators.shape[0],
        "mean": float(np.mean(y)),
        "var": float(np.var(y, ddof=1)),
    }
    out.update(
        term2pert
    )  # update with the term to perturbation mapping (ie a == 'A|control' etc)

    # define formula with reference and with removing the intercept
    if num_perturbations == 2:
        formula = f"y ~ {s[0]} + {s[1]} + {s[2]} + {s[1]}:{s[2]} -1"
    elif num_perturbations == 3:
        formula = f"y ~ {s[0]} + {s[1]} + {s[2]} + {s[3]} + {s[1]}:{s[2]} + {s[1]}:{s[3]} + {s[2]}:{s[3]} + {s[1]}:{s[2]}:{s[3]} -1"

    ################################################################################
    ### SETUP MODEL AND FIT
    ################################################################################

    # define model type based on input method
    if method == "robust":
        out["method"] = "robust"
        vp("\tUsing robust regression")
        regr = smf.rlm(formula, data=indicators)
    elif method == "ols":
        out["method"] = "ols"
        vp("\tUsing OLS regression")
        regr = smf.ols(formula, data=indicators)
    elif method in ["negbin", "nb", "negativebinomial"]:
        vp("\tUsing Negative Binomial regression")
        y_ref = indicators.loc[indicators["term"] == "ref", "y"].values
        alpha_estimated, mean_ref, var_ref = estimate_alpha_empirical(y_ref)
        if alpha_estimated is not None:
            # Using GLM with a Negative Binomial family
            vp(
                f"Data looks overdispersed. Using negative binomial GLM with alpha: {alpha_estimated}"
            )
            out["method"] = "negativebinomial"
            regr = smf.glm(
                formula,
                data=indicators,
                family=sm.families.NegativeBinomial(alpha=alpha_estimated),
            )
        else:
            vp("Data not overdispersed. Using poisson GLM")
            out["method"] = "poisson"
            # use a poison regression
            regr = smf.glm(formula, data=indicators, family=sm.families.Poisson())

    ##fit
    vp("\tFitting model...")
    regr_fit = regr.fit()
    Z = regr_fit.predict(indicators)
    indicators["predicted"] = Z

    for i, term in enumerate(term2pert.keys()):
        out[f"pval_{term}"] = regr_fit.pvalues[i] + eps
        out[f"tstat_{term}"] = regr_fit.tvalues[i]
        out[f"std_err_{term}"] = regr_fit.bse[i]
        out[f"coef_{term}"] = regr_fit.params[i]
        out[f"abs_coef_{term}"] = abs(regr_fit.params[i])

    out["corr_fit"] = spearmanr(Z.values, y)[0]

    for term, pert in term2pert.items():
        inds = data.obs[perturbation_col] == pert
        out[f"n_cells_{term}"] = np.sum(inds)
        out[f"mean_{term}"] = float(np.mean(y[inds]))
        out[f"var_{term}"] = float(np.var(y[inds], ddof=1))
        out[f"predicted_mean_{term}"] = float(np.mean(Z[inds]))
        out[f"median_{term}"] = float(np.median(y[inds]))
        out[f"predicted_median_{term}"] = float(np.median(Z[inds]))

    # if the method is robust or ols, we can just use the coefficients to calculate directly on y given there is no link function
    if method in ["robust", "ols"]:
        vp("\tGetting prediction without interaction effect...")
        indicators_only = indicators.loc[:, s]
        # get prediction without using the last indicator (ie the combo perturbation) or last coefficient (ie the interaction term)
        Z_noInteraction = indicators_only.values @ regr_fit.params[:-1]
        # multiply the indicators without the last column
        combo_term = pert2term[combo_perturbation]
        combo_inds = data.obs[perturbation_col] == combo_perturbation
        out[f"predicted_mean_no_interaction_{combo_term}"] = float(
            np.mean(Z_noInteraction[combo_inds])
        )
        out[f"predicted_median_no_interaction_{combo_term}"] = float(
            np.median(Z_noInteraction[combo_inds])
        )

        if num_perturbations == 2:
            # for now add a manual
            Z_noInteraction_AW = (
                (indicators[s[0]] * out["coef_ref"])
                + (indicators[s[1]] * out["coef_a"])
                + (indicators[s[2]] * out["coef_b"])
            )
            Z_AW = (
                (indicators[s[0]] * out["coef_ref"])
                + (indicators[s[1]] * out["coef_a"])
                + (indicators[s[2]] * out["coef_b"])
                + (indicators[s[1]] * indicators[s[2]] * out["coef_ab"])
            )
            out["predicted_mean_no_interaction_AW"] = float(
                np.mean(Z_noInteraction_AW[combo_inds])
            )
            out["predict_mean_AW"] = float(np.mean(Z_AW[combo_inds]))

        if plot:
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))

            sns.boxplot(
                x=data.obs[perturbation_col], y=y, ax=ax[0], order=perturbations
            )
            ax[0].set_title(f"Target: {target}")
            ax[0].set_ylabel("Expression")
            ax[0].set_xlabel(None)
            ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha="right")

            # set color of each tick to its corresponding hue
            for i, xtick in enumerate(ax[0].get_xticklabels()):
                xtick.set_color(sns.color_palette()[i])
            # add line with text for only single and no interaction
            ax[0].axhline(
                out[f"predicted_mean_no_interaction_{combo_term}"],
                color="red",
                linestyle="--",
                alpha=0.3,
            )
            # ax[0].text(0, out['predicted_mean_abc_only_single'], "only singles", color='red', alpha=0.3)
            # ax[0].text(0, out['predicted_mean_abc_no_interaction'], "all but interaction", color='blue', alpha=0.3)
            ax[0].text(
                0.5,
                0.5,
                "prediction w/o interaction",
                color="red",
                alpha=0.3,
                transform=ax[0].transAxes,
            )

            sns.scatterplot(
                y=-np.log10(regr_fit.pvalues + eps),
                x=regr_fit.params,
                hue=term2pert.values(),
                hue_order=perturbations,
                ax=ax[1],
                alpha=0.7,
            )
            # set legend outside
            ax[1].axhline(-np.log10(0.05), color="black", linestyle="--", alpha=0.3)
            ax[1].set_title("Coefficients")
            ax[1].set_xlabel("Coefficient value")
            ax[1].set_ylabel("-log10(p-value)")
            # remove legend
            ax[1].get_legend().remove()

            fig.tight_layout()
            plt.show()

    ## end the shared 2 or 3 perturbation coding

    # for 2 perturbation, we want to add the no interaction term prediction

    # for 3 perturbations, we want to add no triple interaction, and no digenic term interaction predictions

    # add predictions for ab group when not using the ab coef
    # inds = data.obs[perturbation_col] == double
    # ab_indicators = indicators.loc[inds, single_genes]
    # pred_ab_no_interaction_term = (
    #     out["coef_a"] * ab_indicators[single_genes[0]]
    #     + out["coef_b"] * ab_indicators[single_genes[1]]
    # )
    # out["predicted_mean_ab_no_interaction_term"] = float(
    #     np.mean(pred_ab_no_interaction_term)
    # )
    # out["predicted_median_ab_no_interaction_term"] = float(
    #     np.median(pred_ab_no_interaction_term)
    # )
    # out["corr_ab_no_interaction_term"] = spearmanr(
    #     y[inds], pred_ab_no_interaction_term
    # )[0]

    # out["direction_interaction"] = np.sign(out["coef_ab"])
    # out["direction_interaction_wA"] = np.sign(
    #     out["coef_ab"] * out["coef_a"]
    # )  # negative if there is disagreemtn
    # out["direction_interaction_wB"] = np.sign(
    #     out["coef_ab"] * out["coef_b"]
    # )  # negative if there is disagreemt
    # out["direction_interaction_wA_wB"] = np.sign(
    #     out["coef_ab"] * out["coef_a"] * out["coef_b"]
    # )  # negative if there is any disagreemt
    # out["relative_magnitude_interaction"] = out["abs_coef_ab"] / (
    #     out["abs_coef_a"] + out["abs_coef_b"]
    # )

    # out["predicted_mean_ab_no_interaction"] = out["coef_a"] + out["coef_b"]
    # out["predicted_sign_ab_no_interaction"] = np.sign(
    #     out["predicted_mean_ab_no_interaction"]
    # )
    # out["interaction_effect"] = (
    #     out["predicted_sign_ab_no_interaction"] * out["direction_interaction"]
    # )

    # if plot and not quiet:
    #     fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    #     sns.boxplot(x=data.obs[perturbation_col], y=y, ax=ax[0], order=perturbations)
    #     ax[0].set_title(f"Target: {target}")
    #     ax[0].set_ylabel("Expression (difference from reference)")
    #     sns.scatterplot(
    #         y=y,
    #         x=Z,
    #         hue=data.obs[perturbation_col],
    #         hue_order=perturbations,
    #         ax=ax[1],
    #         alpha=0.7,
    #     )
    #     ax[1].set_ylabel("Actual")
    #     ax[1].set_xlabel("Fit")
    #     model_string = f"{round(out['coef_a'],2)}{out['a']} + {round(out['coef_b'],2)}{out['b']} + {round(out['coef_ab'],2)}{out['perturbation']}"
    #     ax[1].set_title(
    #         f"Spearman: {out['corr_fit']:.2f} - pval AB {round(out['pval_ab'], 3)}\n{model_string}"
    #     )
    #     fig.tight_layout()
    #     plt.show()

    return out


##############################################################################################################
## Updated Modeling function including reference cells and Negative Binomial/Poisson assumption

# import numpy as np
# import pandas as pd
# from scipy.stats import spearmanr, pearsonr
# from scipy.sparse import csr_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from tqdm import tqdm
# from joblib import Parallel, delayed
# import scanpy as sc
# from pyturbseq.utils import get_perturbation_matrix

# import statsmodels.formula.api as smf
# import statsmodels.api as sm


# def get_treatment_effect_model(
#     combo_perturbation,
#     target,
#     adata,
#     perturbation_col="perturbation_treatment",
#     reference="control",
#     delim="|",
#     coef_c="css",
#     eps=1e-250,
#     method="negativebinomial",
#     quiet=False,
#     plot=False,
# ):
#     """
#     Model for interaction between 3 perturbations using statsmodels pacakge.
#     Basic Overiew:
#     1. Indicator model: ref + A + B + C + AB + AC + BC + ABC = y
#     2. Robust linear regression
#     3. Get p-values for coefficients via statsmodels package (ie wald test)
#     """
#     # Prepare data
#     term2pert, s = breakdown_triple_wRef(
#         combo_perturbation, third_pos=coef_c, delim=delim, ref=reference
#     )
#     pert2term = {v: k for k, v in term2pert.items()}

#     perturbations = term2pert.values()
#     data = adata[adata.obs[perturbation_col].isin(perturbations), target]
#     # make sure that coef_c is the 3rd in var
#     indicators = get_perturbation_matrix(
#         data,
#         perturbation_col=perturbation_col,
#         inplace=False,
#         set_ref_1=reference,
#         verbose=False,
#     )
#     indicators = indicators.loc[:, s]
#     # return
#     # df = pd.DataFrame(indicators, columns=s)
#     y = data.X.toarray().flatten()
#     indicators["y"] = list(y)

#     indicators[perturbation_col] = data.obs[perturbation_col]
#     indicators["term"] = [pert2term[x] for x in indicators[perturbation_col]]

#     out = {
#         "perturbation": combo_perturbation,
#         "reference": term2pert["ref"],
#         "target": target,
#         "n_cells": indicators.shape[0],
#         "mean": float(np.mean(y)),
#         "var": float(np.var(y, ddof=1)),
#     }
#     out.update(term2pert)

#     formula = f"y ~ {s[0]} + {s[1]} + {s[2]} + {s[3]} + {s[1]}:{s[2]} + {s[1]}:{s[3]} + {s[2]}:{s[3]} + {s[1]}:{s[2]}:{s[3]} -1"

#     if method == "robust":
#         out["method"] = "robust"
#         print("\tUsing robust regression")
#         regr = smf.rlm(formula, data=indicators)
#     elif method == "ols":
#         out["method"] = "ols"
#         print("\tUsing OLS regression") if not quiet else None
#         regr = smf.ols(formula, data=indicators)
#     elif method in ["negbin", "nb", "negativebinomial"]:
#         print("\tUsing Negative Binomial regression") if not quiet else None
#         y_ref = indicators.loc[indicators["term"] == "ref", "y"].values
#         alpha_estimated, mean_ref, var_ref = estimate_alpha_empirical(y_ref)
#         (
#             print(
#                 f"Estimated alpha: {alpha_estimated} -  Mean: {mean_ref} - Variance: {var_ref}"
#             )
#             if not quiet
#             else None
#         )

#         if alpha_estimated is not None:
#             # Using GLM with a Negative Binomial family
#             out["method"] = "negativebinomial"
#             regr = smf.glm(
#                 formula,
#                 data=indicators,
#                 family=sm.families.NegativeBinomial(alpha=alpha_estimated),
#             )
#         else:
#             out["method"] = "poisson"
#             # use a poison regression
#             regr = smf.glm(formula, data=indicators, family=sm.families.Poisson())

#     regr_fit = regr.fit()
#     Z = regr_fit.predict(indicators)
#     indicators["predicted"] = Z

#     # return regr_fit
#     for i, val in enumerate(term2pert.keys()):
#         out[f"pval_{val}"] = regr_fit.pvalues[i] + eps
#         out[f"tstat_{val}"] = regr_fit.tvalues[i]
#         out[f"std_err_{val}"] = regr_fit.bse[i]
#         out[f"coef_{val}"] = regr_fit.params[i]
#         out[f"abs_coef_{val}"] = abs(regr_fit.params[i])

#     out["-log10_pval_abc"] = -np.log10(regr_fit.pvalues[-1] + eps)

#     # out['score'] = regr_fit.rsquared
#     out["corr_fit"] = spearmanr(Z.values, y)[0]

#     for val, pert in term2pert.items():
#         inds = data.obs[perturbation_col] == pert
#         out[f"n_cells_{val}"] = np.sum(inds)
#         out[f"mean_{val}"] = float(np.mean(y[inds]))
#         out[f"var_{val}"] = float(np.var(y[inds], ddof=1))
#         out[f"predicted_mean_{val}"] = float(np.mean(Z[inds]))
#         out[f"median_{val}"] = float(np.median(y[inds]))
#         out[f"predicted_median_{val}"] = float(np.median(Z[inds]))

#     # add predictions for ab group when not using the ab coef
#     inds = data.obs[perturbation_col] == combo_perturbation
#     abc_indicators = indicators.loc[inds, s]
#     predicted_mean_abc_only_single = np.sum(
#         abc_indicators
#         * np.array([out["coef_ref"], out["coef_a"], out["coef_b"], out["coef_c"]]),
#         axis=1,
#     )
#     out["predicted_mean_abc_only_single"] = float(
#         np.mean(predicted_mean_abc_only_single)
#     )
#     out["predicted_mean_abc_no_interaction"] = float(
#         np.mean(np.sum(abc_indicators.iloc[:, :-1] * regr_fit.params[:-1], axis=1))
#     )

#     if plot and not quiet:
#         fig, ax = plt.subplots(1, 2, figsize=(8, 5))

#         sns.boxplot(x=data.obs[perturbation_col], y=y, ax=ax[0], order=perturbations)
#         ax[0].set_title(f"Target: {target}")
#         ax[0].set_ylabel("Expression (difference from reference)")
#         ax[0].set_xlabel(None)
#         ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha="right")

#         # set color of each tick to its corresponding hue
#         for i, xtick in enumerate(ax[0].get_xticklabels()):
#             xtick.set_color(sns.color_palette()[i])
#         # add line with text for only single and no interaction
#         ax[0].axhline(
#             out["predicted_mean_abc_only_single"],
#             color="red",
#             linestyle="--",
#             alpha=0.3,
#         )
#         ax[0].axhline(
#             out["predicted_mean_abc_no_interaction"],
#             color="blue",
#             linestyle="--",
#             alpha=0.3,
#         )
#         # ax[0].text(0, out['predicted_mean_abc_only_single'], "only singles", color='red', alpha=0.3)
#         # ax[0].text(0, out['predicted_mean_abc_no_interaction'], "all but interaction", color='blue', alpha=0.3)
#         ax[0].text(
#             0.5, 0.5, "only singles", color="red", alpha=0.3, transform=ax[0].transAxes
#         )
#         ax[0].text(
#             0.5,
#             0.4,
#             "all but interaction",
#             color="blue",
#             alpha=0.3,
#             transform=ax[0].transAxes,
#         )

#         eps = 1e-200
#         sns.scatterplot(
#             y=-np.log10(regr_fit.pvalues + eps),
#             x=regr_fit.params,
#             hue=term2pert.values(),
#             hue_order=perturbations,
#             ax=ax[1],
#             alpha=0.7,
#         )
#         # set legend outside
#         ax[1].axhline(-np.log10(0.05), color="black", linestyle="--", alpha=0.3)
#         ax[1].set_title("Coefficients")
#         ax[1].set_xlabel("Coefficient value")
#         ax[1].set_ylabel("-log10(p-value)")
#         # remove legend
#         ax[1].get_legend().remove()

#         fig.tight_layout()
#         plt.show()
#     return out
