##########################################################################
#
# Functions for plotting and visualizing data
#
##########################################################################

import warnings
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import upsetplot as up
from adjustText import adjust_text
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch
from scipy.sparse import csr_matrix, issparse
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import precision_recall_curve, roc_curve

from .interaction import norman_model
from .utils import cluster_df, get_average_precision_score, get_perturbation_matrix

# Additional imports for norman_model_umap
try:
    from onesense import onesense
    from umap import UMAP

    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def plot_label_similarity(similarity_results, **kwargs) -> plt.Figure:
    """
    Plot the distribution of pairwise similarities between labels in an AnnData object, the AUPRC, and AUROC curves.

    Parameters:
        similarity_results (pd.DataFrame): The pairwise similarity results from calculate_label_similarity.
        **kwargs: Additional keyword arguments to pass to seaborn.histplot.

    Example usage:
        plot_label_similarity(similarity_results)
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Violin plot for similarities
    sns.violinplot(x="within", y="similarity", data=similarity_results, ax=axs[0])
    axs[0].set_xticklabels({True: "Within", False: "Across"})

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(
        similarity_results["within"], similarity_results["similarity"]
    )
    baseline = similarity_results["within"].sum() / len(similarity_results["within"])
    axs[1].plot(recall, precision)
    axs[1].plot([0, 1], [baseline, baseline], linestyle="--")
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].set_title("Precision-Recall Curve")

    # ROC Curve
    fpr, tpr, _ = roc_curve(
        ~similarity_results["within"], similarity_results["similarity"]
    )
    axs[2].plot(fpr, tpr)
    axs[2].plot([0, 1], [0, 1], linestyle="--")
    axs[2].set_xlabel("False Positive Rate")
    axs[2].set_ylabel("True Positive Rate")
    axs[2].set_title("ROC Curve")

    # Average Precision and AUROC
    avg_prec = get_average_precision_score(similarity_results)
    auroc = np.trapz(tpr, fpr)
    suptitle = f"Total labels: {len(similarity_results['label1'].unique())} | AUPRC: {avg_prec:.2f} | AUROC: {auroc:.2f}"
    fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()


def plot_filters(
    filters: Union[dict, list], adata: sc.AnnData, axis: str = "obs", **kwargs
) -> plt.Figure:
    """
    Plot the filters on as an upset plot.

    Args:
        filters: Either dictionary of filters in the form accepted by .utils.filter_adata that must contain the "axis" key (default is "obs"). Or a list of filters directly.
        adata: AnnData object
        axis: Axis to filter on. Default is "obs"
        **kwargs: Additional arguments to pass to upsetplot.plot
    Returns:
        The matplotlib figure object containing the upset plot.
    """

    # check if list or dict
    if isinstance(filters, dict):
        filters = filters[axis]
    elif isinstance(filters, list):
        filters = filters
    else:
        raise ValueError("Filters must be either a dictionary or list.")

    if axis == "obs":
        df = adata.obs
    elif axis == "var":
        df = adata.var
    else:
        raise ValueError("Axis must be either 'obs' or 'var'.")

    upset_df = pd.concat([df.eval(filters[i]) for i in range(len(filters))], axis=1)
    upset_df.columns = filters
    print(upset_df.head())

    for arg, val in [
        ("min_subset_size", "0.5%"),
        ("sort_by", "cardinality"),
        ("show_percentages", "{:.0%}"),
    ]:
        if arg not in kwargs:
            kwargs[arg] = val

    up.plot(upset_df.value_counts(), **kwargs)


def target_change_heatmap(
    adata,
    perturbation_column,
    quiet=False,
    heatmap_kws={},
    figsize=None,
    metric="log2fc",
    return_fig=False,
) -> Optional[plt.Figure]:
    """Generate a heatmap showing target gene changes across perturbations.

    Args:
        adata: AnnData object containing perturbation data and target change metrics.
        perturbation_column: Column name in adata.obs containing perturbation labels.
        quiet: Whether to suppress progress messages. Defaults to False.
        heatmap_kws: Additional keyword arguments to pass to seaborn.heatmap. Defaults to {}.
        figsize: Figure size as (width, height) tuple. If None, automatically calculated based on data dimensions.
        metric: Target change metric to plot. Options are 'log2fc', 'zscore', 'pct_change'. Defaults to 'log2fc'.
        return_fig: Whether to return the figure object instead of displaying it. Defaults to False.

    Returns:
        Figure object if return_fig is True, otherwise None.
    """

    if metric not in ["log2fc", "zscore", "pct_change"]:
        raise ValueError(
            f"Metric '{metric}' not recognized. Please choose from 'log2fc', 'zscore', 'pct_change'."
        )

    value = "target_" + metric
    if value not in adata.obsm:
        raise ValueError(
            "Target change metrics not found in adata.obsm. "
            "Please run calculate_target_change first. "
            "If single perturbation data with 'collapse_to_obs' as False."
        )

    if value not in adata.obsm:
        raise ValueError(
            f"No target change metrics found in adata.obsm. Please run calculate_target_change first. Note: if single perturbation data, ensure 'collapse_to_obs' is set to false"
        )

    target_change = (
        adata.obsm[value]
        .groupby(adata.obs[perturbation_column])
        .median()
        .sort_index(axis=0)
        .sort_index(axis=1)
    )

    # check if contains inf
    if np.any(np.isinf(target_change)):
        warnings.warn("Some values are infinite. Replacing with NaN.")
        target_change = target_change.replace([np.inf, -np.inf], np.nan)

    # plot the heatmap
    figsize = (
        (0.3 * len(target_change.columns), 0.3 * len(target_change.index))
        if figsize is None
        else figsize
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for key, val in [
        ("center", 0),
        ("xticklabels", True),
        ("yticklabels", True),
        ("cbar_kws", {"label": value}),
        ("cmap", "coolwarm"),
    ]:
        if key not in heatmap_kws.keys():
            heatmap_kws[key] = val
    sns.heatmap(target_change, ax=ax, **heatmap_kws)
    ax.set_xlabel("Target Genes")
    ax.set_ylabel("Perturbation")

    if return_fig:
        return fig
    else:
        plt.show()
        return None


def target_gene_heatmap(
    adata: sc.AnnData,
    control_value: str,
    perturbation_column: str = "perturbation",
    perturbation_gene_map: Optional[Dict[str, str]] = None,
    quiet: bool = False,
    heatmap_kws: Dict = {},
    figsize: Optional[Tuple[float, float]] = None,
    method: str = "log2FC",
    return_fig: bool = False,
    # check_norm=True, #for now assume that the heatmap should be calculated on adata.X
) -> Optional[plt.Figure]:
    """
    Generate a heatmap of target gene expression changes.

    Parameters:
    adata: AnnData object
    control_value: str
        Value in perturbation_column to use as reference
    perturbation_column: str
        Column in adata.obs containing perturbation information
    perturbation_gene_map: dict, optional
        Mapping from perturbation labels to target genes
    quiet: bool
        Whether to suppress output
    heatmap_kws: dict
        Keyword arguments passed to sns.heatmap
    figsize: tuple, optional
        Figure size (width, height)
    method: str
        Method for calculating changes ('log2FC', 'zscore', etc.)
    return_fig: bool
        Whether to return the figure object

    Returns:
    matplotlib.figure.Figure (if return_fig=True)
    """

    if not quiet:
        print(f"Generating target gene heatmap with method: {method}")
        print(f"Using control value: {control_value}")

    # Generate perturbation matrix
    pm = get_perturbation_matrix(
        adata,
        perturbation_column,
        control_value=control_value,
        inplace=False,
        verbose=not quiet,
    )

    if not quiet:
        print(
            f"\tFound {pm.shape[1]} unique perturbations in {perturbation_column} column."
        )

    # check that the gene a perturbation maps to is actually in adata
    if perturbation_gene_map is not None:
        # for now we assume all the perturbations are in the perturbation_gene_map
        pm.columns = [perturbation_gene_map[x] for x in pm.columns]

    # Warn if np.any(pm.sum(axis=1) > 1)
    if np.any(pm.sum(axis=1) > 1):
        warnings.warn(
            "Some genes are perturbed by more than one perturbation. This is not recommended for this heatmap."
        )

    check = [x in adata.var_names for x in pm.columns]
    if sum(check) == 0:
        raise ValueError(
            f"No perturbations found in adata.var_names. Please check the perturbation_gene_map or perturbation_column."
        )
    elif sum(check) != len(check):
        if not quiet:
            print(
                f"\tMissing {len(check) - sum(check)} perturbations not found in adata.var_names."
            )

    genes = pm.columns[pm.columns.isin(adata.var_names)].sort_values()
    pm = pm.loc[:, pm.columns.sort_values()]
    gene_vals = adata[:, genes].X
    # convert to numpy if sparse
    gene_vals = gene_vals.toarray() if issparse(gene_vals) else gene_vals

    ref_bool = (pm.sum(axis=1) == 0).values
    ref_mean = gene_vals[ref_bool].mean(axis=0)

    if method not in ["log2FC", "zscore", "pct"]:
        raise ValueError(
            f"Method '{method}' not recognized. Please choose from 'log2FC', 'zscore', 'pct'."
        )

    if method == "log2FC":
        target_change = np.log2(gene_vals + 1) - np.log2(ref_mean + 1)
        annot = "log2FC target"
    elif method == "zscore":
        target_change = (gene_vals - ref_mean) / gene_vals[ref_bool].std(axis=0)
        annot = "Zscore target"
    elif method == "pct":
        target_change = ((gene_vals - ref_mean) / ref_mean) * 100
        annot = "Pct target change"

    # get average
    target_change = pm.T @ target_change
    target_change = target_change.T
    target_change /= pm.sum(axis=0).values

    # save to df
    target_change = pd.DataFrame(
        target_change.T.values, columns=genes, index=pm.columns
    )

    # plot the heatmap
    figsize = (
        (0.3 * len(target_change.columns), 0.3 * len(target_change.index))
        if figsize is None
        else figsize
    )
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    for key, val in [
        ("center", 0),
        ("xticklabels", True),
        ("yticklabels", True),
        ("cbar_kws", {"label": annot}),
        ("cmap", "coolwarm"),
    ]:
        if key not in heatmap_kws.keys():
            heatmap_kws[key] = val
    sns.heatmap(target_change, ax=ax, **heatmap_kws)
    ax.set_xlabel("Target Genes")
    ax.set_ylabel("Perturbation")
    # plt.show()
    if return_fig:
        return fig
    else:
        plt.show()
        return None


def dotplot(
    sizes,
    colors,
    return_ax=False,
    ax=None,
    center=0,
    cmap="RdBu",
    cluster=True,
    cluster_kws={},
    cluster_on="colors",
    **kwargs,
) -> Optional[plt.Axes]:
    """Create a dot plot where dot sizes and colors represent different data dimensions.

    Args:
        sizes: DataFrame containing values that determine dot sizes.
        colors: DataFrame containing values that determine dot colors. Must have matching indices and columns with sizes.
        return_ax: Whether to return the axes object instead of displaying the plot. Defaults to False.
        ax: Existing axes object to plot on. If None, creates new figure and axes.
        center: Center value for color mapping. Defaults to 0.
        cmap: Colormap for dot colors. Defaults to 'RdBu'.
        cluster: Whether to cluster rows and columns. Defaults to True.
        cluster_kws: Additional keyword arguments to pass to clustering function. Defaults to {}.
        cluster_on: Which data to use for clustering ('sizes' or 'colors'). Defaults to 'colors'.
        **kwargs: Additional keyword arguments.
    Returns:
        Axes object if return_ax is True, otherwise None.
    """

    assert sizes.shape == colors.shape
    N, M = sizes.shape

    # confirm index and columns are the same
    assert all(sizes.index == colors.index)
    assert all(sizes.columns == colors.columns)

    if cluster:
        if cluster_on == "sizes":
            sizes = cluster_df(sizes, **cluster_kws)
            colors = colors.loc[sizes.index, sizes.columns]
        elif cluster_on == "colors":
            colors = cluster_df(colors, **cluster_kws)
            sizes = sizes.loc[colors.index, colors.columns]
        else:
            raise ValueError("cluster_on must be 'sizes' or 'colors'")

    ylabels = sizes.index
    xlabels = sizes.columns

    x, y = np.meshgrid(np.arange(M), np.arange(N))
    s = sizes.values
    c = colors.values

    if ax is None:
        fig, ax = plt.subplots(figsize=(M, N))

    R = s / s.max() / 2
    circles = [plt.Circle((j, i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
    col = PatchCollection(
        circles,
        array=c.flatten(),
        cmap=cmap,
    )
    ax.add_collection(col)

    ax.set(
        xticks=np.arange(M),
        yticks=np.arange(N),
        xticklabels=xlabels,
        yticklabels=ylabels,
    )
    ax.set_xticks(np.arange(M + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(N + 1) - 0.5, minor=True)
    ax.grid(which="minor")

    cbar = ax.figure.colorbar(col, ax=ax, orientation="vertical", pad=0.00)
    if return_ax:
        return ax
    else:
        plt.show()
        return None


def plot_adj_matr(
    adata,
    row_colors=None,
    col_colors=None,
    row_order=None,
    col_order=None,
    show=False,
    **kwargs,
) -> None:
    """Plot an adjacency matrix with optional row and column colors using a clustermap.

    Args:
        adata: AnnData object containing adjacency matrix in adata.obsm['adjacency'].
        row_colors: Row color specification. Can be string (column name from adata.obs), list, array, or Series. Defaults to None.
        col_colors: Column color specification. Can be string (column name from adata.obs), list, array, or Series. Defaults to None.
        row_order: Order for row color categories. If None, uses unique values from row_colors. Defaults to None.
        col_order: Order for column color categories. If None, uses unique values from col_colors. Defaults to None.
        show: Whether to display the plot. Defaults to False.
        **kwargs: Additional keyword arguments to pass to seaborn.clustermap.
    """

    # check if .obsm['adjacency'] exists
    if "adjacency" not in adata.obsm.keys():
        raise ValueError("No adjacency matrix found in adata.obsm['adjacency']")

    # if row color is list do nothing, if its string, assume its the key from adata.obs

    if type(row_colors) == str:
        row_colors = adata.obs[row_colors]
    elif (
        (type(row_colors) == list)
        | (type(row_colors) == np.ndarray)
        | (type(row_colors) == pd.Series)
    ):
        pass
    else:
        row_colors = None

    if row_colors is not None:
        if row_order is None:
            row_order = list(set(row_colors))
        lut = dict(zip(row_order, sns.color_palette("Set2", len(row_order))))
        # row_colors = row_colors.map(lut).values
        row_colors = [lut[i] for i in row_colors]

    if type(col_colors) == str:
        col_colors = adata.obs[col_colors]
    elif (
        (type(col_colors) == list)
        | (type(col_colors) == np.ndarray)
        | (type(col_colors) == pd.Series)
    ):
        pass
    else:
        col_colors = None

    if col_colors is not None:
        if col_order is None:
            col_order = list(set(col_colors))
        lut = dict(zip(col_order, sns.color_palette("Set2", len(col_order))))
        col_colors = [lut[i] for i in col_colors]

    sns.clustermap(
        adata.obsm["adjacency"], row_colors=row_colors, col_colors=col_colors, **kwargs
    )

    if row_colors is not None:
        handles = [Patch(facecolor=lut[name]) for name in lut]
        plt.legend(
            handles,
            lut,
            title="Species",
            bbox_to_anchor=(0, 0),
            bbox_transform=plt.gcf().transFigure,
            loc="lower left",
        )

    if show:
        plt.show()


def plot_double_single(
    data: Union[pd.DataFrame, sc.AnnData],
    double_condition: str,
    pred: bool = False,
    metric: str = "fit_spearmanr",
    genes: Optional[List[str]] = None,
    delim: str = "|",
    **kwargs,
) -> None:
    """
    Plot a heatmap comparing single and double perturbation expression profiles.

    Args:
        data: DataFrame with perturbations as index and genes as columns
        double_condition: String representing dual perturbation (e.g., "GENE1|GENE2")
        pred: Whether to include model predictions
        metric: Metric to display in title
        genes: Specific genes to include (default: all)
        delim: Delimiter for splitting dual perturbations
        **kwargs: Additional arguments passed to sns.heatmap
    """
    # if data is anndata then make it df
    if type(data) == sc.AnnData:
        print("Found AnnData, densifying to df. This may take a while... ")
        data = data.to_df()

    # confirm that all genes are in data
    if genes is None:
        gs = data.columns
        print("using all genes")
    else:
        gs = [g for g in genes if g in data.columns]
        print(f"{len(gs)}/{len(genes)} genes found in data.")

    # Extract single perturbations from double
    A, B = double_condition.split(delim)
    conds = [A, B, double_condition]

    sub = data.loc[conds, data.columns.isin(gs)]

    subdf = cluster_df(sub, cluster_rows=False)
    # sub.obs
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    if pred:
        # add pred to sub
        m, Z = norman_model(
            subdf, double_condition, targets=gs, plot=False, verbose=False
        )
        if m is not None and Z is not None:
            subdf.loc[f"Predicted", gs] = Z.flatten()
            title = f"{double_condition} \n{round(float(m['coef_a']),2)}({m['a']}) x {round(float(m['coef_b']),2)}({m['b']}) \nSpearman: {round(m[metric],2)}"
        else:
            title = f"{double_condition} (prediction failed)"
    else:
        title = double_condition

    # #palette with centered coloring at 0
    sns.heatmap(subdf, cmap="RdBu_r", center=0, ax=ax, **kwargs)
    # cg.ax_col_dendrogram.set_visible(False)
    plt.ylabel("")
    plt.title(title)
    plt.show()


def comparison_plot(
    pdf: pd.DataFrame,
    x: str = "x",
    y: str = "y",
    metric: str = "metric",
    label: bool = True,
    to_label: float = 0.1,
    yx_line: bool = True,
    show: bool = False,
) -> None:
    """Plot a comparison between two vectors with optional labeling and y=x line.

    Args:
        pdf: DataFrame containing the data to plot.
        x: Column name for x values. Defaults to 'x'.
        y: Column name for y values. Defaults to 'y'.
        metric: Column name for metric used to determine which points to label. Defaults to 'metric'.
        label: Whether to label the top points by metric. Defaults to True.
        to_label: If < 1, the percent of points to label; if > 1, the number of points to label. Defaults to 0.1.
        yx_line: Whether to plot a y=x line. Defaults to True.
        show: Whether to show the plot. Defaults to False.
    """

    # calculate fit and residuals
    sns.scatterplot(data=pdf, x=x, y=y, hue=metric)

    # label top % by metric
    if to_label > 1:
        n = int(to_label)
    else:
        n = int(len(pdf) * to_label)
    topN = pdf.sort_values(metric, ascending=False).head(n)
    texts = []
    for i, row in topN.iterrows():
        texts.append(plt.text(row[x], row[y], i, fontsize=10))
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="k", lw=0.5))

    # add y = x line for min and max

    mn, mx = min(pdf[x].min(), pdf[y].min()), max(pdf[x].max(), pdf[y].max())
    plt.plot([mn, mx], [mn, mx], color="red", linestyle="--")

    if show:
        plt.show()


def plot_kd(
    adata: sc.AnnData, gene: str, ref_val: str, exp_val: str, col: str = "perturbation"
) -> None:
    """Plot histograms comparing gene expression between reference and experimental conditions.

    Args:
        adata: AnnData object containing gene expression data.
        gene: Name of the gene to plot expression for.
        ref_val: Reference condition value (e.g., control).
        exp_val: Experimental condition value (e.g., knockdown).
        col: Column name in adata.obs containing condition labels. Defaults to 'perturbation'.
    """
    gene_vals = adata[:, gene].X.toarray().flatten()
    ##plot AR for AR KD vs NTC|NTC
    gene_inds = adata.obs[col] == exp_val
    NTC_inds = adata.obs[col] == ref_val
    print(f"Number of obs in NTC: {np.sum(NTC_inds)}")
    print(f"Number of obs in {gene} KD: {np.sum(gene_inds)}")

    plt.hist(gene_vals[NTC_inds], label=ref_val, alpha=0.5, bins=30)
    plt.hist(gene_vals[gene_inds], label=exp_val + " KD", alpha=0.5, bins=30)
    # add mean line for each group
    plt.axvline(gene_vals[NTC_inds].mean(), color="blue")
    plt.axvline(gene_vals[gene_inds].mean(), color="orange")
    plt.title(f"{exp_val} KD vs {ref_val} for gene {gene}")
    plt.legend()
    plt.show()


import matplotlib.pyplot as plt
from scipy.stats import pearsonr


def corrfunc(
    x: np.ndarray,
    y: np.ndarray,
    ax: Optional[plt.Axes] = None,
    method: str = "spearman",
    **kws,
) -> None:
    """Plot the correlation coefficient in the top left hand corner of a plot.

    Args:
        x: Array of x values for correlation calculation.
        y: Array of y values for correlation calculation.
        ax: Matplotlib axes object to annotate. If None, uses current axes. Defaults to None.
        method: Correlation method to use ('spearman' or 'pearson'). Defaults to 'spearman'.
        **kws: Additional keyword arguments (currently unused).
    """
    func = spearmanr if method == "spearman" else pearsonr
    r, _ = func(x, y, nan_policy="omit")
    ax = ax or plt.gca()
    ax.annotate(f"ρ = {r:.2f}", xy=(0.1, 0.9), xycoords=ax.transAxes)


def square_plot(
    x: pd.Series,
    y: pd.Series,
    ax: Optional[plt.Axes] = None,
    show: bool = True,
    corr: Optional[str] = None,
    **kwargs,
) -> None:
    """Plot a square scatter plot of x vs y with a y=x diagonal line.

    Args:
        x: Series containing x values.
        y: Series containing y values.
        ax: Matplotlib axes object to plot on. If None, creates new figure and axes. Defaults to None.
        show: Whether to display the plot. Defaults to True.
        corr: Correlation method to calculate and display ('spearman' or 'pearson'). If None, no correlation is shown. Defaults to None.
        **kwargs: Additional keyword arguments to pass to seaborn.scatterplot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))
    sns.scatterplot(x=x, y=y, ax=ax, **kwargs)
    # add y = x line for min and max
    # ax[i].plot([0,1], [0,1], color='red', linestyle='--')
    # get min and max values

    if corr == "spearman":
        corr = spearmanr(x, y, nan_policy="omit")[0]
    elif corr == "pearson":
        corr = pearsonr(x, y)[0]

    if corr is not None:
        # put correlation bottom right
        ax.text(0.8, 0.1, f"r={round(corr,2)}", transform=ax.transAxes)

    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], color="red", linestyle="--")
    if show:
        plt.show()


### guide plotting
##########################################################################


## plot ratio of top 2 against counts:
def plot_top2ratio_counts(features: sc.AnnData, show: bool = False) -> sns.JointGrid:
    """Plot QC metrics showing the relationship between total feature counts and top 2 feature ratio.

    Args:
        features: AnnData object containing feature count data with 'log10_total_feature_counts' and 'log2_ratio_2nd_1st_feature' in obs.
        show: Whether to display the plot. Defaults to False.
    Returns:
        Seaborn JointGrid object containing the plot.
    """
    g = sns.jointplot(
        data=features.obs,
        y="log10_total_feature_counts",
        x="log2_ratio_2nd_1st_feature",
        kind="hex",
    )
    g.ax_joint.set_ylabel("log10(total counts/cell)")
    g.ax_joint.set_xlabel("log2(2nd top feature / top feature)")
    g.fig.suptitle("Cell level metrics")
    g.fig.tight_layout()
    if show:
        plt.show()

    return g


# plot guide call proportions
# def plot_feature_count_metrics(features, ntc_var=None, show=False, ax=None):
#     if ax is None:
#         fig, ax = plt.subplots(1,1)

#     sns.scatterplot(data = features.var, y = 'pct_cells_with_feature', x = 'log10_total_feature_counts', hue=ntc_var, ax=ax)
#     # uniform coverage expectation
#     if ntc_var in features.var.columns:
#         unif = (1/features.var.query(f'{ntc_var} == False').shape[0])/2
#         ax.axhline(unif, color='r', linestyle='--')
#     #add label to axhline
#         xmax = ax.get_xlim()[1]
#         ax.text(xmax, unif, 'Expected %', ha='right', va='bottom', color='r')
#         ax.legend(title='Negative Control')
#     ax.set_xlabel('log10(total counts/feature)')
#     ax.set_ylabel('% cells with feature')
#     ax.set_ylim(0,0.2)
#     #change y ticks to be multiplied by 100
#     ax.set_yticks(plt.yticks()[0], [f'{int(x*100)}%' for x in plt.yticks()[0]])
#     if show:
#         plt.show()
#     return ax


# plot guide call numbers as proportion of all cells
## plot num features
def plot_num_features(
    features: sc.AnnData, show: bool = False, ax: Optional[plt.Axes] = None, **kwargs
) -> plt.Axes:
    """Plot the distribution of number of feature calls per cell as a bar plot.

    Args:
        features: AnnData object containing feature data with 'num_features' column in obs.
        show: Whether to display the plot. Defaults to False.
        ax: Matplotlib axes object to plot on. If None, creates new figure and axes. Defaults to None.
        **kwargs: Additional keyword arguments to pass to seaborn.barplot.
    Returns:
        Matplotlib axes object containing the plot.
    """
    vc = features.obs["num_features"].value_counts()
    vc = vc.sort_index()
    vc = vc / vc.sum() * 100

    if ax is None:
        fig, ax = plt.subplots(1, 1)
    sns.barplot(data=vc.reset_index(), x="num_features", y="count", ax=ax, **kwargs)
    ax.set_xlabel("# feature calls per cell")
    ax.set_ylabel("% of all cells")
    if show:
        plt.show()

    return ax


##########################################################################
# Norman Model UMAP Analysis
##########################################################################


def _get_umap(xdata: np.ndarray, random_state: int, **kwargs) -> Tuple[np.ndarray, int]:
    """Helper function for UMAP transformation.

    Args:
        xdata: Input data for UMAP transformation.
        random_state: Random state for reproducibility.
        **kwargs: Additional arguments passed to UMAP.

    Returns:
        Tuple of (transformed data, random state).
    """
    if not UMAP_AVAILABLE:
        raise ImportError(
            "UMAP and/or onesense packages are required for norman_model_umap. "
            "Install with: pip install umap-learn onesense"
        )

    transformer = UMAP(random_state=random_state, **kwargs)
    x = transformer.fit_transform(xdata)
    return x, random_state


def norman_model_umap(
    gi_df: pd.DataFrame,
    rx: int = 123,
    ry: int = 456,
    plot_metric: str = "coef_norm2",
    save: Optional[
        str
    ] = None,  # path to save the UMAP figure as well as the dataframe itself
    **kwargs,
) -> pd.DataFrame:
    """
    Generate UMAP visualization of genetic interaction results from Norman model.

    Tom's approach for visualizing genetic interactions using UMAP dimensionality reduction
    and onesense clustering to identify interaction patterns.

    Args:
        gi_df: DataFrame containing genetic interaction metrics from norman_model
        rx: Random state for x-axis UMAP
        ry: Random state for y-axis UMAP
        plot_metric: Metric to use for color coding ('coef_norm2', 'abs_log10_ratio_coefs', etc.)
        save: Path prefix to save results (will create .csv and .png files)
        **kwargs: Additional arguments passed to onesense

    Returns:
        pd.DataFrame: Original dataframe with added UMAP coordinates and cluster assignments
    """

    if not UMAP_AVAILABLE:
        raise ImportError(
            "UMAP and onesense packages are required for norman_model_umap. "
            "Install with: pip install umap-learn onesense"
        )

    metric2term = {
        "coef_norm2": "Magnitude",
        "abs_log10_ratio_coefs": "Dominance",
        "dcor_AB_fit": "Model fit",
        "dcor_AnB_AB": "Similarity of singles to double",
        "dcor_A_B": "Similarity between singles",
        "dcor_ratio": "Equality of contribution",
    }

    regr_fit = gi_df.copy()
    xs = regr_fit[["coef_norm2", "abs_log10_ratio_coefs", "dcor_AB_fit"]]
    ys = regr_fit[["dcor_AnB_AB", "dcor_A_B", "dcor_ratio"]]

    x_table = xs.copy()
    x_table = (x_table) / x_table.std()
    y_table = ys.copy()
    y_table = (y_table) / y_table.std()

    x, _ = _get_umap(
        x_table, rx, n_components=1, n_neighbors=5, min_dist=0.05, spread=0.5
    )
    y, _ = _get_umap(
        y_table, ry, n_components=1, n_neighbors=5, min_dist=0.05, spread=0.5
    )

    x = pd.Series(x.flatten().astype(float), index=x_table.index)
    y = pd.Series(y.flatten().astype(float), index=y_table.index)

    xs_list = [xs[col].values for col in xs.columns]
    ys_list = [ys[col].values for col in ys.columns]

    cx, cy = onesense(
        x,
        y,
        regr_fit[plot_metric],
        xs_list,
        ys_list,
        xlabels=[metric2term[col] for col in xs.columns],
        ylabels=[metric2term[col] for col in ys.columns],
        label=True,
        figsize=[15, 15],
        ylims=((0, 1), (0, 1), (0, 1)),
        **kwargs,
    )

    regr_fit["x_umap"] = x
    regr_fit["y_umap"] = y
    regr_fit["x_cluster"] = cx
    regr_fit["y_cluster"] = cy

    if save:
        regr_fit.to_csv(save + ".csv")
        plt.savefig(save + ".png", bbox_inches="tight")

    return regr_fit
