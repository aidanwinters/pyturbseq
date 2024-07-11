##########################################################################
# 
# Functions for plotting and visualizing data
#
##########################################################################

import warnings 
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from adjustText import adjust_text
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr, pearsonr
from scipy.sparse import csr_matrix, issparse

from .utils import cluster_df, get_perturbation_matrix, get_average_precision_score
from .interaction import get_singles, get_model_fit
from matplotlib.patches import Patch
from matplotlib.collections import PatchCollection
import upsetplot as up 
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_label_similarity(similarity_results, **kwargs):
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
    sns.violinplot(x='within', y='similarity', data=similarity_results, ax=axs[0])
    axs[0].set_xticklabels({True: 'Within', False: 'Across'})

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(similarity_results['within'], similarity_results['similarity'])
    baseline = similarity_results['within'].sum() / len(similarity_results['within'])
    axs[1].plot(recall, precision)
    axs[1].plot([0, 1], [baseline, baseline], linestyle='--')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].set_title('Precision-Recall Curve')

    # ROC Curve
    fpr, tpr, _ = roc_curve(~similarity_results['within'], similarity_results['similarity'])
    axs[2].plot(fpr, tpr)
    axs[2].plot([0, 1], [0, 1], linestyle='--')
    axs[2].set_xlabel('False Positive Rate')
    axs[2].set_ylabel('True Positive Rate')
    axs[2].set_title('ROC Curve')

    # Average Precision and AUROC
    avg_prec = get_average_precision_score(similarity_results)
    auroc = np.trapz(tpr, fpr)
    suptitle = f"Total labels: {len(similarity_results['label1'].unique())} | AUPRC: {avg_prec:.2f} | AUROC: {auroc:.2f}"
    fig.suptitle(suptitle)
    fig.tight_layout()
    plt.show()

def plot_filters(
    filters: [dict, list],
    adata: sc.AnnData,
    axis: str = 'obs',
    **kwargs
    ):
    """
    Plot the filters on as an upset plot.

    Args:
        filters: Either dictionary of filters in the form accepted by .utils.filter_adata that must contain the "axis" key (default is "obs"). Or a list of filters directly. 
        adata: AnnData object
        axis: Axis to filter on. Default is "obs"
        **kwargs: Additional arguments to pass to upsetplot.plot
    """

    #check if list or dict
    if isinstance(filters, dict):
        filters = filters[axis]
    elif isinstance(filters, list):
        filters = filters
    else:
        raise ValueError("Filters must be either a dictionary or list.")

    if axis == 'obs':
        df = adata.obs
    elif axis == 'var':
        df = adata.var
    else:
        raise ValueError("Axis must be either 'obs' or 'var'.")

    upset_df = pd.concat([df.eval(filters[i]) for i in range(len(filters))], axis=1)
    upset_df.columns = filters
    print(upset_df.head())

    for arg, val in [('min_subset_size', '0.5%'), ('sort_by', 'cardinality'), ('show_percentages', '{:.0%}')]:
        if arg not in kwargs:
            kwargs[arg] = val
    
    up.plot(
        upset_df.value_counts(),
        **kwargs
        )

def target_change_heatmap(
    adata,
    perturbation_column,
    quiet=False,
    heatmap_kws={},
    figsize=None,
    metric='log2fc',
    return_fig=False,
    ):

    if not quiet: print(f"Calculating target gene heatmap for {perturbation_column} column...")

    if metric not in ['log2fc', 'zscore', 'pct_change']:
        raise ValueError(f"Metric '{metric}' not recognized. Please choose from 'log2fc', 'zscore', 'pct_change'.")

    value = 'target_' + metric
    if value not in adata.obsm:
        raise ValueError(f"Target change metrics not found in adata.obsm. Please run calculate_target_change first. If single perturbation data with 'collapse_to_obs' as False.")

    if value not in adata.obsm:
        raise ValueError(f"No target change metrics found in adata.obsm. Please run calculate_target_change first. Note: if single perturbation data, ensure 'collapse_to_obs' is set to false")
    
    target_change = adata.obsm[value].groupby(adata.obs[perturbation_column]).median().sort_index(axis=0).sort_index(axis=1)
    
    #check if contains inf
    if np.any(np.isinf(target_change)):
        warnings.warn("Some values are infinite. Replacing with NaN.")
        target_change = target_change.replace([np.inf, -np.inf], np.nan)

    if np.any(adata.obsm['perturbation'].sum(axis=1) > 1):
        warnings.warn("Some genes are perturbed by more than one perturbation. This is not recommended for this heatmap.")

    #plot the heatmap
    figsize = (0.3*len(target_change.columns), 0.3*len(target_change.index)) if figsize is None else figsize
    fig, ax = plt.subplots(1,1, figsize=figsize)
    for key, val in [('center', 0), ('xticklabels', True), ('yticklabels', True), ('cbar_kws', {'label': value}), ('cmap', 'coolwarm')]:
        if key not in heatmap_kws.keys():
            heatmap_kws[key] = val
    sns.heatmap(target_change, ax=ax, **heatmap_kws)
    ax.set_xlabel('Target Genes')
    ax.set_ylabel('Perturbation')

    if return_fig:
        return fig
    else:
        plt.show()


def target_gene_heatmap(
    adata,
    reference_value,
    perturbation_column='perturbation',
    perturbation_gene_map=None,
    quiet=False,
    heatmap_kws={},
    figsize=None,
    method='log2FC',
    return_fig=False,
    # check_norm=True, #for now assume that the heatmap should be calculated on adata.X
    ):

    warnings.warn("This function is deprecated. Please use target_change_heatmap instead.")

    if not quiet: print(f"Calculating target gene heatmap for {perturbation_column} column...")

    #if no perturbation matrix, create one
    if perturbation_column is not None:
        if not quiet: print(f"\tGenerating perturbation matrix from '{perturbation_column}' column...")
        pm = get_perturbation_matrix(adata, perturbation_column, reference_value=reference_value, inplace=False, verbose=not quiet)
    elif 'perturbation' in adata.obsm.keys():
        pm = adata.obsm['perturbation']
    else: 
        raise ValueError("No perturbation matrix found in adata.obsm. Please provide a perturbation_column or run get_perturbation_matrix first.")

    if not quiet: print(f"\tFound {pm.shape[1]} unique perturbations in {perturbation_column} column.")

    #check that the gene a perturbation maps to is actually in adata
    if perturbation_gene_map is not None:
        #for now we assume all the perturbations are in the perturbation_gene_map
        pm.columns = [perturbation_gene_map[x] for x in pm.columns]

    #Warn if np.any(pm.sum(axis=1) > 1)
    if np.any(pm.sum(axis=1) > 1):
        warnings.warn("Some genes are perturbed by more than one perturbation. This is not recommended for this heatmap.")


    check = [x in adata.var_names for x in pm.columns]
    if sum(check) == 0:
        raise ValueError(f"No perturbations found in adata.var_names. Please check the perturbation_gene_map or perturbation_column.")
    elif sum(check) != len(check):
        if not quiet: print(f"\tMissing {len(check) - sum(check)} perturbations not found in adata.var_names.")

    genes = pm.columns[pm.columns.isin(adata.var_names)].sort_values()
    pm = pm.loc[:, pm.columns.sort_values()]
    gene_vals = adata[:, genes].X
    #convert to numpy if sparse
    gene_vals = gene_vals.toarray() if issparse(gene_vals) else gene_vals

    ref_bool = (pm.sum(axis=1) == 0).values
    ref_mean = gene_vals[ref_bool].mean(axis=0)

    if method not in ['log2FC', 'zscore', 'pct']:
        raise ValueError(f"Method '{method}' not recognized. Please choose from 'log2FC', 'zscore', 'pct'.")
    
    if method == 'log2FC':
        target_change = np.log2(gene_vals + 1) - np.log2(ref_mean + 1)
        annot = 'log2FC target'
    elif method == 'zscore':
        target_change = (gene_vals - ref_mean) / gene_vals[ref_bool].std(axis=0)
        annot = 'Zscore target'
    elif method == 'pct':
        target_change = ((gene_vals - ref_mean) / ref_mean) * 100
        annot = 'Pct target change'

    #get average
    target_change = pm.T @ target_change
    target_change = target_change.T
    target_change /= pm.sum(axis=0).values

    #save to df
    target_change = pd.DataFrame(target_change.T.values, columns=genes, index=pm.columns)

    #plot the heatmap
    figsize = (0.3*len(target_change.columns), 0.3*len(target_change.index)) if figsize is None else figsize
    fig, ax = plt.subplots(1,1, figsize=figsize)
    for key, val in [('center', 0), ('xticklabels', True), ('yticklabels', True), ('cbar_kws', {'label': annot}), ('cmap', 'coolwarm')]:
        if key not in heatmap_kws.keys():
            heatmap_kws[key] = val
    sns.heatmap(target_change, ax=ax, **heatmap_kws)
    ax.set_xlabel('Target Genes')
    ax.set_ylabel('Perturbation')
    # plt.show()
    if return_fig:
        return fig
    else:
        plt.show()


def dotplot(sizes, colors, return_ax=False, ax=None, center=0, cmap='RdBu', cluster=True, cluster_kws={}, cluster_on='colors', **kwargs):
    """
    Assumes that sizes and colros are dataframes with matching indices and columns
    """

    assert sizes.shape == colors.shape
    N, M = sizes.shape

    #confirm index and columns are the same
    assert all(sizes.index == colors.index)
    assert all(sizes.columns == colors.columns)

    if cluster:
        if cluster_on == 'sizes':
            sizes = cluster_df(sizes, **cluster_kws)
            colors = colors.loc[sizes.index, sizes.columns]
        elif cluster_on == 'colors':
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


    R = s/s.max()/2
    circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]
    col = PatchCollection(circles, array=c.flatten(), cmap=cmap, )
    ax.add_collection(col)

    ax.set(xticks=np.arange(M), yticks=np.arange(N),
        xticklabels=xlabels, yticklabels=ylabels)
    ax.set_xticks(np.arange(M+1)-0.5, minor=True)
    ax.set_yticks(np.arange(N+1)-0.5, minor=True)
    ax.grid(which='minor')

    cbar = ax.figure.colorbar(col, ax=ax, orientation='vertical', pad=0.00)
    if return_ax:
        return ax
    else:
        plt.show()

def plot_adj_matr(
    adata,
    row_colors=None,
    col_colors=None,
    row_order=None,
    col_order=None,
    show=False,
    **kwargs
    ):
    """
    Plot an adjacency matrix with row colors
    Args:  
    """

    #check if .obsm['adjacency'] exists
    if 'adjacency' not in adata.obsm.keys():
        raise ValueError("No adjacency matrix found in adata.obsm['adjacency']")

    # if row color is list do nothing, if its string, assume its the key from adata.obs

   
    if type(row_colors) == str:
        row_colors = adata.obs[row_colors]
    elif (type(row_colors) == list) | (type(row_colors) == np.ndarray) | (type(row_colors) == pd.Series):
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
    elif (type(col_colors) == list) | (type(col_colors) == np.ndarray) | (type(col_colors) == pd.Series):
        pass
    else:
        col_colors = None

    if col_colors is not None:
        if col_order is None:
            col_order = list(set(col_colors))
        lut = dict(zip(col_order, sns.color_palette("Set2", len(col_order))))
        col_colors = [lut[i] for i in col_colors]


    sns.clustermap(
        adata.obsm['adjacency'],
        row_colors=row_colors,
        col_colors=col_colors,
        **kwargs)

    if row_colors is not None:
            handles = [Patch(facecolor=lut[name]) for name in lut]
            plt.legend(handles, lut, title='Species',
                    bbox_to_anchor=(0, 0), bbox_transform=plt.gcf().transFigure, loc='lower left')

    if show:
        plt.show()

def plot_double_single(data, double_condition, pred=False, metric='fit_spearmanr', genes=None, **kwargs):

    # if data is anndata then make it df
    if type(data) == sc.AnnData:
        print("Found AnnData, densifying to df. This may take a while... ")
        data = data.to_df()

    #confirm that all genes are in data
    if genes is None:
        gs = data.columns
        print('using all genes')
    else:
        gs = [g for g in genes if g in data.columns]
        print(f"{len(gs)}/{len(genes)} genes found in data.")

    A, B = get_singles(double_condition)
    conds = [A, B, double_condition]

    sub = data.loc[conds, data.columns.isin(gs)]

    subdf = cluster_df(sub, cluster_rows=False)
    # sub.obs
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    if pred is not None:
        #add pred to sub
        m, Z = get_model_fit(subdf, double_condition, targets=gs, plot=False)
        subdf.loc[f"Predicted",gs] = Z.flatten()
        title = f"{double_condition} \n{round(float(m['coef_a']),2)}({m['a']}) x {round(float(m['coef_b']),2)}({m['b']}) \nSpearman: {round(m[metric],2)}"
        # ax.hlines([1], *ax.get_xlim())
    else:
        title = double_condition

    # #palette with centered coloring at 0
    sns.heatmap(subdf, cmap='RdBu_r', center=0, ax=ax, **kwargs)
    # cg.ax_col_dendrogram.set_visible(False)
    plt.ylabel('')
    plt.title(title)
    plt.show()


def comparison_plot(
    pdf,
    x='x', 
    y='y',
    metric='metric',
    label=True, 
    to_label=0.1, 
    yx_line=True,
    show=False,
    ):
    """
    Plot a comparison between two vectors
    Args:
        x (pd.Series): x values
        y (pd.Series): y values
        label (bool): whether to label the top % of points by metric
        to_label (float): if < 1, the percent of points to label, if > 1, the number of points to label
        yx_line (bool): whether to plot a y=x line
        show (bool): whether to show the plot
        metric (function): function to use to calculate metric between x and y
    """

    #calculate fit and residuals
    sns.scatterplot(data=pdf, x=x, y=y, hue=metric)

    #label top % by metric
    if to_label > 1:
        n = int(to_label)
    else:
        n = int(len(pdf) * to_label)
    topN = pdf.sort_values(metric, ascending=False).head(n)
    texts = []
    for i, row in topN.iterrows():
        texts.append(plt.text(row[x], row[y], i, fontsize=10))
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color='k', lw=0.5))

    #add y = x line for min and max

    mn, mx = min(pdf[x].min(), pdf[y].min()), max(pdf[x].max(), pdf[y].max())
    plt.plot([mn, mx], [mn, mx], color='red', linestyle='--')
    
    if show:
        plt.show()

def plot_kd(adata, gene, ref_val, exp_val, col='perturbation'):
    gene_vals = adata[:,gene].X.toarray().flatten()
    ##plot AR for AR KD vs NTC|NTC
    gene_inds = adata.obs[col] == exp_val
    NTC_inds = adata.obs[col] == ref_val
    print(f"Number of obs in NTC: {np.sum(NTC_inds)}")
    print(f"Number of obs in {gene} KD: {np.sum(gene_inds)}")


    plt.hist(gene_vals[NTC_inds], label=ref_val, alpha=0.5, bins=30)
    plt.hist(gene_vals[gene_inds], label=exp_val + ' KD', alpha=0.5, bins=30)
    #add mean line for each group
    plt.axvline(gene_vals[NTC_inds].mean(), color='blue')
    plt.axvline(gene_vals[gene_inds].mean(), color='orange')
    plt.title(f'{exp_val} KD vs {ref_val} for gene {gene}')
    plt.legend()
    plt.show()


from scipy.stats import pearsonr
import matplotlib.pyplot as plt 

def corrfunc(x, y, ax=None, method='spearman', **kws):
    """Plot the correlation coefficient in the top left hand corner of a plot."""
    func = spearmanr if method == 'spearman' else pearsonr
    r, _ = func(x, y, nan_policy='omit')
    ax = ax or plt.gca()
    ax.annotate(f'ρ = {r:.2f}', xy=(.1, .9), xycoords=ax.transAxes)

def square_plot(x,y, ax=None, show=True, corr=None, **kwargs):
    """
    Plot a square plot of x vs y with a y=x line
    Args:
        x (pd.Series): x values
        y (pd.Series): y values
        ax (matplotlib.axes.Axes): axis to plot on
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))
    sns.scatterplot(x=x, y=y, ax=ax, **kwargs)
    #add y = x line for min and max
    # ax[i].plot([0,1], [0,1], color='red', linestyle='--')
    #get min and max values

    if corr == 'spearman':
        corr = spearmanr(x,y, nan_policy='omit')[0]
    elif corr == 'pearson':
        corr = pearsonr(x,y)[0]
    
    if corr is not None:
        #put correlation bottom right
        ax.text(0.8, 0.1, f"r={round(corr,2)}", transform=ax.transAxes)


    min_val = min(x.min(), y.min())
    max_val = max(x.max(), y.max())
    ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    if show:
        plt.show()




### guide plotting
##########################################################################



## plot ratio of top 2 against counts: 
def plot_top2ratio_counts(features, show=False):
        #plot QC for guide metrics
    g = sns.jointplot(data = features.obs, y = 'log10_total_feature_counts', x = 'log2_ratio_2nd_1st_feature', kind = 'hex')
    g.ax_joint.set_ylabel('log10(total counts/cell)')
    g.ax_joint.set_xlabel('log2(2nd top feature / top feature)')
    g.fig.suptitle('Cell level metrics')
    g.fig.tight_layout()
    if show:
        plt.show()

    return g
    

#plot guide call proportions
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

#plot guide call numbers as proportion of all cells
        ## plot num features
def plot_num_features(features, show=False, ax =None, **kwargs):
    vc = features.obs['num_features'].value_counts()
    vc = vc.sort_index()
    vc = vc/vc.sum() * 100

    if ax is None:
        fig, ax = plt.subplots(1,1)
    sns.barplot(data=vc.reset_index(),x='num_features', y='count', ax=ax, **kwargs)
    ax.set_xlabel('# feature calls per cell')
    ax.set_ylabel('% of all cells')
    if show:
        plt.show()
    
    return ax