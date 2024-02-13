##########################################################################
# 
# Functions for plotting and visualizing data
#
##########################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from adjustText import adjust_text
import numpy as np
import scanpy as sc
from scipy.stats import spearmanr, pearsonr

from .utils import cluster_df
from .interaction import get_singles, get_model_fit


from matplotlib.patches import Patch

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
        corr = pearsonr(x,y, nan_policy='omit')[0]
    
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

#plot guide call proportions
def plot_feature_count_metrics(features, ntc_var=None, show=False, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1,1)

    sns.scatterplot(data = features.var, y = 'pct_cells_with_guide', x = 'log10_total_counts', hue=ntc_var, ax=ax)
    # uniform coverage expectation
    if ntc_var in features.var.columns:
        unif = (1/features.var.query(f'{ntc_var} == False').shape[0])/2
        ax.axhline(unif, color='r', linestyle='--')
    #add label to axhline
        xmax = ax.get_xlim()[1]
        ax.text(xmax, unif, 'Expected %', ha='right', va='bottom', color='r')
        ax.legend(title='Negative Control')
    ax.set_xlabel('log10(total counts/guide)')
    ax.set_ylabel('% cells with guide')
    ax.set_ylim(0,0.2)
    #change y ticks to be multiplied by 100
    ax.set_yticks(plt.yticks()[0], [f'{int(x*100)}%' for x in plt.yticks()[0]])
    if show: 
        plt.show()

    return ax



## plot ratio of top 2 against counts: 
def plot_top2ratio_counts(features, show=False):
        #plot QC for guide metrics

    g = sns.jointplot(data = features.obs, y = 'log10_total_counts', x = 'log2_ratio_2nd_1st', kind = 'hex')
    g.ax_joint.set_ylabel('log10(total counts/cell)')
    g.ax_joint.set_xlabel('log2(2nd top sgRNA / top sgRNA)')
    g.fig.suptitle('Cell level metrics')
    g.fig.tight_layout()
    if show:
        plt.show()

    return g
    

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