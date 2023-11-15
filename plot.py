import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from adjustText import adjust_text

from . import processing as proc

def plot_double_single(data, double_condition, pred=False, genes=None, **kwargs):

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

    subdf = proc.cluster_df(sub, cluster_rows=False)
    # sub.obs
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))

    if pred is not None:
        #add pred to sub
        m, Z = get_model_fit(subdf, double_condition, targets=gs, plot=False)
        subdf.loc[f"Predicted",gs] = Z.flatten()
        title = f"{double_condition} \n{round(float(m['coef_a']),2)}({m['a']}) x {round(float(m['coef_b']),2)}({m['b']}) \nSpearman: {round(m['corr_fit'],2)}"
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