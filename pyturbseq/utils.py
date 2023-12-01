import anndata as ad
import pandas as pd

def pct_target_change(adata, perturbation_column, reference_label):
    """
    Compute the "percent change" for each cell against a reference.
    
    Parameters:
    - adata: anndata.AnnData object containing expression data. assumed to be transformed as desired
    - perturbation_column: column in adata.obs indicating the perturbation/knockdown
    - reference_label: label of the reference population in perturbation_column
    
    Returns:
    - An AnnData object with an additional column in obs containing the "percent knocked down" for each cell.
    """
    
    # Get mean expression values for reference population
    reference_means = adata[adata.obs[perturbation_column] == reference_label, :].X.toarray().mean(axis=0)
    
    # Placeholder for percent knocked down values
    percent_kd_values = []
    reference_mean_values = []
    
    # Loop through cells
    for cell, perturb in zip(adata.X.toarray(), adata.obs[perturbation_column]):
        if perturb == reference_label or perturb not in adata.var_names:
            percent_kd_values.append(None)
            reference_mean_values.append(None)
            continue

        # print('here')
        gene_idx = adata.var_names.get_loc(perturb)
        
        percent_value = ((cell[gene_idx]+1) / (reference_means[gene_idx] + 1)) * 100
        percent_kd_values.append(percent_value)
        reference_mean_values.append(reference_means[gene_idx])
    
    # Add to adata
    adata.obs['pct_change'] = percent_kd_values
    adata.obs['reference_mean'] = reference_mean_values

    return adata
