##########################################################################
# 
# Functions for Differential Expression testing
#
##########################################################################


# # from pydeseq2.dds import DeseqDataSet
# # from pydeseq2.ds import DeseqStats
# # import pandas as pd
# # import numpy as np

# # from statsmodels.stats.multitest import multipletests

# # from adjustText import adjust_text

# # def get_degs(adata, design_col, ref_val=None, n_cpus=16, quiet=True):
# #     """
# #     This function runs DESeq2 on an AnnData object. It returns a dataframe with the results of the DESeq2 analysis.
# #     Args:
# #         adata: AnnData object
# #         design_col: column in adata.obs that contains the design matrix
# #         ref_val: 
# #             reference value for the design matrix. If None, all values in design_col are used as contrasts and listed in alphabetical order. 
# #             This should be set to syncronize logFC values across multiple DEG runs or for convenience of interpreting. 
# #             When set, positive logFC values indicate that the reference value is higher than alternative value. More than 2 values in contrast is currently not supported. 
# #         n_cpus: number of cpus to use for DESeq2
# #         quiet: whether to print DESeq2 output
# #     """
# #     dds = DeseqDataSet(
# #         counts=pd.DataFrame(

# #             adata.X.toarray() if type(adata.X) is not np.ndarray else adata.X,
# #             index=adata.obs.index, columns=adata.var.index
# #         ),
# #         metadata=adata.obs,
# #         design_factors=design_col,
# #         n_cpus=n_cpus,
# #         quiet=quiet,
# #     )
# #     try:
# #         dds.deseq2()
# #     except Exception as e:
# #         print(f"Exception:" + str(e))
# #         return pd.DataFrame()

# #     if ref_val is None:
# #         contrast = [design_col] + list(adata.obs[design_col].unique()) 
# #     else:
# #         print(f"Using {ref_val} as reference value for {design_col}")
# #         design_vals = adata.obs[design_col].unique()
# #         alt_val = [x for x in design_vals if x != ref_val] #check to make sure that this is a single calue
# #         if len(alt_val) > 1:
# #             raise ValueError(f"More than one alternative value for {design_col} in adata. This is currently not supported.")
# #         else: 
# #             alt_val = alt_val[0]
# #         contrast = [design_col, ref_val, alt_val]
# #     stat_res = DeseqStats(
# #         dds, contrast=contrast , n_cpus=n_cpus, quiet=quiet
# #     )
# #     stat_res.summary()
# #     df = stat_res.results_df
# #     df['padj_bh'] = multipletests(df['pvalue'], method='fdr_bh')[1]
# #     return df

# # def get_all_degs(adata, design_col, reference, conditions=None, n_cpus=8):

# #     def get_degs_subset(condition):
# #         df = get_degs(
# #             adata[adata.obs[design_col].isin([condition, reference])],
# #             design_col,
# #             ref_val=reference,
# #             n_cpus=n_cpus
# #         )
# #         df['condition'] = condition
# #         return df
    
# #     if conditions is None: #get all conditions if not specified
# #         conditions = adata.obs[design_col].unique()
# #         #remove reference from conditions
# #         conditions = [x for x in conditions if x != reference]

# #     #get all dfs
# #     dfs = [get_degs_subset(condition) for condition in conditions]

# #     #concatenate
# #     df = pd.concat(dfs)
# #     return df

# from concurrent.futures import ThreadPoolExecutor, as_completed
# from tqdm import tqdm
# import pandas as pd
# from pydeseq2.dds import DeseqDataSet
# from pydeseq2.ds import DeseqStats
# from statsmodels.stats.multitest import multipletests
# import numpy as np

# import io
# import sys
# import multiprocessing

# def get_degs(adata, design_col, ref_val=None, n_cpus=16, quiet=True):
#     """
#     Run DESeq2 analysis on single-cell RNA sequencing data.

#     Parameters:
#     adata (AnnData): AnnData object containing the single-cell RNA-seq data.
#     design_col (str): Column name in adata.obs that contains the design matrix.
#     ref_val (str, optional): Reference value for the design matrix. Defaults to None.
#     n_cpus (int, optional): Number of CPUs to use for DESeq2. Defaults to 16.
#     quiet (bool, optional): Flag to suppress DESeq2 output. Defaults to True.
#     verbose (bool, optional): Flag to enable verbose output. Defaults to False.

#     Returns:
#     pd.DataFrame: DataFrame containing DESeq2 results.
#     """
#     if not quiet:
#         print(f"Running DESeq2 for design column: {design_col} with reference value: {ref_val}")

#     dds = DeseqDataSet(
#         counts=pd.DataFrame(
#             adata.X.toarray() if type(adata.X) is not np.ndarray else adata.X,
#             index=adata.obs.index, columns=adata.var.index
#         ),
#         metadata=adata.obs,
#         design_factors=design_col,
#         n_cpus=n_cpus,
#         quiet=quiet,
#     )

#     if not quiet:
#         print(f"DESeq2 object created for {design_col}.")

#     try:
#         # create a text trap and redirect stdout
#         if quiet:
#                     # create a text trap and redirect stdout
#             text_trap = io.StringIO()
#             sys.stdout = text_trap

#         # execute function
#         dds.deseq2()

#         if quiet:
#             # now restore stdout function
#             sys.stdout = sys.__stdout__

#     except Exception as e:
#         print(f"Exception in DESeq2 execution: {e}")
#         return pd.DataFrame()

#     # Setting up contrast for DESeq2
#     if ref_val is None:
#         contrast = [design_col] + list(adata.obs[design_col].unique())
#     else:
#         design_vals = adata.obs[design_col].unique()
#         alt_val = [x for x in design_vals if x != ref_val]
#         if len(alt_val) > 1:
#             raise ValueError(f"More than one alternative value for {design_col} in adata. This is currently not supported.")
#         contrast = [design_col, alt_val[0], ref_val]

#     # Running the statistical analysis

#     if quiet:
#                 # create a text trap and redirect stdout
#         text_trap = io.StringIO()
#         sys.stdout = text_trap

#     stat_res = DeseqStats(dds, contrast=contrast, n_cpus=n_cpus, quiet=quiet)
#     stat_res.summary()

#     if quiet:
#         # now restore stdout function
#         sys.stdout = sys.__stdout__

#     df = stat_res.results_df
#     df['padj_bh'] = multipletests(df['pvalue'], method='fdr_bh')[1]

#     if not quiet:
#         print(f"DESeq2 analysis completed for {design_col}.")

#     return df

# def get_all_degs(adata, design_col, reference, conditions=None, n_cpus=8, max_workers=4, quiet=False):
#     """
#     Run DESeq2 analysis in parallel for multiple conditions.

#     Parameters:
#     adata (AnnData): AnnData object containing the single-cell RNA-seq data.
#     design_col (str): Column name in adata.obs that contains the design matrix.
#     reference (str): Reference condition for the differential expression test.
#     conditions (list, optional): List of conditions to test against the reference. Defaults to None.
#     n_cpus (int, optional): Number of CPUs to use for each DESeq2 task. Defaults to 8.
#     max_workers (int, optional): Maximum number of parallel tasks. Defaults to 4.
#     verbose (bool, optional): Flag to enable verbose output. Defaults to False.

#     Returns:
#     pd.DataFrame: Concatenated DataFrame containing results for all conditions.
#     """
#     def get_degs_subset(condition):
#         if not quiet:
#             print(f"Processing condition: {condition}")
#         df = get_degs(
#             adata[adata.obs[design_col].isin([condition, reference])],
#             design_col,
#             ref_val=reference,
#             n_cpus=n_cpus,
#             quiet=quiet
#         )
#         df['condition'] = condition
#         return df

#     if n_cpus * max_workers > multiprocessing.cpu_count():
#         # raise ValueError(f"Number of CPUs ({n_cpus}) * max_workers ({max_workers}) exceeds number of available CPUs ({multiprocessing.cpu_count()}).")
#         #warn instead
#         print(f"Number of CPUs ({n_cpus}) * max_workers ({max_workers}) exceeds number of available CPUs ({multiprocessing.cpu_count()}).")
#         print(f"Setting n_cpus to {multiprocessing.cpu_count() // max_workers}")
#         n_cpus = multiprocessing.cpu_count() // max_workers

#     if conditions is None:
#         conditions = list(set(adata.obs[design_col]) - {reference})

#     if not quiet:
#         print(f"Starting parallel DESeq2 analysis for {len(conditions)} conditions.")

#     dfs = []
#     try:
#         with ThreadPoolExecutor(max_workers=max_workers) as executor:
#             # Mapping of futures to their respective conditions
#             future_to_condition = {executor.submit(get_degs_subset, condition): condition for condition in conditions}

#             # Initialize tqdm progress bar
#             progress = tqdm(as_completed(future_to_condition), total=len(conditions), desc="Processing", unit="task")

#             for future in progress:
#                 condition = future_to_condition[future]
#                 # Collect results and append to list
#                 dfs.append(future.result())

#     except KeyboardInterrupt:
#         print("Cancellation requested by user. Shutting down...")
#         executor.shutdown(wait=False)
#         raise

#     if not quiet:
#         print("All conditions processed. Concatenating results...")

#     return pd.concat(dfs)


from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import pandas as pd
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from statsmodels.stats.multitest import multipletests
import numpy as np
import multiprocessing

def get_degs(adata, design_col, ref_val=None, n_cpus=16, quiet=False):
    """
    Run DESeq2 analysis on single-cell RNA sequencing data.
    
    Parameters:
    adata (AnnData): AnnData object containing the single-cell RNA-seq data.
    design_col (str): Column name in adata.obs that contains the design matrix.
    ref_val (str, optional): Reference value for the design matrix. Defaults to None.
    n_cpus (int, optional): Number of CPUs to use for DESeq2. Defaults to 16.
    quiet (bool, optional): Flag to suppress DESeq2 output. Defaults to True.

    Returns:
    pd.DataFrame: DataFrame containing DESeq2 results.
    """
    dds = DeseqDataSet(
        counts=pd.DataFrame(
            adata.X.toarray() if type(adata.X) is not np.ndarray else adata.X,
            index=adata.obs.index, columns=adata.var.index
        ),
        metadata=adata.obs,
        design_factors=design_col,
        n_cpus=n_cpus,
        quiet=quiet,  # Passing the quiet argument
    )

    try:
        dds.deseq2()  # Passing the quiet argument
    except Exception as e:
        print(f"Exception in DESeq2 execution: {e}")
        return pd.DataFrame()

    # Setting up contrast for DESeq2
    if ref_val is None:
        contrast = [design_col] + list(adata.obs[design_col].unique())
    else:
        design_vals = adata.obs[design_col].unique()
        alt_val = [x for x in design_vals if x != ref_val]
        if len(alt_val) > 1:
            raise ValueError(f"More than one alternative value for {design_col} in adata. This is currently not supported.")
        contrast = [design_col, alt_val[0], ref_val]

    # Running the statistical analysis
    stat_res = DeseqStats(dds, contrast=contrast, n_cpus=n_cpus, quiet=quiet)
    stat_res.summary()

    df = stat_res.results_df
    df['padj_bh'] = multipletests(df['pvalue'], method='fdr_bh')[1]

    return df

# def get_degs_subset(adata,  condition,  **kwargs):
#     df = get_degs(
#         adata,
#         design_col,
#         ref_val=reference,
#         **kwargs
#     )
#     df['condition'] = condition
#     return df

def get_all_degs(adata, design_col, reference, conditions=None, n_cpus=8, max_workers=4, quiet=False):
    """
    Run DESeq2 analysis in parallel for multiple conditions.

    Parameters:
    adata (AnnData): AnnData object containing the single-cell RNA-seq data.
    design_col (str): Column name in adata.obs that contains the design matrix.
    reference (str): Reference condition for the differential expression test.
    conditions (list, optional): List of conditions to test against the reference. Defaults to None.
    n_cpus (int, optional): Number of CPUs to use for each DESeq2 task. Defaults to 8.
    max_workers (int, optional): Maximum number of parallel tasks. Defaults to 4.
    quiet (bool, optional): Flag to suppress DESeq2 output. Defaults to False.

    Returns:
    pd.DataFrame: Concatenated DataFrame containing results for all conditions.
    """
    def get_degs_subset(condition):
        df = get_degs(
            adata[adata.obs[design_col].isin([condition, reference])],
            design_col,
            ref_val=reference,
            n_cpus=n_cpus,
            quiet=quiet
        )
        df['condition'] = condition
        return df

    available_cpus = multiprocessing.cpu_count()
    n_cpus = min(n_cpus, available_cpus // max_workers)

    if conditions is None:
        conditions = list(set(adata.obs[design_col]) - {reference})

    dfs = []
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Mapping of futures to their respective conditions
            future_to_condition = {executor.submit(get_degs_subset, adata[adata.obs[design_col].isin([condition, reference])]): condition for condition in conditions}

            # Initialize tqdm progress bar
            progress = tqdm(as_completed(future_to_condition), total=len(conditions), desc="Processing", unit="task")

            for future in progress:
                condition = future_to_condition[future]
                # Collect results and append to list
                dfs.append(future.result())

    except KeyboardInterrupt:
        print("Cancellation requested by user. Shutting down...")
        executor.shutdown(wait=False)
        raise 

    return pd.concat(dfs)


def get_all_degs_synchronously(adata, design_col, reference, conditions=None, n_cpus=8):

    def get_degs_subset(condition):
        df = get_degs(
            adata[adata.obs[design_col].isin([condition, reference])],
            design_col,
            ref_val=reference,
            n_cpus=n_cpus
        )
        df['condition'] = condition
        return df
    
    if conditions is None: #get all conditions if not specified
        conditions = adata.obs[design_col].unique()
        #remove reference from conditions
        conditions = [x for x in conditions if x != reference]

    #get all dfs
    dfs = [get_degs_subset(condition) for condition in conditions]

    #concatenate
    df = pd.concat(dfs)
    return df
