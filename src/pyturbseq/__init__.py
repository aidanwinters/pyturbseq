"""
pyturbseq - A Python package for perturbational single-cell data analysis

This package provides tools for:
- Perturbation calling and quality control
- Differential expression analysis
- Interaction analysis for dual perturbation experiments
- Visualization of perturbation screen data
- Data processing for large-scale single-cell datasets
"""

__version__ = "0.1.0"
__author__ = "Aidan Winters"
__email__ = "aidanw@arcinstitute.org"

# Version checking for dependencies
import sys

# Import main modules
from . import calling, cellranger, de, guides, interaction, plot, utils

if sys.version_info < (3, 9):
    raise ImportError("pyturbseq requires Python 3.9 or higher")

from .de import get_all_degs, get_degs
from .interaction import norman_model
from .plot import plot_label_similarity, target_change_heatmap, target_gene_heatmap

# Make key functions easily accessible
from .utils import (
    calculate_target_change,
    filter_adata,
    generate_perturbation_matrix,
    get_perturbation_matrix,
    pseudobulk,
    subsample_on_covariate,
)

__all__ = [
    # Modules
    "utils",
    "de",
    "interaction",
    "plot",
    "calling",
    "cellranger",
    "guides",
    # Key functions
    "generate_perturbation_matrix",
    "get_perturbation_matrix",
    "calculate_target_change",
    "filter_adata",
    "subsample_on_covariate",
    "pseudobulk",
    "get_degs",
    "get_all_degs",
    "norman_model",
    "target_change_heatmap",
    "target_gene_heatmap",
    "plot_label_similarity",
]
