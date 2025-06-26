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

# Import main modules
from . import utils
from . import de
from . import interaction
from . import plot
from . import calling
from . import cellranger
from . import guides

# Version checking for dependencies
import sys
if sys.version_info < (3, 9):
    raise ImportError("pyturbseq requires Python 3.9 or higher")

# Make key functions easily accessible
from .utils import (
    generate_perturbation_matrix,
    get_perturbation_matrix,
    calculate_target_change,
    filter_adata,
    subsample_on_covariate,
    pseudobulk
)

from .de import (
    get_degs,
    get_all_degs
)

from .interaction import (
    get_model_fit,
    fit_many,
    get_model
)

from .plot import (
    target_change_heatmap,
    target_gene_heatmap,
    plot_label_similarity
)

__all__ = [
    # Modules
    "utils", "de", "interaction", "plot", "calling", "cellranger", "guides",
    # Key functions
    "generate_perturbation_matrix", "get_perturbation_matrix", "calculate_target_change",
    "filter_adata", "subsample_on_covariate", "pseudobulk",
    "get_degs", "get_all_degs",
    "get_model_fit", "fit_many", "get_model",
    "target_change_heatmap", "target_gene_heatmap", "plot_label_similarity"
]