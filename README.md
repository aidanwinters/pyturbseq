# pyturbseq

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Aidan Winters
**Email:** aidanw@arcinstitute.org

## Description

`pyturbseq` is a comprehensive Python package for processing and analysis of single-cell perturbation data, with a particular focus on CRISPR-based perturbation screens. The package provides tools for:

- **Perturbation calling and quality control**: Identify and validate perturbations in single cells
- **Differential expression analysis**: Compare gene expression between perturbed and control cells
- **Interaction analysis**: Detect and quantify genetic interactions in dual perturbation experiments
- **Visualization**: Generate publication-ready plots for perturbation screen analysis
- **Data processing**: Handle large-scale single-cell datasets with efficient algorithms

## Key Features

- **Multi-perturbation support**: Handle single and dual perturbation experiments
- **Flexible data input**: Compatible with standard single-cell formats (AnnData, h5ad)
- **Statistical analysis**: Robust statistical methods for differential expression and interaction detection
- **Scalable processing**: Efficient algorithms for large datasets
- **Comprehensive visualization**: Rich plotting functions for data exploration and publication

## Installation

### Requirements
- Python 3.9 or higher
- Tested extensively with Python 3.9-3.13

### Install from PyPI (recommended)

```bash
pip install pyturbseq
```

### Install from source

```bash
git clone https://github.com/aidanwinters/pyturbseq.git
cd pyturbseq
pip install -e .
```

### Development installation

For development with testing and linting tools:

```bash
pip install -e ".[dev,test]"
```

## Quick Start

```python
import pyturbseq as prtb
import scanpy as sc

# Load your single-cell perturbation data
adata = sc.read_h5ad("your_perturbation_data.h5ad")

# Generate perturbation matrix
prtb.utils.get_perturbation_matrix(adata, perturbation_col='feature_call')

# Calculate target gene changes
prtb.utils.calculate_target_change(adata, perturbation_column='feature_call')

# Perform differential expression analysis
# Single comparison: compare specific perturbation vs control
deg_results = prtb.de.get_degs(adata, design_col='feature_call', control_value='NTC')

# Multiple comparisons: test all perturbations vs control
all_deg_results = prtb.de.get_all_degs(adata, design_col='feature_call', control_value='NTC')

# Analyze genetic interactions (for dual perturbation data)
# Single interaction: analyze specific dual perturbation
result, prediction = prtb.interaction.norman_model(data, 'GENE1|GENE2')

# Multiple interactions: analyze specific dual perturbations
interaction_results = prtb.interaction.norman_model(data, dual_perturbation_list)

# Auto-detect and analyze all dual perturbations (default behavior)
interaction_results = prtb.interaction.norman_model(data)

# Parallel processing for large datasets
interaction_results = prtb.interaction.norman_model(data, parallel=True, processes=8)

# Generate visualizations
prtb.plot.target_change_heatmap(adata, perturbation_column='feature_call')
```

## Main Modules

- **`utils`**: Core utilities for data processing and perturbation matrix generation
- **`de`**: Differential expression analysis tools
- **`interaction`**: Genetic interaction analysis for dual perturbation experiments
- **`plot`**: Visualization functions for perturbation screen data
- **`calling`**: Perturbation calling and quality control
- **`cellranger`**: Integration with Cell Ranger outputs

## Documentation

For detailed documentation and tutorials, see the included `Tutorial.ipynb` notebook which demonstrates:
- Data loading and preprocessing
- Perturbation calling
- Differential expression analysis
- Interaction analysis
- Visualization workflows

## Citation

If you use pyturbseq in your research, please cite:

```
Winters, A. (2024). pyturbseq: A Python package for perturbational single-cell data analysis.
```

## Support

For questions and support:
- Open an issue on [GitHub](https://github.com/aidanwinters/pyturbseq/issues)
- Email: aidanw@arcinstitute.org

## License

This project is licensed under the MIT License - see the LICENSE file for details.
