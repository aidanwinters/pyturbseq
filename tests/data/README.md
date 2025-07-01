# Test Datasets

This directory contains small real datasets used for integration testing of pyturbseq.

## Datasets

### `norman2019_subset_singles.h5ad.gz`
- **Source**: Norman & Weissman 2019
- **Content**: Single perturbation CRISPR screen data
- **Perturbations**:
  - `control` (500 cells)
  - Single gene knockouts: `SET`, `KLF1`, `SAMD1`, `PTPN12`, `COL2A1`
- **Use**: Testing basic perturbation analysis workflows

### `norman2019_subset_doubles.h5ad.gz`
- **Source**: Norman & Weissman 2019
- **Content**: Single and dual perturbation CRISPR screen data
- **Perturbations**:
  - `control` (500 cells)
  - Single gene knockouts: `SET`, `KLF1`, `SAMD1`, `PTPN12`, `COL2A1`
  - Dual gene knockouts: `SET_KLF1`, `SAMD1_PTPN12`, `KLF1_COL2A1`
- **Use**: Testing interaction analysis and dual perturbation workflows

## Data Structure

Both datasets use the Norman2019 format:
- **Perturbation column**: `adata.obs['perturbation']`
- **Dual perturbation separator**: `_` (underscore)
- **Control label**: `control`
- **Gene expression**: Raw UMI counts in `adata.X`

## Generation

These datasets were generated from the full Norman & Weissman 2019 dataset using `tests/prepare_test_data.py`. The script:
1. Filters to specific perturbations and experimental conditions
2. Subsamples control cells to 500
3. Selects top 1000 highly variable genes + target genes
4. Preserves raw count data for realistic testing

## Citation

Norman, T.M., Horlbeck, M.A., Replogle, J.M. et al. Exploring genetic interaction manifolds constructed from rich single-cell phenotypes. Science 365, 786-793 (2019).
