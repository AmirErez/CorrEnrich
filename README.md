# CorrEnrich: Gene Ontology-based Expression Analysis

ClusteringGO is a Python package designed for bioinformatics analysis to identify and analyze clusters of co-regulated genes based on their Gene Ontology (GO) terms. It evaluates gene clusters for both internal correlation and significant differential expression between experimental conditions.

### Features

* **Data Processing**: A pipeline to read, normalize (TPM/RPM), filter, and transform gene expression data (imputation, log2, z-scoring).

* **GO Tree Construction**: Automatically downloads the Gene Ontology and builds a hierarchical tree of GO terms.

* **Gene Mapping**: Maps genes from expression data to their corresponding GO terms using Ensembl BioMart.

* **Flexible Statistical Analysis**:

    * Analyzes conditions based on user-specified columns in the metadata (e.g., 'Drug', 'Treatment').
    * Calculates average pairwise Spearman correlation within gene sets. 
    * Compares cluster correlation against a random background distribution. 
    * Performs Mann-Whitney U tests to assess differential expression of gene clusters.

* **Automated Analysis**: A main script to run the end-to-end analysis pipeline across multiple conditions.

* **Visualization**: Generates plots showing the relationship between gene set size and random correlation.

### Installation

To install the package, clone this repository and install it using pip:
```
git clone <repository-url>
cd clusteringgo-package
pip install .
```

Alternatively, for development, install in editable mode:
```
pip install -e .
```

### Usage

The main analysis can be run from the `run_analysis`.py script. Before running, you must update the `DATA_DIRECTORY` variable in the script to point to your data folder.

The expected data directory structure is:

```
<DATA_DIRECTORY>/
 |- metadata.xlsx
 |- RASflow stats 2023_09_17.csv
 |- new normalization/
     |- transcriptome_2023-09-17-genes_norm_named.tsv
```

### Command-Line Arguments

* `data_dir`: Path to your root data directory.

* `output_dir`: Path where results will be saved.

* `--primary_col`: The main metadata column for comparison (e.g., `Drug`).

* `--control_val`: The value in the primary column to use as the control group (e.g., `PBS`).

* `--secondary_col` (Optional): A second metadata column for a nested, or two-factor, analysis (e.g., `Treatment`).

### Example Commands

**1. Simple Analysis (One-Factor)**

Compare all values in the `Drug` column against the PBS control.

```
python run_analysis.py path/to/data/ path/to/output/ --primary_col Drug --control_val PBS
```

**2. Nested Analysis (Two-Factor)**

Compare all values in the `Drug` column against `PBS`, but do so separately for each value in the `Treatment` column.

```
python run_analysis.py path/to/data/ path/to/output/ --primary_col Drug --control_val PBS --secondary_col Treatment
```

Results, including TSV files with statistics for each GO term and diagnostic plots, will be saved in the specified output directory.


### Package Structure

`clusteringgo/`: The main package source code.

`data_processing.py`: Functions for data loading, cleaning, and transformation.

`tree.py`: Code for building the GO tree and mapping genes.

`stats.py`: Statistical tests and correlation calculations.

`utils.py`: Helper functions for plotting and saving results.

`tests/`: Unit tests for the package.

`run_analysis.py`: Example script to execute the full pipeline.

`setup.py`: Package installation script.
