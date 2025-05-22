# CorrEnrich

**CorrEnrich** (Correlation Enrichment) is a Python toolkit for identifying and evaluating biologically meaningful gene clusters by analyzing their expression tightness, functional enrichment (via GO terms), and statistical significance under various experimental conditions.

---

## 🧠 Purpose

This codebase supports the analysis of gene expression data to:

- Build a gene ontology (GO) tree branch of biological processes.
- Map genes to GO terms using Ensembl annotations.
- Calculate statistical properties of gene sets such as:
  - Average pairwise correlation
  - Expression tightness (standard deviation)
  - Mann–Whitney U and t-tests
  - Enrichment significance (hypergeometric p-values)
- Simulate random gene clusters to estimate null distributions.
- Apply False Discovery Rate (FDR) corrections for multiple hypotheses.
- Output structured tables and plots to assess GO term relevance.

---

## 📂 Project Structure

```
CorrEnrich.py
├── ./Data/                       # Raw gene expression data and metadata
├── ./Private/                   # Processed outputs, GO cluster metrics
│   ├── clusters_properties/     # Top GO clusters with significant enrichment
│   ├── random_tightness/        # Plots of correlation vs. group size
│   ├── data process/            # Intermediate files (z-scored, imputed)
├── go-basic.obo                 # Gene Ontology DAG (downloaded automatically)
```

---

## ⚙️ Key Features

### 🧬 Data Processing
- Reads transcriptomic data and metadata.
- Handles merging datasets, filtering low-quality samples, and removing mitochondrial genes.
- Normalizes expression, imputes missing values, and applies log2+z-score transformation.

### 🌲 Ontology Construction
- Parses GO DAG from `go-basic.obo`.
- Builds a GO tree rooted at “biological_process” (GO:0008150).
- Maps Ensembl genes to GO terms using BioMart.

### 📊 Statistical Analysis
- Correlation and variance of gene expression per GO term.
- Randomization tests for significance of observed tightness/correlation.
- FDR-corrected p-values for:
  - Hypergeometric enrichment
  - ECDF-based tightness
  - Median-based t-tests

### 📈 Visualization
- Generates plots of average pairwise correlation vs. gene group size (with error bars).
- Outputs TSV files with ranked GO terms per treatment and condition.

---

## 📦 Requirements

Install dependencies via pip:

```bash
pip install numpy pandas scipy matplotlib seaborn wget anytree goatools statsmodels biomart
```

---

## 🚀 Usage

Run from command line with a suffix to differentiate runs:

```bash
python CorrEnrich.py <experiment_suffix>
```

Outputs are saved in `./Private/` with the run type appended to filenames (e.g., `transformed_data_experiment_suffix.csv`).

---

## 📑 Main Functions

| Function | Description |
|---------|-------------|
| `read_process_files()` | Loads and normalizes raw data, merges large datasets. |
| `transform_data()` | Replaces missing values, log-transforms, z-scores data. |
| `impute_zeros()` | Fills in missing values with group-wise mean/median. |
| `build_tree()` | Builds/loads the GO term tree for biological processes. |
| `calculate_correlation()` | Evaluates expression correlation/tightness per GO cluster. |
| `get_random_corr()` | Generates null distributions for tightness tests. |

---

## 📝 Notes

- Default configuration focuses on antibiotics (`Amp`, `Met`, `Neo`, `Van`, `Mix`) and treatments (`IP`, `IV`, `PO`).
- Mitochondrial genes are excluded by default.
- Requires `metadata.xlsx` and gene count matrices in `./Data/`.

---

## 📜 License

See [LICENSE](LICENSE) for details.