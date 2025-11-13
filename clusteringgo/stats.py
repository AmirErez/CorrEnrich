"""
Contains functions for statistical analysis, including correlation and significance testing.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, ttest_ind, hypergeom
from scipy.stats.mstats import gmean

from statsmodels.stats.multitest import fdrcorrection

from joblib import Parallel, delayed  # <-- IMPORT FOR PARALLELIZATION


def get_abx_pbs_data(primary_value, control_value, gene, expression_data, metadata,
                     primary_col, secondary_col=None, secondary_val=None):
    """
    Splits expression data for a gene into a primary group and a control group.
    """
    abx_query = f"`{primary_col}` == '{primary_value}'"
    pbs_query = f"`{primary_col}` == '{control_value}'"

    if secondary_col and secondary_val is not None:
        abx_query += f" and `{secondary_col}` == '{secondary_val}'"
        pbs_query += f" and `{secondary_col}` == '{secondary_val}'"

    abx_meta = metadata.query(abx_query)
    pbs_meta = metadata.query(pbs_query)

    abx_samples = [s for s in abx_meta['ID'] if s in expression_data.columns]
    pbs_samples = [s for s in pbs_meta['ID'] if s in expression_data.columns]

    row = expression_data.loc[gene]

    abx_expression = row[abx_samples].dropna()
    pbs_expression = row[pbs_samples].dropna()

    return abx_expression, pbs_expression


def average_pairwise_spearman(gene_data):
    """
    Calculates the average pairwise Spearman correlation for a set of genes.
    """
    if gene_data.shape[0] < 2:
        return np.nan
    corr_matrix = gene_data.T.corr(method='spearman')
    upper_tri_indices = np.triu_indices(corr_matrix.shape[0], k=1)
    if not upper_tri_indices[0].size:
        return np.nan
    corr_values = corr_matrix.values[upper_tri_indices]
    return np.nanmean(corr_values)


# def get_random_corr(size, df, times=1000):
#     """
#     Generates a distribution of correlations from random gene sets.
#     """
#     if size > df.shape[0]:
#         return np.nan, np.nan, {}
#
#     sample_indices = [np.random.choice(df.shape[0], size=size, replace=False) for _ in range(times)]
#     random_dist = [average_pairwise_spearman(df.iloc[indices]) for indices in sample_indices]
#     random_dist = [x for x in random_dist if not np.isnan(x)]
#
#     if not random_dist:
#         return np.nan, np.nan, {}
#
#     ecdf_data = save_ecdf_efficient(np.array(random_dist))
#     return np.mean(random_dist), np.std(random_dist), ecdf_data
def _calculate_random_corr_sample(gene_indices, df):
    """
    Helper function for parallel processing in get_random_corr.
    Calculates avg spearman correlation for a single sample.

    Args:
        gene_indices (np.array): An array of integer indices for slicing df.
        df (pd.DataFrame): The expression dataframe (genes x samples).

    Returns:
        float: The average pairwise spearman correlation.
    """
    # Use .iloc for fast integer-based row slicing
    return average_pairwise_spearman(df.iloc[gene_indices])


def get_random_corr(size, df, times=1000):
    """
    Calculates the random correlation distribution in parallel.

    Args:
        size (int): The number of genes to sample.
        df (pd.DataFrame): The expression dataframe.
        times (int): The number of permutations.

    Returns:
        tuple: (mean, std, ecdf_data)
    """
    # Get integer indices for faster slicing
    all_gene_indices = np.arange(df.shape[0])

    # Create a list of all random index sets
    sample_indices_list = [
        np.random.choice(all_gene_indices, size=size, replace=False)
        for _ in range(times)
    ]

    # Run the correlation calculation in parallel
    # n_jobs=-1 uses all available CPU cores
    # This will be significantly faster than the old loop
    random_dist_list = Parallel(n_jobs=-1)(
        delayed(_calculate_random_corr_sample)(indices, df)
        for indices in sample_indices_list
    )

    random_dist = np.array(random_dist_list)

    ecdf_data = save_ecdf_efficient(random_dist, tail_threshold=0.05, mid_step=0.05)

    return random_dist.mean(), random_dist.std(), ecdf_data


def save_ecdf_efficient(bootstrap_results, tail_threshold=0.05, mid_step=0.05):
    """
    Efficiently stores an ECDF for p-value calculation.
    """
    sorted_data = np.sort(bootstrap_results)
    n = len(sorted_data)
    ecdf_values = np.arange(1, n + 1) / n

    return {'data': sorted_data, 'ecdf': ecdf_values}


def calculate_pvalue_from_ecdf(observed_value, ecdf_data, tail='two-sided'):
    """
    Calculates a p-value for an observed value from a stored ECDF.
    """
    data = ecdf_data['data']
    ecdf = ecdf_data['ecdf']

    if not data.size:
        return 1.0

    # Find position of observed value
    idx = np.searchsorted(data, observed_value, side='right')

    # Correct for edges
    if idx == 0:
        p_lower = 1 / (2 * len(data))
    else:
        p_lower = ecdf[idx - 1]

    p_upper = 1.0 - p_lower

    if tail == 'lower':
        return p_lower
    elif tail == 'upper':
        return p_upper
    else:  # two-sided
        return 2 * min(p_lower, p_upper)


def median_mwu(primary_value, control_value, genes_data, expression, meta,
               primary_col, secondary_col=None, secondary_val=None):
    """
    Performs a Mann-Whitney U test on the median expression of a gene set.
    """
    genes_in_expression = [g for g in genes_data if g in expression.index]
    if not genes_in_expression:
        return np.nan, 1.0

    median_expression = expression.loc[genes_in_expression].median()

    abx_query = f"`{primary_col}` == '{primary_value}'"
    pbs_query = f"`{primary_col}` == '{control_value}'"

    if secondary_col and secondary_val is not None:
        abx_query += f" and `{secondary_col}` == '{secondary_val}'"
        pbs_query += f" and `{secondary_col}` == '{secondary_val}'"

    abx_meta = meta.query(abx_query)
    pbs_meta = meta.query(pbs_query)

    abx_samples = [s for s in abx_meta['ID'] if s in median_expression.index]
    pbs_samples = [s for s in pbs_meta['ID'] if s in median_expression.index]

    abx_vals = median_expression[abx_samples].dropna()
    pbs_vals = median_expression[pbs_samples].dropna()

    if abx_vals.empty or pbs_vals.empty or abx_vals.equals(pbs_vals):
        return np.nan, 1.0

    try:
        stat, p_val = mannwhitneyu(abx_vals, pbs_vals, alternative='two-sided')
        return stat, p_val
    except ValueError:
        return np.nan, 1.0


# def split_genes_by_trend(primary_value, control_value, genes_data, expression, meta,
#                          primary_col, secondary_col=None, secondary_val=None):
#     """
#     Splits genes into enhanced or suppressed based on expression trend.
#     """
#     enhanced, suppressed = set(), set()
#     for gene in genes_data:
#         abx, pbs = get_abx_pbs_data(primary_value, control_value, gene, expression, meta,
#                                     primary_col, secondary_col, secondary_val)
#         if abx.mean() > pbs.mean():
#             enhanced.add(gene)
#         else:
#             suppressed.add(gene)
#     return list(enhanced), list(suppressed)
def calculate_hypergeometric_pvalue(N, K, n, k):
    """
    Calculate the hypergeometric p-value.
    (Copied from ClusteringGO.py)

    Parameters:
    N : int
        Total number of genes
    K : int
        Total number of significant genes
    n : int
        Number of genes in the GO term
    k : int
        Number of significant genes in the GO term

    Returns:
    float
        The p-value (survival function)
    """
    # Calculate the probability of getting k or more successes
    # Using survival function (1 - cdf) is more accurate for upper tail
    return hypergeom.sf(k - 1, N, K, n)


# def genes_data_split(primary_value, control_value, genes_data, expression, meta,
#                      primary_col, secondary_col=None, secondary_val=None, threshold=0.05):
#     """
#     Splits genes into enhanced or suppressed based on expression trend AND
#     returns a dictionary of genes that are individually significant.
#     (Based on ClusteringGO.py logic)
#     """
#     enhanced, suppressed = set(), set()
#     significant_genes = {}
#
#     for gene in genes_data:
#         if gene not in expression.index:
#             continue
#
#         abx, pbs = get_abx_pbs_data(primary_value, control_value, gene, expression, meta,
#                                     primary_col, secondary_col, secondary_val)
#
#         # Skip if no data for comparison
#         if abx.empty or pbs.empty:
#             continue
#
#         try:
#             # Perform t-test
#             t_stat, p_val = ttest_ind(abx, pbs)
#
#             if pd.isna(t_stat):
#                 continue
#
#             # Split by trend
#             if t_stat > 0:  # meaning the abx is enhanced
#                 enhanced.add(gene)
#             else:
#                 suppressed.add(gene)
#
#             # Check for significance
#             if p_val < threshold:
#                 significant_genes[gene] = p_val
#
#         except ValueError:
#             # Catches errors from t-test (e.g., no variance)
#             continue
#
#     return list(enhanced), list(suppressed), significant_genes




def genes_data_split(primary_value, control_value, genes_data, expression, meta,
                                primary_col, secondary_col=None, secondary_val=None, threshold=0.05):
    """
    Splits genes into enhanced or suppressed based on expression trend AND
    returns a dictionary of genes that are individually significant.
    (Vectorized for efficiency)
    """

    # 1. Identify control (pbs) and primary (abx) sample groups ONCE
    # Find sample IDs from the metadata
    pbs_mask = (meta[primary_col] == control_value)
    abx_mask = (meta[primary_col] == primary_value)

    if secondary_col and secondary_val is not None:
        secondary_mask = (meta[secondary_col] == secondary_val)
        pbs_mask &= secondary_mask
        abx_mask &= secondary_mask

    # We use .loc[mask, 'ID'] which is equivalent to meta.query(...)[ID]
    pbs_samples = meta.loc[pbs_mask, 'ID']
    abx_samples = meta.loc[abx_mask, 'ID']

    # 2. Filter expression data ONCE

    # Find samples that exist in *both* the metadata and expression columns
    # We must convert expression.columns to a pd.Index for .intersection()
    valid_pbs_samples = pbs_samples[pbs_samples.isin(expression.columns)]
    valid_abx_samples = abx_samples[abx_samples.isin(expression.columns)]

    # Find genes that exist in *both* genes_data and the expression index
    expression_genes_index = pd.Index(expression.index)
    common_genes = expression_genes_index.intersection(genes_data)

    # Check for empty groups after filtering
    if valid_pbs_samples.empty or valid_abx_samples.empty or common_genes.empty:
        if valid_pbs_samples.empty:
            print("Warning: No valid control (pbs) samples found.")
        if valid_abx_samples.empty:
            print("Warning: No valid primary (abx) samples found.")
        if common_genes.empty:
            print("Warning: No common genes found.")
        return [], [], {}

    # Create the two final data matrices for comparison
    pbs_data = expression.loc[common_genes, valid_pbs_samples]
    abx_data = expression.loc[common_genes, valid_abx_samples]

    # 3. Perform t-test for all genes at once
    # axis=1 compares data along the rows (i.e., compares sample groups for each gene)
    # nan_policy='omit' handles NaNs within sample groups
    # equal_var=False performs Welch's T-test, which is generally safer
    try:
        t_stat, p_val = ttest_ind(abx_data, pbs_data, axis=1, nan_policy='omit', equal_var=True)
    except ValueError as e:
        print(f"T-test failed (e.g., all-NaN slice): {e}. Returning empty results.")
        return [], [], {}

    # 4. Combine results into a DataFrame for easy filtering
    results_df = pd.DataFrame({
        't_stat': t_stat,
        'p_val': p_val
    }, index=common_genes)

    # Drop genes where t-test failed (e.g., no variance in both groups)
    results_df = results_df.dropna()

    if results_df.empty:
        return [], [], {}

    # 5. Categorize genes using efficient boolean masking

    # Enhanced: t_stat > 0
    enhanced_mask = results_df['t_stat'] > 0
    enhanced = results_df.index[enhanced_mask].tolist()

    # Suppressed: t_stat <= 0
    suppressed = results_df.index[~enhanced_mask].tolist()

    # Significant
    significant_mask = results_df['p_val'] < threshold
    significant_genes_series = results_df.loc[significant_mask, 'p_val']
    significant_genes = significant_genes_series.to_dict()

    return enhanced, suppressed, significant_genes