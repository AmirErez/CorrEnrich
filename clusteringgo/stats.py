"""
Contains functions for statistical analysis, including correlation and significance testing.
"""
import numpy as np
from scipy.stats import mannwhitneyu
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


def split_genes_by_trend(primary_value, control_value, genes_data, expression, meta,
                         primary_col, secondary_col=None, secondary_val=None):
    """
    Splits genes into enhanced or suppressed based on expression trend.
    """
    enhanced, suppressed = set(), set()
    for gene in genes_data:
        abx, pbs = get_abx_pbs_data(primary_value, control_value, gene, expression, meta,
                                    primary_col, secondary_col, secondary_val)
        if abx.mean() > pbs.mean():
            enhanced.add(gene)
        else:
            suppressed.add(gene)
    return list(enhanced), list(suppressed)
