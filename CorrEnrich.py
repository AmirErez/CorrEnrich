import json
import os
import sys
import pickle
from collections import defaultdict, deque
from json import JSONEncoder
from typing import Dict, Set, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import wget
from anytree import NodeMixin, PostOrderIter
from anytree.importer import JsonImporter
from goatools import obo_parser
from scipy import stats
from scipy.stats import mannwhitneyu, ttest_ind
from scipy.stats.mstats import gmean
from statsmodels.stats.multitest import fdrcorrection

treatments = ['IP', 'IV', 'PO']
antibiotics = ['Amp', 'Met', 'Neo', 'Van', 'Mix']

mitochondrial_genes_translation = {
    "ND1": "mt-nd1",
    "ND2": "mt-nd2",
    "ND3": "mt-nd3",
    "ND4": "mt-nd4",
    "ND4L": "mt-nd4l",
    "ND5": "mt-nd5",
    "ND6": "mt-nd6",
    "COX1": "mt-co1",
    "COX2": "mt-co2",
    "COX3": "mt-co3",
    "CYTB": "mt-cytb",
    "ATP6": "mt-atp6",
    "ATP8": "mt-atp8",
    "tRNA-F": "mt-TF",
    "tRNA-V": "mt-TV",
    "tRNA-L": "mt-TL1",
    "tRNA-I": "mt-TI",
    "tRNA-Q": "mt-TQ",
    "tRNA-M": "mt-TM",
    "tRNA-W": "mt-TW",
    "tRNA-A": "mt-TA",
    "tRNA-N": "mt-TN",
    "tRNA-C": "mt-TC",
    "tRNA-Y": "mt-TY",
    "tRNA-S1": "mt-TS1",
    "tRNA-D": "mt-TD",
    "tRNA-K": "mt-TK",
    "tRNA-G": "mt-TG",
    "tRNA-R": "mt-TR",
    "tRNA-H": "mt-TH",
    "tRNA-S2": "mt-TS2",
    "tRNA-L2": "mt-TL2",
    "tRNA-E": "mt-TE",
    "tRNA-T": "mt-TT",
    "tRNA-P": "mt-TP",
    "12S rRNA": "mt-rnr1",
    "16S rRNA": "mt-rnr2"
}

# save the values of the dictionary in a list
mitochondrial_genes = list(mitochondrial_genes_translation.values())

private = os.path.join(".", "Private")

directory = private
path = os.path.join(private, "clusters_properties\\")

# Note the use of ".." to go up a directory from the current location
data_folder = os.path.join(".", "Data")


class GeneNode(NodeMixin, JSONEncoder):  # Add Node feature
    def __init__(self, go_id, level, name, go_obj, parents=None, children=None):
        super().__init__()
        self.go_id = go_id
        self.level = level
        self.name = name
        category = list(get_ancestor(go_obj))
        self.category = category[0].name if len(category) else "biological process"
        # self.go_object = go_obj
        self.parents = parents if parents else set()
        self.parent = parents
        self.children = children if children else set()
        # self.all_children = children if children else set()
        self.gene_set = set()
        self.pearson_corr = None
        self.spearman_corr = None
        self.dist = np.inf

    def __repr__(self):
        return self.go_id

    def __str__(self):
        return self.go_id

    def serialize(self):
        self.parents = list(self.parents)
        self.children = list(self.children)
        # self.all_children = list(self.children)
        self.gene_set = list(self.gene_set)

    def unserialize(self):
        self.parents = set(self.parents)
        self.children = set(self.children)
        # self.all_children = set(self.children)
        self.gene_set = set(self.gene_set)

    def toJson(self, o):
        if isinstance(o, GeneNode):
            return o.__dict__
        else:
            raise TypeError


def get_ancestor(go_term):
    last = set()
    to_check = {go_term}
    checked = set()
    while to_check:
        term = to_check.pop()
        if term not in checked:
            for parent in term.parents:
                if parent.id == "GO:0008150":
                    last.add(term)
                else:
                    to_check.add(parent)
        checked.add(term)
    return last


def get_go(download_anyway=False):
    go_obo_url = 'http://purl.obolibrary.org/obo/go/go-basic.obo'
    data_folder = os.getcwd() + '/all_data'
    # Check if we have the ./all_data path already
    if not os.path.isfile(data_folder):
        # Emulate mkdir -p (no error if folder exists)
        try:
            os.mkdir(data_folder)
        except OSError as e:
            if e.errno != 17:
                raise e
    else:
        raise Exception('Data path (' + data_folder + ') exists as a file. Please rename, remove or change the desired '
                                                      'location of the all_data path.')
    # Check if the file exists already
    if not os.path.isfile(data_folder + '/go-basic.obo') or download_anyway:
        go_obo = wget.download(go_obo_url, data_folder + '/go-basic.obo')
    else:
        go_obo = data_folder + '/go-basic.obo'
    return go_obo


def build_genomic_tree(biological_processes: Any, go: Dict) -> Tuple[GeneNode, int]:
    """
    BFS traverse of the DAG
    """
    visited: Set[str] = set()
    root = GeneNode(go_id=biological_processes.id, level=biological_processes.level,
                    name=biological_processes.name, go_obj=biological_processes, parents=None)
    to_visit = deque([root])
    id_to_node: Dict[str, GeneNode] = {biological_processes.id: root}
    nodes = 0

    while to_visit:
        current = to_visit.popleft()
        if current.go_id not in visited:
            visited.add(current.go_id)
            nodes += 1

            if current.go_id in go:
                for child in go[current.go_id].children:
                    if child.id not in id_to_node:
                        temp_node = GeneNode(go_id=child.id, level=child.level, name=child.name, go_obj=child)
                        id_to_node[child.id] = temp_node
                        to_visit.append(temp_node)
                    else:
                        temp_node = id_to_node[child.id]
                    current.children += (temp_node,)
                    temp_node.parents.add(current)

        if nodes % 5000 == 0:
            print(f"Processed {nodes} nodes")

    print(f"Total {nodes} nodes for {biological_processes.id}")
    bio_terms = [term for term in go if go[term].namespace == 'biological_process']
    missing_terms = set(bio_terms) - visited
    print(f"missing {len(missing_terms)}, Examples: {list(missing_terms)[:5]}")

    return root, nodes


def get_random_corr(size, df, plot=False, times=10_000):
    # Create an array to store the random samples
    sample_genes = np.array([np.random.choice(df.index, size=size, replace=False) for _ in range(times)])

    # Compute the standard deviation of the selected rows, take the mean across columns, and store the result
    random_corr = np.array([average_pairwise_spearman(df.loc[sample_genes[i]]) for i in range(times)])

    ecdf_data = save_ecdf_efficient(random_corr, tail_threshold=0.05, mid_step=0.05)

    if plot:
        # log_dist = np.log(random_dist[random_dist > 0])
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot histogram
        ax.hist(random_corr, bins=50, density=True, alpha=0.7, color='skyblue')

        # Plot kernel density estimation
        # sns.kdeplot(log_dist, ax=ax, color='navy')
        mean = np.mean(random_corr)
        std = np.std(random_corr)
        x = np.linspace(-1, 6, 100)
        plt.plot(x, (1 / (np.sqrt(2 * np.pi * std ** 2))) * np.exp(-np.power(x - mean, 2) / (2 * (std ** 2))))
        # Add labels and title
        ax.set_xlabel('Mean Standard Deviation')
        ax.set_ylabel('Density')
        ax.set_title('Distribution of Random Correlations')

        # Add text with mean and standard deviation
        mean = np.mean(random_corr)
        std = np.std(random_corr)
        ax.text(0.95, 0.95, f'Mean: {mean:.4f}\nStd: {std:.4f}',
                transform=ax.transAxes, verticalalignment='top',
                horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        plt.title(f"size {len(genes)}")

        # Show the plot
        plt.show()

    # Return the mean and standard deviation of the computed distribution
    return random_corr.mean(), random_corr.std(), ecdf_data


def mean_mwu(anti, genes_data, expression, meta, treatment, condition):
    mw = np.zeros(len(genes_data))
    for i, gene in enumerate(genes_data):
        # get Mann–Whitney score for the gene
        abx, pbs = get_abx_pbs(anti, expression, gene, meta, treatment, condition)

        # maximal = len(abx) * len(pbs)
        if len(pbs) == 0 or len(abx) == 0 or (np.sum(abx) == 0 and np.sum(pbs) == 0):
            mw[i] = np.nan
            continue
        if np.array_equal(abx.values, pbs.values):
            print(f"abx and pbs are the same for {anti}_{treatment}_{gene}")
            mw[i] = np.nan
            continue
        # Perform Mann-Whitney U test once
        _, mw[i] = mannwhitneyu(abx, pbs, alternative='two-sided')
    return gmean(mw)


def mean_fold(anti, genes_data, expression, meta, treatment, condition):
    fold_change = np.zeros(len(genes_data))
    for i, gene in enumerate(genes_data):
        abx, pbs = get_abx_pbs(anti, expression, gene, meta, treatment, condition)
        if len(pbs) == 0 or len(abx) == 0 or (np.sum(abx) == 0 and np.sum(pbs) == 0) or np.array_equal(abx.values,
                                                                                                       pbs.values):
            fold_change[i] = np.nan
            continue
        if pbs.median():
            fold_change[i] = abx.median() - pbs.median()
        else:
            print(f"pbs median is 0 for {anti}_{treatment}_{gene}")
            fold_change[i] = np.nan
    return np.nanmean(fold_change)


def median_t_test(anti, genes_data, current, meta, treatment, condition):
    """
    get the average treat-test over all genes
    """
    # get treat-test score for the median
    abx_samples = meta[(meta['Drug'] == anti) & (meta[condition] == treatment)]
    pbs_samples = meta[(meta['Drug'] == 'PBS') & (meta[condition] == treatment)]
    gene = current.loc[genes_data].median()
    abx = gene[abx_samples['ID']]  # .dropna()
    pbs = gene[pbs_samples['ID']]  # .dropna()
    t_pbs, t_p_pbs = ttest_ind(pbs, abx)
    return t_p_pbs


def median_fold_change(anti, genes_data, current, meta, treatment, condition):
    """
    Calculate the difference in median z-scores between antibiotic and PBS groups.

    Steps:
    1. Calculate median z-score per sample for the selected genes
    2. Calculate mean of median z-scores per group (antibiotic and PBS)
    3. Calculate the difference between these means
    4. Return the z-score difference
    """
    # get FC for the median
    abx_samples = meta[(meta['Drug'] == anti) & (meta[condition] == treatment)]
    pbs_samples = meta[(meta['Drug'] == 'PBS') & (meta[condition] == treatment)]
    # Get gene expression data for the selected genes
    gene_data = current.loc[genes_data]
    # Step 1: Calculate median per sample
    abx_medians = gene_data[abx_samples['ID']].median()
    pbs_medians = gene_data[pbs_samples['ID']].median()
    # Step 2: Calculate mean of medians per group
    abx_mean = np.mean(abx_medians)
    pbs_mean = np.mean(pbs_medians)
    # Step 3: Calculate fold change
    z_score_difference = abx_mean - pbs_mean
    return z_score_difference


def genes_data_split(anti, genes_data, current, meta, treatment, condition, threshold=0.05):
    enhanced, suppressed = set(), set()
    significant_genes = {}
    for gene in genes_data:
        # get treat-test score for the gene
        abx, pbs = get_abx_pbs(anti, current, gene, meta, treatment, condition)
        t_abx, t_p_abx = ttest_ind(abx, pbs)
        if t_abx > 0:  # meaning the abx is enhanced
            enhanced.add(gene)
        else:
            suppressed.add(gene)
        if t_p_abx < threshold:
            significant_genes[gene] = t_p_abx
    return enhanced, suppressed, significant_genes
    # return list(enhanced), list(suppressed)


def get_abx_pbs(anti, current, gene, meta, treatment, condition):
    abx_data = meta[(meta['Drug'] == anti) & (meta[condition] == treatment)]
    pbs_data = meta[(meta['Drug'] == 'PBS') & (meta[condition] == treatment)]
    # if len(current.loc[gene]) > 1, take the row with the fewer nan values
    row = current.loc[gene]
    if len(row.shape) > 1 and len(row) > 1:
        row = row.dropna()
        print(f"for some reason gene {gene} appears twice")
    abx = (row[abx_data['ID']]).dropna()
    pbs = (row[pbs_data['ID']]).dropna()
    return abx, pbs


def plot_curve(random_cutoff, random_std, path):
    # Extract keys, values, and standard deviations
    keys = list(random_cutoff.keys())
    values = [random_cutoff[key] for key in keys]
    std_devs = [random_std[key] for key in keys]

    # Plotting
    plt.errorbar(keys, values, yerr=std_devs, fmt='o', capsize=5, capthick=1, ecolor='red')
    plt.xlabel('Genes number (group size)')
    plt.ylabel('Average mean-std of the group')
    plt.title('Values with Error Bars')
    plt.savefig(path + ".png")
    # plt.show()
    plt.close()


def save_ecdf_efficient(bootstrap_results, tail_threshold=0.05, mid_step=0.05):
    sorted_data = np.sort(bootstrap_results)
    n = len(sorted_data)

    # Calculate full ECDF
    ecdf_values = np.arange(1, n + 1) / n

    # Initialize lists to store selected points
    selected_data = []
    selected_ecdf = []

    # Save tail data (both lower and upper)
    lower_idx = np.searchsorted(ecdf_values, tail_threshold)
    upper_idx = np.searchsorted(ecdf_values, 1 - tail_threshold)

    selected_data.extend(sorted_data[:lower_idx])
    selected_ecdf.extend(ecdf_values[:lower_idx])

    # Save middle data with larger steps
    current_value = tail_threshold
    while current_value < (1 - tail_threshold):
        idx = np.searchsorted(ecdf_values, current_value)
        selected_data.append(sorted_data[idx])
        selected_ecdf.append(ecdf_values[idx])
        current_value += mid_step

    selected_data.extend(sorted_data[upper_idx:])
    selected_ecdf.extend(ecdf_values[upper_idx:])

    return {'data': np.array(selected_data), 'ecdf': np.array(selected_ecdf)}


def calculate_pvalue_ecdf_efficient_lower_tail(observed_value, ecdf_data):
    """
        Calculate the one-tailed p-value for a given observed value using pre-saved ECDF data.

        Parameters:
            observed_value (float): The observed mean pairwise correlation.
            ecdf_data (dict): A dictionary with keys 'data' and 'ecdf' from `save_ecdf_efficient`.
                - 'data': The sorted null distribution values.
                - 'ecdf': The corresponding ECDF values.

        Returns:
            p_value (float): The one-tailed p-value.
        """
    null_data = ecdf_data['data']
    ecdf_values = ecdf_data['ecdf']

    # Find the position where the observed value would be inserted into the null data
    idx = np.searchsorted(null_data, observed_value, side='right')
    correction = 1 / (2 * len(null_data))
    if idx == 0:
        return 1.0 - correction  # Avoid p-value of 1
    # If the observed value is greater than all null values, p-value is close to 0
    if idx == len(null_data):
        return correction  # Avoid p-value of 0

    # Get the ECDF value corresponding to the observed value
    # Interpolate between the two nearest points
    x0, x1 = null_data[idx - 1], null_data[idx]
    y0, y1 = ecdf_values[idx - 1], ecdf_values[idx]
    interpolated_ecdf = y0 + (observed_value - x0) * (y1 - y0) / (x1 - x0)

    # Calculate p-value as 1 - ECDF
    p_value = 1.0 - interpolated_ecdf
    return max(min(p_value, 1.0 - correction), correction)  # Apply continuity correction


def average_pairwise_spearman(gene_data):
    """
    Calculate the average pairwise Spearman correlation for all pairs of genes.

    Parameters:
    gene_data (pd.DataFrame): A DataFrame where rows are genes and columns are samples.

    Returns:
    float: The average pairwise Spearman correlation.
    """
    corr_matrix = gene_data.T.corr(method='spearman')
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    corr_values = upper_tri.values[np.triu_indices(corr_matrix.shape[0], k=1)]
    # corr_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).dropna().values
    return np.nanmean(corr_values)


def calculate_hypergeometric_pvalue(N, K, n, k):
    """
    Calculate the hypergeometric p-value.

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
        The p-value
    """
    # Calculate the probability of getting k or more successes
    pvalue = 1 - stats.hypergeom.cdf(k - 1, N, K, n)

    return pvalue


def calculate_correlation(root, expression, meta, size, antis, treats, gene_to_check, exp_type, condition,
                          remove=('N18'), significance_threshold=0.05):
    go_to_ensmbl_dict = get_go_to_ensmusg()
    go = obo_parser.GODag(get_go())

    folder_dir = f"./data/"
    df = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")
    id_to_name = df.set_index('gene_id')['gene_name'].to_dict()
    dist = {}
    top = pd.DataFrame()
    for anti in antis:
        dist[anti] = {}
        for treatment in treats:
            print(f"starting {anti} {treatment}")
            temp = pd.DataFrame()
            dist[anti][treatment] = np.zeros(size)
            # drug = anti.lower()
            samples = meta[((meta['Drug'] == anti) | (meta['Drug'] == 'PBS')) & (meta[condition] == treatment)]
            # remove missing samples
            for sample in remove:
                if sample in samples['ID'].values:
                    samples = samples.drop(samples[samples['ID'] == sample].index)
                    print(f"{sample} removed")
            iter_samples = samples['ID'].to_list()
            for sample in iter_samples:
                if sample not in expression.columns:
                    print(f"{sample} not in expression")
                    samples = samples[samples['ID'] != sample]
                    meta = meta[meta['ID'] != sample]
            current = expression[samples['ID']]
            abx_samples = samples[samples['Drug'] == anti]
            pbs_samples = samples[samples['Drug'] == 'PBS']
            current_abx = expression[abx_samples['ID']]
            current_pbs = expression[pbs_samples['ID']]
            # Calculate trend for all genes for this condition
            genes_enhanced_all, genes_suppressed_all, significant_genes = genes_data_split(anti, current.index, current,
                                                                                           meta, treatment, condition,
                                                                                           threshold=significance_threshold)

            counter = 0
            random_cutoff_enh = {}
            random_std_enh = {}
            ecdf_storage_enh = {}
            random_cutoff_supp = {}
            random_std_supp = {}
            ecdf_storage_supp = {}
            # mwu_cutoff = {}
            for i, node in enumerate(go_to_ensmbl_dict):
                # for i, node in enumerate(PreOrderIter(root)):
                if len(set(go_to_ensmbl_dict[node])) == 0:
                    continue
                if node not in go:
                    continue
                genes_not_in_data = set(set(go_to_ensmbl_dict[node]) - set(current.index))
                no_genes = len(genes_not_in_data)

                go_to_ensmbl_dict[node] = [gene for gene in go_to_ensmbl_dict[node] if gene not in genes_not_in_data]
                if no_genes:
                    print(
                        f"{no_genes} genes were not in all_data ({anti} {treatment}) for {go[node].name if node in go else 'NO_NAME'}")

                genes_enhanced = [gene for gene in go_to_ensmbl_dict[node] if
                                  ((gene in genes_enhanced_all) and (gene in significant_genes))]
                genes_suppressed = [gene for gene in go_to_ensmbl_dict[node] if
                                    ((gene in genes_suppressed_all) and (gene in significant_genes))]

                GO_significance = calculate_hypergeometric_pvalue(len(current.index), len(significant_genes),
                                                                  len(go_to_ensmbl_dict[node]),
                                                                  len([gene for gene in go_to_ensmbl_dict[node] if
                                                                       gene in significant_genes]))
                enhanced = True
                for genes_data in [genes_enhanced, genes_suppressed]:
                    # if dist is not np.nan and distance > 0 and len(genes_data) > 0:
                    # if distance > 0 and len(genes_data) > 0:
                    if len(genes_data) > 1:  # 0?
                        category_size = round(len(genes_data) / 10) * 10 if len(genes_data) > 50 else len(genes_data)
                        if category_size == 0:
                            continue
                        if enhanced:
                            if category_size not in random_cutoff_enh:
                                random_cutoff_enh[category_size], random_std_enh[category_size], ecdf_storage_enh[
                                    category_size] = get_random_corr(category_size,
                                                                     current.loc[list(genes_enhanced_all)])

                        else:

                            if category_size not in random_cutoff_supp:
                                # calculate random groups pairwise correlation
                                random_cutoff_supp[category_size], random_std_supp[category_size], ecdf_storage_supp[
                                    category_size] = get_random_corr(category_size,
                                                                     current.loc[list(genes_suppressed_all)])

                        distance = np.nanmean(np.nanstd(current.loc[genes_data], axis=0))
                        distance_abx = np.nanmean(np.nanstd(current_abx.loc[genes_data], axis=0))
                        distance_pbs = np.nanmean(np.nanstd(current_pbs.loc[genes_data], axis=0))

                        correlation = average_pairwise_spearman(current.loc[genes_data])
                        correlation_abx = average_pairwise_spearman(current_abx.loc[genes_data])
                        correlation_pbs = average_pairwise_spearman(current_pbs.loc[genes_data])

                        variance = np.nanmean(np.nanvar(current.loc[genes_data], axis=1))
                        variance_abx = np.nanmean(np.nanvar(current_abx.loc[genes_data], axis=1))
                        variance_pbs = np.nanmean(np.nanvar(current_pbs.loc[genes_data], axis=1))

                        dist[anti][treatment][counter] = correlation
                        counter += 1
                        # best = mean_mwu(anti, genes_data, current, meta_data, treatment)
                        # best = geomean_t_test(anti, genes_data, current, meta, treatment, condition)
                        median_ttest = median_t_test(anti, genes_data, current, meta, treatment, condition)
                        mwu = mean_mwu(anti, genes_data, current, meta, treatment, condition)
                        fold_change = mean_fold(anti, genes_data, current, meta, treatment, condition)
                        median_zscore_diff = median_fold_change(anti, genes_data, current, meta, treatment, condition)

                        genes_id_to_write = [id_to_name[gene] for gene in genes_data if
                                             ((gene in id_to_name) and (len(genes_data) < 3000))]
                        genes_to_write = genes_data if len(genes_data) < 300 else f"size = {len(genes_data)}"

                        suf = "_enh" if enhanced else "_supp"
                        all_ancestors = list(get_ancestor(go[node])) if node in go else None
                        if not all_ancestors:
                            category_name = "NOT_BP"
                        else:
                            category_name = all_ancestors[0].name if len(all_ancestors) == 1 else [ancestor.name for
                                                                                                   ancestor in
                                                                                                   all_ancestors]
                        curr_storage = ecdf_storage_enh[category_size] if enhanced else ecdf_storage_supp[category_size]
                        line = {'Antibiotics': anti, 'Condition': treatment, 'GO term': node + suf,
                                'name': f"{category_name}:{go[node].name if node in go else 'NO_NAME'}",
                                'genes': genes_to_write, 'gene names': genes_id_to_write,
                                'GO significance': GO_significance, 'correlation': correlation,
                                'correlation_pbs': correlation_pbs, 'corrlation_abx': correlation_abx,
                                'distance': distance_abx, '\"log(distance)\"': np.log2(distance),
                                'mean variance between samples': variance_abx, 'distance_all': distance,
                                'distance_pbs': distance_pbs, 'mean variance between samples all mice': variance,
                                'mean variance between samples pbs': variance_pbs, "size": len(genes_data),

                                f'p-value correlation': calculate_pvalue_ecdf_efficient_lower_tail(correlation,
                                                                                                   curr_storage),
                                'random correlation': random_cutoff_enh[category_size] if enhanced else
                                random_cutoff_supp[category_size],
                                'std correlation': random_std_enh[category_size] if enhanced else random_std_supp[
                                    category_size],
                                'median t-test p-value': median_ttest, 't-test less than 5%': median_ttest < 0.05,
                                'normalized fold change': fold_change,  # 'log2 fold change': np.log2(fold_change),
                                'median zscore diff': median_zscore_diff,
                                "enhanced?": enhanced, "relative size": len(genes_data) / len(go_to_ensmbl_dict[node]),
                                'MWU': mwu, 'MWU less than 5%': mwu < 0.05, }

                        line_df = pd.DataFrame([line])  # Convert the dictionary to a DataFrame

                        temp = pd.concat([temp, line_df], ignore_index=True)
                    enhanced = False

            print(f"{no_genes} were not in all_data ({anti} {treatment})")
            # print(temp.head())
            temp["fdr GO significance"] = fdrcorrection(temp["GO significance"])[1]

            temp["fdr correlation"] = np.nan
            filtered_p_values = \
                temp[(temp["fdr GO significance"] < 0.05) & temp["p-value correlation"].notna()][
                    "p-value correlation"]
            # Apply FDR correction to the filtered p-values
            fdr_corrected = fdrcorrection(filtered_p_values.to_list())[1]
            # temp["fdr correlation"] = fdrcorrection(temp["p-value correlation"])[1]
            temp.loc[
                (temp["fdr GO significance"] < 0.05) & temp[
                    "p-value correlation"].notna(), "fdr correlation"] = fdr_corrected
            # temp["fdr t-test"] = fdrcorrection(temp["treat-test p-value"])[1]
            # Filter the rows where p-value correlation is less than 0.05
            filtered_p_values = temp[(temp["fdr correlation"] < 0.05) & temp["median t-test p-value"].notna()][
                "median t-test p-value"]
            # Apply FDR correction to the filtered p-values
            fdr_corrected = fdrcorrection(filtered_p_values)[1]
            # Create a new column with NaN values
            temp["fdr median t-test"] = np.nan
            # Assign the FDR corrected values back to the DataFrame
            # temp.loc[temp["fdr correlation"] < 0.05, "fdr median t-test"] = fdr_corrected
            temp.loc[(temp["fdr correlation"] < 0.05) & temp[
                "median t-test p-value"].notna(), "fdr median t-test"] = fdr_corrected

            temp.to_csv(f'./Private/clusters_properties/{exp_type}/top_correlated_GO_terms_{anti}_{treatment}.tsv',
                        sep='\t', index=False)
            top = pd.concat([top, temp], ignore_index=True)
            plot_curve(random_cutoff_enh, random_std_enh,
                       f'./Private/random_tightness/{exp_type}_{anti}_{treatment}_corr-vs-size_enh')
            plot_curve(random_cutoff_supp, random_std_supp,
                       f'./Private/random_tightness/{exp_type}_{anti}_{treatment}_corr-vs-size_supp')
    top.to_csv(f'./Private/clusters_properties/{exp_type}/top_correlated_GO_terms.tsv', sep='\t', index=False)
    return dist


def impute_zeros(to_impute, meta_data, condition, run_type='', skip_if_exist=False, mean=False):
    """
    replaces all zeros by mean of other gene expression of same treatment and same antibiotic
    """
    # if the file f'./Private/imputed_all_log_zeros_removed{run_type}.csv' exists, return it
    if skip_if_exist and os.path.exists(f'./Private/imputed_all_zeros_removed{run_type}.csv'):
        return pd.read_csv(f'./Private/imputed_all_zeros_removed{run_type}.csv', index_col=0)
    if mean:
        means, stds = get_mean_all(to_impute, meta_data, condition, skip_if_exist, run_type)
    # add column of nan counts of the row
    to_impute['nans'] = to_impute.isnull().sum(axis=1)
    to_impute = to_impute.drop(to_impute[to_impute['nans'] >= 0.2 * to_impute.shape[1]].index).drop('nans', axis=1)
    row, col = np.where(to_impute.isnull())
    total = len(row)
    counter = 1
    all_other_are_zeros = 0
    all_other_are_zeros_conditions = 0
    too_big = 0
    zeros = set()
    for i, j in zip(row, col):
        # assert it is nan
        assert np.isnan(to_impute.iloc[i, j])
        name = to_impute.columns[j]
        antibiotic = meta_data[meta_data['ID'] == name]['Drug'].values[0]
        treatment = meta_data[meta_data['ID'] == name][condition].values[0]
        # print(name, antibiotic, treatment)
        if counter % 5000 == 0:
            print(f"{counter}/{total} zeros imputed")
        counter += 1
        mice = meta_data[(meta_data['Drug'] == antibiotic) & (meta_data[condition] == treatment) &
                         (meta_data['ID'] != name)]['ID']
        if mean:
            # replace the zero with the geometric mean of the other mice
            mean = np.nanmean(to_impute.iloc[i][mice])
            if np.isnan(mean) or (antibiotic, treatment, to_impute.index[i]) in zeros:
                zeros.add((antibiotic, treatment, to_impute.index[i]))
                all_other_are_zeros += 1
                all_other_are_zeros_conditions += 1 / (len(mice) + 1)
                # set mean to be the min non-zero value of this sample (column)
                mean = np.min(to_impute[name][~to_impute[name].isnull()])
            if abs(mean - means[antibiotic][treatment]) > stds[antibiotic][treatment]:
                too_big += 1
                continue
            to_impute.iloc[i, j] = mean
        else:
            to_impute.iloc[i, j] = np.nanmin(to_impute.iloc[i][mice])

    row, col = np.where(to_impute.isnull())
    print(
        f"Now left with {len(row)} zeros, {set([to_impute.columns[c] for c in col])}")  # , but {all_other_are_zeros} are zeros in all other mice")
    # replace na with 1
    to_impute = to_impute.fillna(1)
    to_impute.to_csv(f'./Private/imputed_all_zeros_removed{run_type}.csv')
    return to_impute


def get_mean_all(df, meta, condition, skip, run_type):
    # if the file f'./Private/means{run_type}.json' exists, return it and the stds
    if skip and os.path.exists(f'./Private/means{run_type}.json'):
        with open(f"./Private/means{run_type}.json", "r") as f:
            all_means = json.load(f)
        with open(f"./Private/stds{run_type}.json", "r") as f:
            all_stds = json.load(f)
        return all_means, all_stds
    all_means = {}
    seen = {}
    for i in range(df.shape[0]):
        if i % 100 == 0:
            print(f"calculating mean for {i}/{df.shape[0]}")
        for j in range(df.shape[1]):
            name = df.columns[j]
            antibiotic = meta[meta['ID'] == name]['Drug'].values[0]
            treatment = meta[meta['ID'] == name][condition].values[0]
            if antibiotic not in all_means:
                all_means[antibiotic] = {}
                seen[antibiotic] = {}
            if treatment not in all_means[antibiotic]:
                all_means[antibiotic][treatment] = []
                seen[antibiotic][treatment] = set()
            if df.index[i] not in seen[antibiotic][treatment]:
                seen[antibiotic][treatment].add(df.index[i])
                mice = meta[(meta['Drug'] == antibiotic) & (meta[condition] == treatment)]['ID']
                all_means[antibiotic][treatment].append(gmean(df.iloc[i][mice]))
    # keep mean and std of all lists for each antibiotic and treatment
    all_stds = {}
    for antibiotic in all_means:
        all_stds[antibiotic] = {}
        for treatment in all_means[antibiotic]:
            all_stds[antibiotic][treatment] = np.std(all_means[antibiotic][treatment])
            all_means[antibiotic][treatment] = np.mean(all_means[antibiotic][treatment])
    try:
        # save the means and stds
        with open(f"./Private/means{run_type}.json", "w") as f:
            json.dump(all_means, f)
        with open(f"./Private/stds{run_type}.json", "w") as f:
            json.dump(all_stds, f)
    except:
        print("couldn't save means and stds")
    return all_means, all_stds


def get_go_to_ensmusg(bio_path="http://www.ensembl.org/biomart", cache_file="go_to_ensmusg.pkl"):
    # Check if the dictionary already exists
    if os.path.exists(cache_file):
        print(f"Loading dictionary from {cache_file}...")
        with open(cache_file, "rb") as f:
            go_to_ensmusg = pickle.load(f)
        return go_to_ensmusg

    from biomart import BiomartServer
    # try:
    # Connect to the BioMart server
    server = BiomartServer(bio_path)

    # Choose the Ensembl database
    mart = server.datasets['mmusculus_gene_ensembl']

    # Define the attributes you want to retrieve
    attributes = [
        'ensembl_gene_id',
        'go_id'
    ]
    filters = {
        'go_parent_term': 'GO:0008150'  # This is the root term for Biological Process
    }

    # Query BioMart
    response = mart.search({
        'filters': filters,
        'attributes': attributes
    })

    # Parse the response
    go_to_ensmusg = defaultdict(set)
    for line in response.iter_lines():
        decoded_line = line.decode('utf-8')
        ensembl_gene_id, go_id = decoded_line.split("\t")
        if go_id:
            go_to_ensmusg[go_id].add(ensembl_gene_id)
    # Save the dictionary for future use
    print(f"Saving dictionary to {cache_file}...")
    with open(cache_file, "wb") as f:
        pickle.dump(go_to_ensmusg, f)
    return go_to_ensmusg


def add_genes_ids(root: Any, go_to_ensmbl_dict: Dict[str, Set[str]],
                  progress_interval: int = 1000,
                  max_examples: int = 5,
                  gene_name_file: str = "./data/transcriptome_2023-09-17-genes_norm_named.tsv") -> Any:
    empty_nodes_counter = 0
    added: Set[str] = set()

    for i, node in enumerate(PostOrderIter(root)):
        node_genes = go_to_ensmbl_dict.get(node.go_id, set())
        if node_genes:
            node.gene_set = node.gene_set.union(node_genes)
            added.add(node.go_id)
        else:
            empty_nodes_counter += 1

        if i % progress_interval == 0:
            print(f"### {i} nodes were updated ###")

    print(f"{empty_nodes_counter} empty nodes")
    print(f"Out of {len(go_to_ensmbl_dict)} mmusculus_gene_ensembl GOs, {len(added)} were added")

    missing = set(go_to_ensmbl_dict.keys()) - added
    print("Examples:")

    try:
        df = pd.read_csv(gene_name_file, sep="\t")
        id_to_name = df.set_index('gene_id')['gene_name'].to_dict()

        for i, go in enumerate(missing):
            if i >= max_examples:
                break
            gene_names = [id_to_name.get(name, name) for name in go_to_ensmbl_dict[go]]
            print(go, gene_names)
    except FileNotFoundError:
        print(f"Warning: Gene name file not found at {gene_name_file}")
    except KeyError as e:
        print(f"Warning: Expected column not found in gene name file: {e}")

    return root


def build_tree(download=False):
    go = obo_parser.GODag(get_go(download_anyway=download))
    # cell_cycle_genes()
    filename = "genomic_tree.json"
    # check if file "genomic_tree.jason" is in current path
    if os.path.exists(f"./Private/{filename}"):
        tree = JsonImporter().read(open(f"./Private/{filename}"))
        for node in PostOrderIter(tree):
            node.unserialize()
    else:
        tree, tree_size = build_genomic_tree(go['GO:0008150'], go)
        bio_terms = [term for term in go if go[term].namespace == 'biological_process']
        print(f"{tree_size} nodes were built out of {len(bio_terms)} biological process GO terms")

        go_to_ensmbl_dict = get_go_to_ensmusg()
        print(f"GO genes number {len(go_to_ensmbl_dict)}")
        print(f"terms not in dictionary file: {len([term for term in go.values() if term not in go_to_ensmbl_dict])}")
        add_genes_ids(tree, go_to_ensmbl_dict)

    return tree, tree_size


def read_process_files(new=False, filter_value=0.55, merge_big_abx=True, remove_mitochondrial=True, gene_name=False):
    partek_df = pd.read_csv(
        "./data/New Partek_bell_all_Normalization_Normalized_counts1.csv")
    partek_df = partek_df.set_index("Gene Symbol")
    folder_dir = f"./data/"
    genome_df = pd.read_csv(folder_dir + "rpkm_named_genome-2023-09-26.tsv", sep="\t")
    transcriptome_df = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")

    if gene_name:
        # replace all empty cells in gene_name with the value in gene_id
        genome_df["gene_name"] = genome_df.apply(
            lambda row: row["gene_id"] if pd.isna(row["gene_name"]) else row["gene_name"], axis=1)
        transcriptome_df["gene_name"] = transcriptome_df.apply(
            lambda row: row["gene_id"] if pd.isna(row["gene_name"]) else row["gene_name"], axis=1)
        genome_df = genome_df.set_index("gene_name")
        transcriptome_df = transcriptome_df.set_index("gene_name")
        genome_df = genome_df.drop("gene_id", axis=1)
        transcriptome_df = transcriptome_df.drop("gene_id", axis=1)
    else:
        genome_df = genome_df.drop("gene_name", axis=1)
        transcriptome_df = transcriptome_df.drop("gene_name", axis=1)
        genome_df.rename(columns={'gene_id': 'gene_name'}, inplace=True)
        transcriptome_df.rename(columns={'gene_id': 'gene_name'}, inplace=True)
        genome_df = genome_df.set_index("gene_name")
        transcriptome_df = transcriptome_df.set_index("gene_name")

    # replace partek nans with 0
    partek_df = partek_df.fillna(0)

    metadata = get_metadata(data_folder, type="", only_old=not new, filter=filter_value)

    # change genome and transcriptome column names using metadata: replace the name which is 'Sample' to the
    # equivalent 'ID'
    genome_df = genome_df.rename(columns=metadata.set_index('Sample')['ID'].to_dict())
    transcriptome_df = transcriptome_df.rename(columns=metadata.set_index('Sample')['ID'].to_dict())

    # keep in all 3 DFs only columns that are in metadata["ID"].values
    genome_df = genome_df[[col for col in genome_df.columns if col in metadata["ID"].values]]
    transcriptome_df = transcriptome_df[[col for col in transcriptome_df.columns if col in metadata["ID"].values]]
    partek_df = partek_df[[col for col in partek_df.columns if col in metadata["ID"].values]]

    if merge_big_abx:
        new_path = r"./data/"
        new_data = pd.read_csv(new_path + "mRNA_NEBNext_20200908_genes_norm_named.tsv", sep="\t")
        # sum rows with the same gene_name and drop the gene_id column
        # new_data = new_data.drop("gene_id", axis=1).groupby("gene_name").sum()
        new_stats = pd.read_csv(new_path + r"big_abx_stats.csv")
        # remove all samples with "aligned" < 0.5
        columns_to_keep = new_stats[new_stats["aligned"] > filter_value]["Sample Name"]
        # new_data = new_data[columns_to_keep.append(pd.Series(["gene_name", "gene_id"]))]
        columns_to_keep = columns_to_keep.tolist()  # Convert to list if needed
        columns_to_keep.append("gene_name")  # Append to the list
        columns_to_keep.append("gene_id")
        new_data.columns = [col.split("_")[-1] if "gene" not in col else col for col in new_data.columns]
        # drop columns C1, C2, C3 as they already exist in the other df
        new_data = new_data.drop(["C1", "C2", "C3"], axis=1)

        if gene_name:
            new_data["gene_name"] = new_data.apply(
                lambda row: row.name if pd.isna(row["gene_name"]) else row["gene_name"], axis=1)
            new_data = new_data.set_index("gene_name").drop("gene_id", axis=1)
        else:
            new_data = new_data.drop("gene_name", axis=1)
            new_data.rename(columns={'gene_id': 'gene_name'}, inplace=True)
            new_data = new_data.set_index("gene_name")
        transcriptome_df = pd.merge(transcriptome_df, new_data, left_index=True, right_index=True)
        new_metadata = get_metadata(data_folder, type="", only_old=not new, filter=False)
        new_metadata = new_metadata[new_metadata["ID"].isin(new_data.columns)]
        metadata = pd.concat([metadata, new_metadata])

    # sum rows from transcriptome and genome with the same index TODO
    # print indexes that appear twice in genome and transcriptome
    if len(genome_df.index[genome_df.index.duplicated()]) > 0:
        print("indexes that appear twice in genome:\n", genome_df.index[genome_df.index.duplicated()])
        print("and transcriptome:\n", transcriptome_df.index[transcriptome_df.index.duplicated()])
    genome_df = genome_df.groupby(genome_df.index).sum()
    transcriptome_df = transcriptome_df.groupby(transcriptome_df.index).sum()

    # remove sparse genes (more than 50% zeros in a row):
    # check all sparse genes (more than 50% zeros in a row) in each df, and check if the non-zero samples are the same
    # condition, using the metadata
    partek_zeros = partek_df[partek_df == 0].count(axis=1)
    partek_sparse = partek_zeros[partek_zeros > 0.5 * partek_df.shape[1]]
    genome_zeros = genome_df[genome_df == 0].count(axis=1)
    genome_sparse = genome_zeros[genome_zeros > 0.5 * genome_df.shape[1]]
    transcriptome_zeros = transcriptome_df[transcriptome_df == 0].count(axis=1)
    transcriptome_sparse = transcriptome_zeros[transcriptome_zeros > 0.5 * transcriptome_df.shape[1]]
    partek_df = partek_df.drop(partek_sparse.index)
    genome_df = genome_df.drop(genome_sparse.index)
    transcriptome_df = transcriptome_df.drop(transcriptome_sparse.index)

    if remove_mitochondrial:
        matching_indices = transcriptome_df.index[
            transcriptome_df.index.str.lower().isin(set(mitochondrial_genes))].tolist()

        # remove mitochondrial genes from the dataframes
        genome_df = genome_df.drop(matching_indices, errors='ignore')
        transcriptome_df = transcriptome_df.drop(matching_indices, errors='ignore')
        partek_df = partek_df.drop(matching_indices, errors='ignore')

    partek_df = (partek_df * 1000000).divide(partek_df.sum(axis=0), axis=1)
    genome_df = (genome_df * 1000000).divide(genome_df.sum(axis=0), axis=1)
    transcriptome_df = (transcriptome_df * 1000000).divide(transcriptome_df.sum(axis=0), axis=1)

    # NOTICE! drop C9, C10, C18, M13, V14 from all DFs and metadata
    to_remove = ["C9", "C10", "C18", "M13", "V14", "V11"]
    transcriptome_df = transcriptome_df.drop(to_remove, axis=1)
    metadata = metadata[~metadata["ID"].isin(to_remove)]

    return genome_df, metadata, partek_df, transcriptome_df


def get_metadata(folder, type="", only_old=True, filter=0.55):
    meta = pd.read_excel(os.path.join(folder, "metadata.xlsx"))
    meta['ID'] = meta.apply(lambda row: row['ID'] + 'N' if row['New/Old'] == 'N' else row['ID'], axis=1)
    meta['Drug'] = meta.apply(lambda row: row['Drug'].replace('mix', 'Mix').replace('ampicillin', 'Amp')
                              .replace('Control ', 'PBS').replace('METRO', 'Met').replace('NEO', 'Neo')
                              .replace('VANCO', 'Van'), axis=1)
    if filter:
        file = "RASflow stats 2023_09_26.csv" if type else "RASflow stats 2023_09_17.csv"
        qc = pd.read_csv(os.path.join(folder, file))
        # get Sample Name from qc if aligned > filter
        samples = qc[qc['aligned'] > filter]['Sample Name']
        # print the filtered out samples, sorted lexically
        print(sorted([sample for sample in qc['Sample Name'] if sample not in samples.values]))
        # keep only metadata rows with Sample Name in Sample
        meta = meta[meta['Sample'].isin(samples)]
    # # print samples that are in samples and not in meta.Sample
    # print([sample for sample in samples if sample not in meta['Sample'].values])
    if only_old:
        # remove all samples that end with N from metadata and from data
        meta = meta[~meta['ID'].str.endswith('N')]
    return meta


def zscore_all_by_pbs(data, metadata):
    for treat in treatments:
        pbs = metadata[(metadata['Drug'] == "PBS") & (metadata["Treatment"] == treat)]
        # get the pbs mice data
        pbs_data = data[pbs['ID']]
        # calculate the mean and std of the pbs mice
        pbs_mean = pbs_data.mean(axis=1)
        pbs_std = pbs_data.std(axis=1)
        # replace pbs_std 0 values by np.nanmin(pbs_std)
        pbs_std[pbs_std == 0] = np.nanmin(pbs_std[pbs_std != 0])
        data[pbs['ID']] = data[pbs['ID']].sub(pbs_mean, axis=0)
        data[pbs['ID']] = data[pbs['ID']].div(pbs_std, axis=0)
        for anti in antibiotics:
            abx = metadata[(metadata['Drug'] == anti) & (metadata["Treatment"] == treat)]
            # normalize the data by the mean and std of the pbs mice: subtract pbs_mean from every row and divide by std
            data[abx['ID']] = data[abx['ID']].sub(pbs_mean, axis=0)
            data[abx['ID']] = data[abx['ID']].div(pbs_std, axis=0)
    # return the normalized data
    return data


def zscore_all_by_pbs_gf(data_gf, metadata_gf):
    pbs = metadata_gf[metadata_gf['Drug'] == "PBS"]
    # get the pbs mice data
    pbs_data = data_gf[pbs['ID']]
    # calculate the mean and std of the pbs mice
    pbs_mean = pbs_data.mean(axis=1)
    pbs_std = pbs_data.std(axis=1)
    # replace pbs_std 0 values by np.nanmin(pbs_std)
    pbs_std[pbs_std == 0] = np.nanmin(pbs_std[pbs_std != 0])
    data_gf[pbs['ID']] = data_gf[pbs['ID']].sub(pbs_mean, axis=0)
    data_gf[pbs['ID']] = data_gf[pbs['ID']].div(pbs_std, axis=0)
    abx = metadata_gf[metadata_gf['Drug'] == "Van"]
    # normalize the data by the mean and std of the pbs mice: subtract pbs_mean from every row and divide by std
    data_gf[abx['ID']] = data_gf[abx['ID']].sub(pbs_mean, axis=0)
    data_gf[abx['ID']] = data_gf[abx['ID']].div(pbs_std, axis=0)
    # return the normalized data
    return data_gf


def transform_data(data, metadata, run_type, skip=False, save=False, gf=False):
    # replace all zeros with nan
    data = data.replace(0, np.nan)
    if save:
        folder_dir = f"./data/"
        df = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")
        id_to_name = df.set_index('gene_id')['gene_name'].to_dict()
        data['gene_original_name'] = data.index.map(id_to_name)

    data = impute_zeros(data, metadata, 'Treatment', run_type, skip_if_exist=skip)
    if save:
        data['gene_original_name'] = data.index.map(id_to_name)
        data.to_csv("./Private/data process/imputed.csv")
        # data.to_csv("./Private/data process/no_v11_imputed.csv")
        data = data.drop('gene_original_name', axis=1)
    data = np.log2(data)
    if save:
        data['gene_original_name'] = data.index.map(id_to_name)
        data.to_csv("./Private/data process/imputed_log.csv")
        # data.to_csv("./Private/data process/no_v11_imputed_log.csv")
        data = data.drop('gene_original_name', axis=1)
    # z-score by PBS
    data = zscore_all_by_pbs(data, metadata) if not gf else zscore_all_by_pbs_gf(data, metadata)
    if save:
        data['gene_original_name'] = data.index.map(id_to_name)
        data.to_csv("./Private/data process/imputed_log_zscore.csv")
        # data.to_csv("./Private/data process/no_v11_imputed_log_zscore.csv")
        data = data.drop('gene_original_name', axis=1)
    return data, metadata


def get_ensmus_dict():
    folder_dir = f"./data/"
    df = pd.read_csv(folder_dir + "transcriptome_2023-09-17-genes_norm_named.tsv", sep="\t")
    return df.set_index('gene_id')['gene_name'].to_dict()


def set_plot_defaults():
    plt.rcParams.update({
        'font.family': 'Helvetica',
        'font.size': 10,
        'axes.labelsize': 10,
        'axes.titlesize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })
    sns.set_theme(rc=plt.rcParams)


if __name__ == "__main__":
    run_type = sys.argv[1]

    genome, metadata, partek, transcriptome = read_process_files(new=False)

    genes_dict = get_ensmus_dict()
    genes = [genes_dict[gene] for gene in transcriptome.index]
    to_save = transcriptome.copy()
    to_save = to_save.reset_index()
    to_save.index = genes
    to_save.to_csv(f"./Private/data{run_type}.csv")
    data = transcriptome
    data, metadata = transform_data(data, metadata, run_type, skip=True)
    genes = [genes_dict[gene] for gene in data.index]
    to_save = data.copy()
    to_save.reset_index()
    to_save.index = genes
    # save the data
    to_save.to_csv(f"./Private/transformed_data{run_type}.csv")
    metadata.to_csv(f"./Private/transformed_metadata{run_type}.csv")
    tree, tree_size = build_tree(True)
    corr = calculate_correlation(tree, data, metadata, tree_size, antibiotics, treatments, "H2-Ab1",
                                 f"diff_abx{run_type}", 'Treatment')
