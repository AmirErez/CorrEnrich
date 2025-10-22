"""
Utility functions, including plotting.
"""
import matplotlib.pyplot as plt
import pandas as pd


def get_gene_name_map(filepath):
    """Loads a map from gene ID to gene name."""
    try:
        df = pd.read_csv(filepath, sep="\t")
        return df.set_index('gene_id')['gene_name'].to_dict()
    except (FileNotFoundError, KeyError):
        return {}


def plot_random_corr_curve(random_cutoff, random_std, output_path):
    """Plots the random correlation curve with error bars."""
    if not random_cutoff:
        return

    keys = sorted(random_cutoff.keys())
    values = [random_cutoff[key] for key in keys]
    std_devs = [random_std.get(key, 0) for key in keys]

    plt.figure(figsize=(10, 6))
    plt.errorbar(keys, values, yerr=std_devs, fmt='o', capsize=5, capthick=1, ecolor='red', markerfacecolor='blue')
    plt.xlabel('Number of Genes in Group')
    plt.ylabel('Average Pairwise Spearman Correlation')
    plt.title('Random Correlation vs. Gene Set Size')
    plt.grid(True)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


def save_results(df, output_path):
    """Saves a DataFrame to a TSV file."""
    df.to_csv(output_path, sep='\t', index=False)
