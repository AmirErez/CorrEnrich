"""
Example script to run the full ClusteringGO analysis pipeline.
"""
import os
import argparse
import pandas as pd
from anytree import PreOrderIter
from tqdm import tqdm # <-- IMPORT FOR PROGRESS BAR

from clusteringgo.data_processing import read_process_files, transform_data
from clusteringgo.tree import build_tree
from clusteringgo.stats import (
    average_pairwise_spearman, get_random_corr, median_mwu,
    split_genes_by_trend, calculate_pvalue_from_ecdf
)
from clusteringgo.utils import plot_random_corr_curve, save_results, get_gene_name_map

def run_analysis(data_folder, output_dir, primary_condition_col, control_value, secondary_condition_col=None):
    """
    Main function to execute the gene clustering and correlation analysis.

    Args:
        data_folder (str): Path to the root data directory.
        output_dir (str): Path to save results and plots.
        primary_condition_col (str): The main metadata column to test (e.g., 'Drug').
        control_value (str): The value in `primary_condition_col` that is the control (e.g., 'PBS').
        secondary_condition_col (str, optional): A second column for nested analysis (e.g., 'Treatment').
    """
    print("Step 1: Reading and processing data...")
    os.makedirs(output_dir, exist_ok=True)
    transcriptome, metadata = read_process_files(data_folder)

    print("Step 2: Transforming data (impute, log2, z-score)...")
    data, metadata = transform_data(transcriptome, metadata)

    print("Step 3: Building Gene Ontology tree...")
    tree, _ = build_tree(data_dir=output_dir)

    print("Step 4: Calculating correlations and significance...")
    all_results = []

    gene_map_path = os.path.join(data_folder, "new normalization", "transcriptome_2023-09-17-genes_norm_named.tsv")
    id_to_name = get_gene_name_map(gene_map_path)

    primary_conditions = [c for c in metadata[primary_condition_col].unique() if c != control_value]
    secondary_conditions = metadata[secondary_condition_col].unique() if secondary_condition_col else [None]

    # Convert tree iterator to a list ONCE to get the total count for the progress bar
    nodes_to_process = list(PreOrderIter(tree))
    total_nodes = len(nodes_to_process)

    for primary_val in primary_conditions:
        for secondary_val in secondary_conditions:
            if secondary_val:
                current_meta = metadata[
                    ((metadata[primary_condition_col] == primary_val) | (metadata[primary_condition_col] == control_value)) &
                    (metadata[secondary_condition_col] == secondary_val)
                ]
                condition_name_desc = f"{primary_val}-{secondary_val}"
            else:
                current_meta = metadata[
                    (metadata[primary_condition_col] == primary_val) | (metadata[primary_condition_col] == control_value)
                ]
                condition_name_desc = primary_val

            print(f"\n--- Analyzing: {condition_name_desc} ---")

            current_sample_ids = [s for s in current_meta['ID'] if s in data.columns]
            current_expression = data[current_sample_ids]

            if current_expression.empty or len(current_meta[primary_condition_col].unique()) < 2:
                print(f"Skipping {condition_name_desc} due to insufficient data.")
                continue

            results_list = []
            random_cutoff, random_std, ecdf_storage = {}, {}, {}

            # --- WRAP YOUR LOOP WITH TQDM ---
            # This shows a progress bar: e.g., "Amp-IP: 25%|██▌ | 5000/20000 [00:10<00:30, 499.50term/s]"
            for node in tqdm(nodes_to_process, desc=condition_name_desc, unit="term", total=total_nodes):
                if not node.gene_set:
                    continue

                genes_in_data = list(node.gene_set.intersection(current_expression.index))
                if len(genes_in_data) < 2:
                    continue

                enhanced, suppressed = split_genes_by_trend(
                    primary_val, control_value, genes_in_data, current_expression, metadata,
                    primary_condition_col, secondary_condition_col, secondary_val
                )

                for trend_genes, trend_label in [(enhanced, 'enhanced'), (suppressed, 'suppressed')]:
                    if len(trend_genes) < 2:
                        continue

                    correlation = average_pairwise_spearman(current_expression.loc[trend_genes])
                    if pd.isna(correlation):
                        continue

                    size_category = round(len(trend_genes) / 10) * 10 if len(trend_genes) > 50 else len(trend_genes)
                    if size_category > 1 and size_category not in random_cutoff:
                        # This expensive step is now parallelized (see stats.py)
                        rc, rs, ecdf = get_random_corr(size_category, current_expression)
                        random_cutoff[size_category], random_std[size_category], ecdf_storage[size_category] = rc, rs, ecdf

                    _, mwu_p_value = median_mwu(
                        primary_val, control_value, trend_genes, current_expression, metadata,
                        primary_condition_col, secondary_condition_col, secondary_val
                    )

                    p_val_corr = 1.0
                    if size_category in ecdf_storage:
                        tail = 'upper' if trend_label == 'enhanced' else 'lower'
                        p_val_corr = calculate_pvalue_from_ecdf(correlation, ecdf_storage[size_category], tail=tail)

                    gene_names = [id_to_name.get(g, g) for g in trend_genes]

                    results_list.append({
                        primary_condition_col: primary_val,
                        secondary_condition_col if secondary_condition_col else 'Group': secondary_val if secondary_val else 'All',
                        'GO_Term': node.go_id,
                        'GO_Name': node.name,
                        'Trend': trend_label,
                        'N_Genes': len(trend_genes),
                        'Correlation': correlation,
                        'Correlation_PValue': p_val_corr,
                        'Random_Corr_Mean': random_cutoff.get(size_category),
                        'MWU_PValue': mwu_p_value,
                        'Genes': ','.join(trend_genes),
                        'Gene_Names': ','.join(gene_names)
                    })

            if results_list:
                results_df = pd.DataFrame(results_list)
                condition_name = f"{primary_val}_{secondary_val}" if secondary_val else primary_val
                condition_results_path = os.path.join(output_dir, f'results_{condition_name}.tsv')
                save_results(results_df, condition_results_path)
                all_results.append(results_df)

                plot_path = os.path.join(output_dir, f'random_corr_{condition_name}.png')
                plot_random_corr_curve(random_cutoff, random_std, plot_path)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)
        final_path = os.path.join(output_dir, 'all_go_term_results.tsv')
        save_results(final_df, final_path)
        print(f"\nAnalysis complete. All results saved to {final_path}")
    else:
        print("\nAnalysis complete. No results were generated.")


def main():
    parser = argparse.ArgumentParser(description="Run the ClusteringGO analysis pipeline.")
    parser.add_argument("data_dir", help="Path to the root data directory.")
    parser.add_argument("output_dir", help="Path to save results and plots.")
    parser.add_argument("--primary_col", required=True, help="The main metadata column to test (e.g., 'Drug').")
    parser.add_argument("--control_val", required=True, help="The control value in the primary column (e.g., 'PBS').")
    parser.add_argument("--secondary_col", default=None, help="Optional second column for nested analysis (e.g., 'Treatment').")

    args = parser.parse_args()

    if not os.path.exists(args.data_dir):
        print(f"ERROR: Data directory not found at '{args.data_dir}'")
    else:
        run_analysis(args.data_dir, args.output_dir, args.primary_col, args.control_val, args.secondary_col)

if __name__ == "__main__":
    main()

