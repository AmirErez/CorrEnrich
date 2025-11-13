"""
Handles data loading, cleaning, normalization, and imputation.
"""
import os
import numpy as np
import pandas as pd

# Constants moved to run_analysis script or derived dynamically
MITOCHONDRIAL_GENES = [
    "mt-nd1", "mt-nd2", "mt-nd3", "mt-nd4", "mt-nd4l", "mt-nd5", "mt-nd6",
    "mt-co1", "mt-co2", "mt-co3", "mt-cytb", "mt-atp6", "mt-atp8", "mt-tf",
    "mt-tv", "mt-tl1", "mt-ti", "mt-tq", "mt-tm", "mt-tw", "mt-ta", "mt-tn",
    "mt-tc", "mt-ty", "mt-ts1", "mt-td", "mt-tk", "mt-tg", "mt-tr", "mt-th",
    "mt-ts2", "mt-tl2", "mt-te", "mt-tt", "mt-tp", "mt-rnr1", "mt-rnr2"
]


def get_metadata(folder, qc_file_suffix="", only_old=True, filter_threshold=0.55):
    """
    Reads and filters metadata based on QC stats.
    Args:
        folder (str): Path to the data folder containing metadata.xlsx.
        qc_file_suffix (str): Suffix for the QC stats file.
        only_old (bool): If True, only include 'Old' samples.
        filter_threshold (float): Minimum 'aligned' value to keep a sample.
    Returns:
        pd.DataFrame: Filtered metadata.
    """
    meta_path = os.path.join(folder, "metadata.xlsx")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at {meta_path}")
    meta = pd.read_excel(meta_path)
    meta['ID'] = meta.apply(lambda row: row['ID'] + 'N' if row['New/Old'] == 'N' else row['ID'], axis=1)
    meta['Drug'] = meta['Drug'].replace({
        'mix': 'Mix', 'ampicillin': 'Amp', 'Control ': 'PBS',
        'METRO': 'Met', 'NEO': 'Neo', 'VANCO': 'Van'
    })

    if filter_threshold:
        qc_file = f"RASflow stats {qc_file_suffix}.csv"
        qc_path = os.path.join(folder, qc_file)
        if not os.path.exists(qc_path):
            raise FileNotFoundError(f"QC file not found at {qc_path}")
        qc = pd.read_csv(qc_path)
        samples = qc[qc['aligned'] > filter_threshold]['Sample Name']
        meta = meta[meta['Sample'].isin(samples)]

    if only_old:
        meta = meta[~meta['ID'].str.endswith('N')]
    return meta


def read_process_files(data_folder, new=False, filter_value=0.55, remove_mitochondrial=True, use_gene_name=False):
    """
    Reads and processes the raw gene expression files.
    Args:
        data_folder (str): The root folder containing the data files.
        new (bool): Whether to include new data.
        filter_value (float): QC filter threshold.
        remove_mitochondrial (bool): If True, removes mitochondrial genes.
        use_gene_name (bool): If True, use gene names as index, otherwise gene IDs.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Processed transcriptome data and metadata.
    """
    norm_folder = os.path.join(data_folder, "new normalization")
    transcriptome_path = os.path.join(norm_folder, "transcriptome_2023-09-17-genes_norm_named.tsv")

    if not os.path.exists(transcriptome_path):
        raise FileNotFoundError(f"Transcriptome file not found at: {transcriptome_path}")

    transcriptome_df = pd.read_csv(transcriptome_path, sep="\t")

    index_col = "gene_name" if use_gene_name else "gene_id"
    drop_col = "gene_id" if use_gene_name else "gene_name"

    transcriptome_df[index_col] = transcriptome_df.apply(
        lambda row: row["gene_id"] if pd.isna(row[index_col]) else row[index_col], axis=1)
    transcriptome_df = transcriptome_df.set_index(index_col).drop(drop_col, axis=1)

    qc_suffix = "2023_09_17"
    metadata = get_metadata(norm_folder, qc_file_suffix=qc_suffix, only_old=not new, filter_threshold=filter_value)

    id_map = metadata.set_index('Sample')['ID'].to_dict()
    transcriptome_df = transcriptome_df.rename(columns=id_map)
    transcriptome_df = transcriptome_df[[col for col in transcriptome_df.columns if col in metadata["ID"].values]]

    # merge big abx:
    new_path = os.path.join(norm_folder, r"mRNA_NEBNext_20200908")
    new_data = pd.read_csv(os.path.join(new_path, "mRNA_NEBNext_20200908_genes_norm_named.tsv"), sep="\t")
    # sum rows with the same gene_name and drop the gene_id column
    # new_data = new_data.drop("gene_id", axis=1).groupby("gene_name").sum()
    new_stats = pd.read_csv(os.path.join(new_path,r"big_abx_stats.csv"))
    # remove all samples with "aligned" < 0.5
    columns_to_keep = new_stats[new_stats["aligned"] > filter_value]["Sample Name"]
    # new_data = new_data[columns_to_keep.append(pd.Series(["gene_name", "gene_id"]))]
    columns_to_keep = columns_to_keep.tolist()  # Convert to list if needed
    columns_to_keep.append("gene_name")  # Append to the list
    columns_to_keep.append("gene_id")
    new_data.columns = [col.split("_")[-1] if "gene" not in col else col for col in new_data.columns]
    # drop columns C1, C2, C3 as they already exist in the other df
    new_data = new_data.drop(["C1", "C2", "C3"], axis=1)


    new_data[index_col] = new_data.apply(lambda row: row["gene_id"] if pd.isna(row[index_col]) else row[index_col], axis=1)
    new_data = new_data.set_index(index_col).drop(drop_col, axis=1)

    transcriptome_df = pd.merge(transcriptome_df, new_data, left_index=True, right_index=True)
    new_metadata = get_metadata(norm_folder, qc_file_suffix=qc_suffix, only_old=not new, filter_threshold=False)
    new_metadata = new_metadata[new_metadata["ID"].isin(new_data.columns)]
    metadata = pd.concat([metadata, new_metadata])


    transcriptome_df = transcriptome_df.groupby(transcriptome_df.index).sum()

    # Remove sparse genes
    transcriptome_zeros = transcriptome_df[transcriptome_df == 0].count(axis=1)
    transcriptome_sparse = transcriptome_zeros[transcriptome_zeros > 0.5 * transcriptome_df.shape[1]]
    transcriptome_df = transcriptome_df.drop(transcriptome_sparse.index)

    if remove_mitochondrial:
        matching_indices = transcriptome_df.index[
            transcriptome_df.index.str.lower().isin(set(MITOCHONDRIAL_GENES))].tolist()
        transcriptome_df = transcriptome_df.drop(matching_indices, errors='ignore')

    # Normalize to TPM/RPM
    transcriptome_df = (transcriptome_df * 1e6).divide(transcriptome_df.sum(axis=0), axis=1)

    # Clean up samples
    to_remove = ["C9", "C10", "C18", "M13", "V14"]
    transcriptome_df = transcriptome_df.drop(to_remove, axis=1, errors='ignore')
    metadata = metadata[~metadata["ID"].isin(to_remove)]

    return transcriptome_df, metadata


# def impute_zeros(to_impute, meta_data, condition, missing_threshold=.2):
#     """
#     Replaces all zeros by the minimum of other gene expressions of the same group.
#     Args:
#         to_impute (pd.DataFrame): DataFrame with zeros to impute.
#         meta_data (pd.DataFrame): Metadata for grouping.
#         condition (str): The metadata column to group by (e.g., 'Treatment').
#     Returns:
#         pd.DataFrame: Data with zeros imputed.
#     """
#     imputed_df = to_impute.copy()
#     imputed_df = imputed_df.replace(0, np.nan)
#
#     rows, cols = np.where(imputed_df.isnull())
#
#     for r, c in zip(rows, cols):
#         sample_id = imputed_df.columns[c]
#         gene_id = imputed_df.index[r]
#
#         sample_info = meta_data[meta_data['ID'] == sample_id]
#         if sample_info.empty:
#             continue
#
#         drug = sample_info['Drug'].values[0]
#         treatment = sample_info[condition].values[0]
#
#         mice_ids = meta_data[(meta_data['Drug'] == drug) & (meta_data[condition] == treatment) &
#                              (meta_data['ID'] != sample_id)]['ID']
#
#         group_values = imputed_df.loc[gene_id, mice_ids].dropna()
#
#         if not group_values.empty:
#             imputed_df.iloc[r, c] = np.min(group_values)
#         else:
#             # If all other samples in the group are also NaN, fill with a very small number
#             # or the global min for that gene. For now, we'll keep it NaN to be handled later.
#             pass
#
#     # If any NaNs remain (e.g., single-sample groups), fill them with the row minimum
#     imputed_df = imputed_df.apply(lambda row: row.fillna(row.min()), axis=1)
#     # If a whole row was NaN, it will still be NaN. Fill with a global small value.
#     imputed_df = imputed_df.fillna(1e-6)
#
#
#     return imputed_df
def impute_zeros(genes_df, meta_data, condition, missing_threshold=0.2):
    """
    Efficiently imputes missing values in a gene expression DataFrame using a vectorized approach.

    The process is as follows:
    1. Replaces all 0s with NaN.
    2. (Preprocessing) Removes genes (rows) with a high percentage of missing values.
    3. Imputes NaNs with the minimum value of their corresponding sample group.
       A group is defined by the unique combination of 'Drug' and the specified 'condition' column.
    4. Handles any remaining NaNs by first using the gene's row-minimum, and finally a small constant.

    Args:
        genes_df (pd.DataFrame): DataFrame of gene expressions (genes x samples).
        meta_data (pd.DataFrame): Metadata where rows correspond to samples. Must contain
                                  'ID', 'Drug', and the column specified by `condition`.
        condition (str): The metadata column to group by along with 'Drug' (e.g., 'Treatment').
        missing_threshold (float): The proportion of missing values (e.g., 0.2 for 20%)
                                   above which a gene will be removed.

    Returns:
        pd.DataFrame: A new DataFrame with missing values imputed, without the filtered-out genes.
    """
    # 1. Prepare the DataFrame for imputation
    imputed_df = genes_df.copy()
    imputed_df.replace(0, np.nan, inplace=True)

    # 2. Pre-processing: Remove rows with too many missing values
    n_samples = imputed_df.shape[1]
    # Keep only the rows where the number of nulls is below the threshold
    imputed_df = imputed_df.loc[imputed_df.isnull().sum(axis=1) < n_samples * missing_threshold]

    # --- Vectorized Imputation ---
    # Transpose the DataFrame so we can group the samples (columns) efficiently
    imputed_T = imputed_df.T # Now it's samples x genes

    # Create a grouper series from the metadata that aligns with the transposed DataFrame's index
    # This creates a unique group label (e.g., ('DrugA', 'TreatmentX')) for each sample ID
    meta_data_indexed = meta_data.set_index('ID')
    grouper = meta_data_indexed.loc[imputed_T.index].apply(
        lambda x: (x['Drug'], x[condition]), axis=1
    )

    # 3. Use groupby().transform() to calculate group minimums
    # .transform('min') calculates the min for each group and broadcasts the result
    # back to the original shape of imputed_T. This is the key to vectorization. 🚀
    group_mins = imputed_T.groupby(grouper).transform('min')

    # Fill the NaNs in our data with the calculated group minimums
    imputed_T.fillna(group_mins, inplace=True)

    # Transpose back to the original orientation (genes x samples)
    imputed_df = imputed_T.T

    # 4. Handle any remaining NaNs (for cases where a whole group was NaN for a gene)
    # First fallback: fill with the row (gene) minimum
    imputed_df = imputed_df.apply(lambda row: row.fillna(row.min()), axis=1)

    # Second fallback: fill any remaining NaNs (from all-NaN rows) with a tiny value
    imputed_df.fillna(1e-6, inplace=True)

    return imputed_df

def zscore_all_by_pbs(data, metadata):
    """
    Calculates z-score for each gene based on the PBS control group for each treatment.
    Args:
        data (pd.DataFrame): The gene expression data.
        metadata (pd.DataFrame): The sample metadata.
    Returns:
        pd.DataFrame: Z-scored data.
    """
    zscored_data = data.copy()
    for treat in metadata['Treatment'].unique():
        pbs_samples = metadata[(metadata['Drug'] == "PBS") & (metadata["Treatment"] == treat)]['ID']

        # Ensure we only use samples present in the data columns
        pbs_samples_in_data = [s for s in pbs_samples if s in data.columns]
        if not pbs_samples_in_data:
            continue

        pbs_data = data[pbs_samples_in_data]
        pbs_mean = pbs_data.mean(axis=1)
        pbs_std = pbs_data.std(axis=1)
        # Avoid division by zero for groups with no variance
        pbs_std[pbs_std == 0] = 1
        pbs_std[np.isnan(pbs_std)] = 1

        treatment_samples = metadata[metadata["Treatment"] == treat]['ID']
        treatment_samples_in_data = [s for s in treatment_samples if s in data.columns]

        for sample in treatment_samples_in_data:
            zscored_data[sample] = (data[sample] - pbs_mean) / pbs_std

    return zscored_data.fillna(0) # Fill any NaNs that may result from missing groups


def transform_data(data, metadata):
    """
    Applies a full transformation pipeline: impute zeros, log2 transform, and z-score.
    Args:
        data (pd.DataFrame): Raw (but normalized) data.
        metadata (pd.DataFrame): Sample metadata.
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Transformed data and metadata.
    """
    data_imputed = impute_zeros(data, metadata, 'Treatment')
    data_log = np.log2(data_imputed)
    data_zscored = zscore_all_by_pbs(data_log, metadata)
    return data_zscored, metadata


