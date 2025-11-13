import unittest
import pandas as pd
import numpy as np
import os
from unittest.mock import patch, mock_open

# Adjust import path to reflect the new package structure
from clusteringgo.data_processing import impute_zeros, get_metadata, zscore_all_by_pbs


class TestDataProcessing(unittest.TestCase):

    def setUp(self):
        """Set up test data for all test cases."""
        self.metadata = pd.DataFrame({
            'ID': ['M1', 'M2', 'M3', 'M4', 'M5', 'M6'],
            'Drug': ['A', 'PBS', 'A', 'PBS', 'A', 'PBS'],
            'Treatment': ['T1', 'T1', 'T1', 'T1', 'T2', 'T2'],
            'Sample': ['S1', 'S2', 'S3', 'S4', 'S5', 'S6'],
            'New/Old': ['O', 'O', 'O', 'O', 'O', 'O']
        })
        self.raw_data = pd.DataFrame({
            'M1': [10, 0, 30],
            'M2': [5, 15, 0],
            'M3': [12, 5, 33],
            'M4': [4, 12, 4]
        }, index=['G1', 'G2', 'G3'])

    def test_impute_zeros(self):
        """Test that zeros are correctly imputed."""
        data_with_zeros = self.raw_data.copy()
        imputed_data = impute_zeros(data_with_zeros, self.metadata, 'Treatment')

        # Check that no zeros remain
        self.assertFalse((imputed_data == 0).any().any(), "Zeros should have been imputed.")

        # G2, M1 is 0. Group (A, T1) has M3 with value 5. So M1 should be 5.
        self.assertEqual(imputed_data.loc['G2', 'M1'], 5)
        # G3, M2 is 0. Group (PBS, T1) has M4 with value 4. So M2 should be 4.
        self.assertEqual(imputed_data.loc['G3', 'M2'], 4)

    def test_impute_zeros_no_valid_group(self):
        """Test imputation when a group has no other valid samples."""
        metadata = pd.DataFrame({
            'ID': ['M1', 'M2'], 'Drug': ['A', 'PBS'], 'Treatment': ['T1', 'T1']
        })
        data = pd.DataFrame({'M1': [0], 'M2': [10]}, index=['G1'])
        imputed = impute_zeros(data, metadata, 'Treatment')
        # M1 has no other 'A' drug samples, so it should fill with the row min (10)
        self.assertEqual(imputed.loc['G1', 'M1'], 10)

    @patch("pandas.read_excel")
    @patch("pandas.read_csv")
    def test_get_metadata_filtering(self, mock_read_csv, mock_read_excel):
        """Test the metadata filtering based on QC stats."""
        # Mock the file reads
        mock_read_excel.return_value = self.metadata
        mock_qc_data = pd.DataFrame({
            'Sample Name': ['S1', 'S2', 'S3', 'S4', 'S5'],
            'aligned': [0.9, 0.8, 0.4, 0.7, 0.9]  # S3 is below threshold
        })
        mock_read_csv.return_value = mock_qc_data

        # Mock os.path.exists to return True
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = True

            # Run the function
            filtered_meta = get_metadata(folder=".", qc_file_suffix="test", filter_threshold=0.55)

            # Check that S3 was filtered out
            self.assertEqual(len(filtered_meta), 4)
            self.assertNotIn('S3', filtered_meta['Sample'].values)
            self.assertIn('S1', filtered_meta['Sample'].values)
            self.assertIn('S4', filtered_meta['Sample'].values)

    def test_zscore_all_by_pbs(self):
        """Test z-scoring relative to PBS controls."""
        data = pd.DataFrame({
            'M1': [10, 20],  # Drug A, T1
            'M2': [2, 10],  # PBS, T1
            'M3': [12, 25],  # Drug A, T1
            'M4': [4, 12],  # PBS, T1
            'M5': [100, 200],  # Drug A, T2
            'M6': [50, 150]  # PBS, T2
        }, index=['G1', 'G2'])

        # For T1, PBS mean/std for G1 is (2+4)/2=3, std=1.414. For G2 is 11, std=1.414
        # For T2, PBS mean/std for G1 is 50, std=N/A->1. For G2 is 150, std=N/A->1
        zscored = zscore_all_by_pbs(data, self.metadata)

        # Check a value from T1. Uses sample std dev (ddof=1 by default in pandas).
        # G1, M1: (10 - 3) / 1.414 = 4.95
        self.assertAlmostEqual(zscored.loc['G1', 'M1'], (10 - 3) / np.std([2, 4], ddof=1), places=2)

        # Check a value from T2
        # G1, M5: (100 - 50) / 1 = 50 (std is 1 because only one sample)
        self.assertAlmostEqual(zscored.loc['G1', 'M5'], 50.0, places=2)

        # Check a PBS value itself (should be z-scored)
        # G1, M2: (2 - 3) / 1.414 = -0.707. Uses sample std dev (ddof=1).
        self.assertAlmostEqual(zscored.loc['G1', 'M2'], (2 - 3) / np.std([2, 4], ddof=1), places=2)


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)

