import sys
import os
import tempfile
import json
from pathlib import Path
import pandas as pd
from rdkit.Chem import rdFMCS
root_dir = Path(__file__).parents[3]
sys.path.append(str(root_dir))
import unittest
from unittest.mock import patch, MagicMock
from SynRBL.SynMCSImputer.SubStructure.mcs_graph_detector import MCSMissingGraphAnalyzer
from SynRBL.SynMCSImputer.SubStructure.mcs_process import single_mcs, ensemble_mcs  




class TestMCSFunctions(unittest.TestCase):

    def setUp(self):
        # Example reaction data for testing
        self.sample_reaction_data = {
            'R-id': 'example_id',
            'reactants': 'CCO.C[O]',
            'products': 'CCO',
            'carbon_balance_check': 'balanced'
        }
        self.root_dir = Path('.')
        self.conditions = [{'RingMatchesRingOnly': True, 'CompleteRingsOnly': True, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': True}]

    @patch('SynRBL.SynMCSImputer.SubStructure.mcs_graph_detector.MCSMissingGraphAnalyzer.fit')
    def test_single_mcs(self, mock_fit):
        # Mocking MCSMissingGraphAnalyzer.fit to return predefined values
        mock_fit.return_value = ([], [], [], None)
        
        result = single_mcs(self.sample_reaction_data)
        self.assertEqual(result['R-id'], 'example_id')
        self.assertIsInstance(result['mcs_results'], list)
        self.assertIsInstance(result['sorted_reactants'], list)

    @patch('SynRBL.SynUtils.data_utils.save_database')
    @patch('joblib.Parallel')
    def test_ensemble_mcs(self, mock_parallel, mock_save_database):
        # Creating a temporary directory to mock save_dir
        with tempfile.TemporaryDirectory() as tmpdirname:
            save_dir = Path(tmpdirname)

            # Mocking parallel processing and save_database function
            mock_parallel.return_value = MagicMock()
            mock_parallel.return_value.__enter__.return_value = lambda x: [{} for _ in x]
            mock_save_database.return_value = None

            ensemble_mcs([self.sample_reaction_data], self.root_dir, save_dir, self.conditions, batch_size=1)

            # Check if the log file and output file are created
            self.assertTrue(os.path.exists(save_dir / 'process.log'))
            self.assertTrue(os.path.exists(save_dir / 'Condition_1.json.gz'))

            # Optional: Check the content of the output file
            with open(save_dir / 'Condition_1.json.gz', 'rt') as f:
                data = json.load(f)
                self.assertIsInstance(data, list)



if __name__ == '__main__':
    unittest.main()
