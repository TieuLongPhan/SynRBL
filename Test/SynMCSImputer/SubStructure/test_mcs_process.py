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


if __name__ == '__main__':
    unittest.main()
