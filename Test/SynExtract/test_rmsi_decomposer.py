import sys
from pathlib import Path
import unittest
import pandas as pd
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
from SynRBL.SynExtract.rsmi_decomposer import RSMIDecomposer  


class TestRSMIDecomposer(unittest.TestCase):

    def setUp(self):
        self.decomposer = RSMIDecomposer()

    def test_decompose_valid(self):
        # Test decompose method with a valid SMILES string
        smiles = 'CCO'
        composition = RSMIDecomposer.decompose(smiles)
        self.assertEqual(composition, {'C': 2, 'O': 1, 'H': 6, 'Q': 0})

    def test_decompose_invalid(self):
        # Test decompose method with an invalid SMILES string
        smiles = 'InvalidString'
        composition = RSMIDecomposer.decompose(smiles)
        self.assertIsNone(composition)

    def test_data_decomposer_valid(self):
        # Test data_decomposer method with valid data
        data = pd.DataFrame({'reactants': ['CCO', 'CC'], 'products': ['C=O', 'C=C']})
        decomposer = RSMIDecomposer(data=data, parallel=False)
        reactants, products = decomposer.data_decomposer()
        self.assertEqual(len(reactants), 2)
        self.assertEqual(len(products), 2)

    def test_data_decomposer_invalid(self):
        # Test data_decomposer method with invalid data
        data = pd.DataFrame({'reactants': ['InvalidString', 'CC'], 'products': ['C=O', 'InvalidString']})
        decomposer = RSMIDecomposer(data=data, parallel=False)
        reactants, products = decomposer.data_decomposer()
        self.assertEqual(len(reactants), 2)
        self.assertEqual(len(products), 2)
        self.assertIsNone(reactants[0])
        self.assertIsNone(products[1])

if __name__ == '__main__':
    unittest.main()
