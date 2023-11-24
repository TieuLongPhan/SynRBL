import unittest
import sys
sys.path.append('../../')
from SynRBL.SynExtract.rsmi_decomposer import RSMIDecomposer, decompose  

class TestRSMIDecomposer(unittest.TestCase):

    def setUp(self):
        # Setup code for the tests
        self.smiles_string = 'CCO'
        self.decomposer = RSMIDecomposer(smiles=self.smiles_string, parallel=False)

    def test_calculate_mol_weight(self):
        # Test for the calculate_mol_weight method
        expected_weight = 46.069  # Expected molecular weight for 'CCO'
        calculated_weight = self.decomposer.calculate_mol_weight(self.smiles_string)
        self.assertAlmostEqual(calculated_weight, expected_weight, places=3)

    def test_data_decomposer(self):
        # Test for the data_decomposer method
        test_data = [{'reactants': 'CCO', 'products': 'C=O'}]
        decomposer = RSMIDecomposer(data=test_data, parallel=False)
        reactants, products = decomposer.data_decomposer()

        expected_reactants = [{'6': 2, '8': 1, '0': 0}]  # Expected composition for 'CCO'
        self.assertEqual(reactants, expected_reactants)

    def test_decompose_function(self):
        # Test for the standalone decompose function
        smiles = 'CCO'
        expected_composition = {'6': 2, '8': 1, '0': 0}  # Expected atomic composition for 'CCO'
        composition = decompose(smiles)
        self.assertEqual(composition, expected_composition)

if __name__ == '__main__':
    unittest.main()
