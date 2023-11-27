import sys
from pathlib import Path
import unittest

# Calculate the path to the root directory (two levels up)
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))

from SynRBL.SynCleaning import SMILESStandardizer

class TestSMILESStandardizer(unittest.TestCase):

    def setUp(self):
        self.standardizer = SMILESStandardizer()

    def test_standardize_smiles_simple(self):
        # Test standardization of a simple molecule
        smiles = 'CCO'
        standardized_smiles = self.standardizer.standardize_smiles(smiles)
        self.assertEqual(standardized_smiles, 'CCO')  # Expected standardized SMILES

    def test_standardize_smiles_salt(self):
        # Test salt removal
        smiles = 'CCO.Na'
        standardized_smiles = self.standardizer.standardize_smiles(smiles, remove_salts=True)
        self.assertEqual(standardized_smiles, 'CCO')  # Expected SMILES after salt removal

    def test_standardize_reaction(self):
        # Test reaction standardization
        reaction_smiles = 'CCO>>CCOC'
        standardized_reaction = self.standardizer.standardize_reaction(reaction_smiles)
        self.assertEqual(standardized_reaction, 'CCO>>CCOC')  # Expected standardized reaction SMILES

    def test_standardize_invalid_smiles(self):
        # Test handling of invalid SMILES strings
        invalid_smiles = 'XYZ'
        result = self.standardizer.standardize_smiles(invalid_smiles)
        self.assertIsNone(result)  # Expect None for invalid SMILES

    # Additional tests for other methods and edge cases can be added here

if __name__ == '__main__':
    unittest.main()
