import sys
from pathlib import Path
import unittest

# Calculate the path to the root directory (two levels up)
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))

import unittest
from SynRBL.SynCleaning import SMILESStandardizer

class TestSMILESStandardizer(unittest.TestCase):

    def setUp(self):
        self.standardizer = SMILESStandardizer()

    def test_standardize_smiles_normalize(self):
        # Test normalization
        smiles = 'C1=CC=CC=C1'  # Aromatic compound for normalization
        normalized_smiles = self.standardizer.standardize_smiles(smiles,
                                                                normalizer=self.standardizer.normalizer, tautomer=self.standardizer.tautomer,
                                                                salt_remover=self.standardizer.salt_remover,normalize=True)
        self.assertEqual(normalized_smiles, 'C1=CC=CC=C1')  # Expected normalized SMILES

    def test_standardize_smiles_tautomerize(self):
        # Test tautomerization
        smiles = 'C1=CC=C2C(=C1)C=CC=C2'  # Tautomerizable compound
        tautomerized_smiles = self.standardizer.standardize_smiles(smiles, normalize=False, tautomerize=True,
                                                                    normalizer=self.standardizer.normalizer, tautomer=self.standardizer.tautomer,
                                                                    salt_remover=self.standardizer.salt_remover)
                                                                   
        self.assertEqual(tautomerized_smiles, 'C1=CC=CC=C1')  # Expected tautomerized SMILES

    def test_standardize_smiles_remove_salts(self):
        # Test salt removal
        smiles = 'CCO.Na'  # Compound with a salt (sodium)
        without_salts_smiles = self.standardizer.standardize_smiles(smiles, remove_salts=True,
                                                                    normalizer=self.standardizer.normalizer, tautomer=self.standardizer.tautomer,
                                                                    salt_remover=self.standardizer.salt_remover,normalize=True)
        self.assertEqual(without_salts_smiles, 'CCO')  # Expected SMILES after salt removal

    def test_standardize_smiles_handle_charges(self):
        # Test handling of charges
        smiles = 'CC[NH+](CC)CC'  # Compound with charge
        charged_smiles = self.standardizer.standardize_smiles(smiles, handle_charges=True,
                                                            normalizer=self.standardizer.normalizer, tautomer=self.standardizer.tautomer,
                                                            salt_remover=self.standardizer.salt_remover,normalize=True)
        self.assertEqual(charged_smiles, 'CCN(CC)CC')  # Expected SMILES with charges handled

    def test_standardize_smiles_handle_stereochemistry(self):
        # Test handling of stereochemistry
        smiles = 'CC(C)C(=O)O[C@@H]1C[C@H](O)[C@@H](CO)O1'  # Chiral compound
        stereo_handled_smiles = self.standardizer.standardize_smiles(smiles, tautomerize=True, handle_stereo=True,
                                                                    normalizer=self.standardizer.normalizer, tautomer=self.standardizer.tautomer,
                                                                    salt_remover=self.standardizer.salt_remover,normalize=True)
        self.assertEqual(stereo_handled_smiles, 'CC(C)C(=O)O[C@@H]1C[C@H](O)[C@@H](CO)O1')  # Expected SMILES with stereo handled

    def test_standardize_smiles_clean_radicals(self):
        # Test radical cleaning (custom logic required)
        smiles = 'CC(C)(C)[C](C)(C)C'  # Compound with radicals
        cleaned_radicals_smiles = self.standardizer.standardize_smiles(smiles, clean_radicals=True,
                                                                        normalizer=self.standardizer.normalizer, tautomer=self.standardizer.tautomer,
                                                                        salt_remover=self.standardizer.salt_remover,normalize=True)
        self.assertEqual(cleaned_radicals_smiles, 'CC(C)(C)[C](C)(C)C')  # Expected SMILES with radicals cleaned

    def test_standardize_smiles_dearomatize(self):
        # Test dearomatization
        smiles = 'C1=CC=CC=C1'  # Aromatic compound for dearomatization
        dearomatized_smiles = self.standardizer.standardize_smiles(smiles, dearomatize=True,
                                                                   normalizer=self.standardizer.normalizer, tautomer=self.standardizer.tautomer,
                                                                    salt_remover=self.standardizer.salt_remover,normalize=True)
        self.assertEqual(dearomatized_smiles, 'C1C=CC=CC1')  # Expected dearomatized SMILES

    def test_standardize_smiles_aromatize(self):
        # Test aromatization
        smiles = 'C1C=CC=CC1'  # Non-aromatic compound for aromatization
        aromatized_smiles = self.standardizer.standardize_smiles(smiles, aromatize=True,
                                                                 normalizer=self.standardizer.normalizer, tautomer=self.standardizer.tautomer,
                                                                    salt_remover=self.standardizer.salt_remover,normalize=True)
        self.assertEqual(aromatized_smiles, 'C1=CC=CC=C1')  # Expected aromatized SMILES

    def test_standardize_smiles_invalid(self):
        # Test handling of invalid SMILES strings
        invalid_smiles = 'XYZ'
        result = self.standardizer.standardize_smiles(invalid_smiles,
                                                      normalizer=self.standardizer.normalizer, tautomer=self.standardizer.tautomer,
                                                                    salt_remover=self.standardizer.salt_remover,normalize=True)
        self.assertIsNone(result)  # Expect None for invalid SMILES

if __name__ == '__main__':
    unittest.main()


