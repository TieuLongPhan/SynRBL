import unittest

from synrbl.SynRuleImputer import AutomaticSmilesExtraction


class TestAutomaticSmilesExtraction(unittest.TestCase):
    def setUp(self):
        # Sample reactions data
        self.reactions = [
            {"reactants": "CCO", "products": "CCOCC"},
            {"reactants": "CC", "products": "C"},
        ]
        self.extractor = AutomaticSmilesExtraction(self.reactions, n_jobs=1, verbose=0)

    def test_get_smiles(self):
        # Test extraction of SMILES strings
        smiles = AutomaticSmilesExtraction.get_smiles(self.reactions)
        self.assertEqual(smiles, ["CCO", "CCOCC", "CC", "C"])

    def test_calculate_mol_weight(self):
        # Test calculation of molecular weight
        weight = AutomaticSmilesExtraction.calculate_mol_weight("CCO")
        self.assertAlmostEqual(weight, 46.069, places=2)

    def test_count_carbon_atoms(self):
        # Test counting carbon atoms
        n_C = AutomaticSmilesExtraction.count_carbon_atoms("CCO")
        self.assertEqual(n_C, 2)

    def test_get_fragments(self):
        # Test filtering fragments based on criteria
        input_dict = {"smiles": ["CCO", "C"], "mw": [46.069, 16.043], "n_C": [2, 1]}
        filtered = AutomaticSmilesExtraction.get_fragments(
            input_dict, mw=50, n_C=1, combination="union"
        )
        self.assertEqual(filtered["smiles"], ["CCO", "C"])


if __name__ == "__main__":
    unittest.main()
