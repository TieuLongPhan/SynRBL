import unittest

from rdkit import Chem

from synrbl.SynRuleImputer import SyntheticRuleImputer


# not fixed
class TestSyntheticRuleImputer(unittest.TestCase):
    def setUp(self):
        self.rule_dict = [
            {"smiles": "C1=CC=CC=C1", "Composition": {"C": 6, "H": 6}},
            {"smiles": "CCO", "Composition": {"C": 2, "H": 6, "O": 1}},
        ]
        self.imputer = SyntheticRuleImputer(self.rule_dict)

    def test_single_impute(self):
        # Test single imputation of missing data
        missing_data = {
            "Diff_formula": {"C": 4, "H": 4, "Q": 0},
            "Unbalance": "Products",
            "reactants": "C2H6",
            "products": "C2H6",
        }
        imputed_data = SyntheticRuleImputer.single_impute(missing_data, self.rule_dict)

        # Check if 'new_reaction' is added to the dictionary, which implies
        # successful imputation
        if any(rule["smiles"] in imputed_data["products"] for rule in self.rule_dict):
            self.assertIn("new_reaction", imputed_data)
        else:
            self.assertNotIn("new_reaction", imputed_data)

    def test_parallel_impute(self):
        # Test parallel imputation of missing data
        missing_data_list = [
            {
                "Diff_formula": {"C": 4, "H": 4},
                "Unbalance": "Products",
                "reactants": "C2H6",
                "products": "C2H6",
            },
            {
                "Diff_formula": {"C": 3, "H": 6},
                "Unbalance": "Reactants",
                "reactants": "C2H6",
                "products": "C2H6",
            },
        ]
        imputed_data_list = self.imputer.parallel_impute(missing_data_list)
        self.assertEqual(len(imputed_data_list), len(missing_data_list))

    def test_get_and_validate_smiles(self):
        # Test concatenation and validation of SMILES strings
        solution = [{"smiles": "CCO", "Ratio": 1}]
        smiles = SyntheticRuleImputer.get_and_validate_smiles(solution)
        self.assertIsInstance(smiles, str)
        self.assertTrue(Chem.MolFromSmiles(smiles) is not None)


if __name__ == "__main__":
    unittest.main()
