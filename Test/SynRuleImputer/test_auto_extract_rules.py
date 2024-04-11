import unittest

from synrbl.SynRuleImputer import AutomaticRulesExtraction


class TestAutomaticRulesExtraction(unittest.TestCase):
    def setUp(self):
        self.extractor = AutomaticRulesExtraction()

    def test_smiles_to_molecular_formula(self):
        # Test conversion from SMILES to molecular formula
        smiles = "CCO"
        formula = AutomaticRulesExtraction.smiles_to_molecular_formula(smiles)
        self.assertEqual(formula, "C2H6O")

    def test_add_new_entries(self):
        # Test adding new entries
        filtered_fragments = {"smiles": ["CCO", "CC"]}
        self.extractor.add_new_entries(filtered_fragments)
        self.assertEqual(len(self.extractor.new_smiles_dict), 2)
        self.assertEqual(self.extractor.new_smiles_dict[0]["smiles"], "CCO")

    def test_extract_rules(self):
        # Test extracting rules
        filtered_fragments = {"smiles": ["CCO", "CC"]}
        self.extractor.add_new_entries(filtered_fragments)
        rules = self.extractor.extract_rules()
        self.assertEqual(len(rules), 2)
        self.assertIn("smiles", rules[0])
        self.assertIn("formula", rules[0])


if __name__ == "__main__":
    unittest.main()
