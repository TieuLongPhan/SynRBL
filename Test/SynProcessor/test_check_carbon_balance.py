import unittest

from synrbl.SynProcessor.check_carbon_balance import CheckCarbonBalance


class TestCheckCarbonBalance(unittest.TestCase):
    def setUp(self):
        self.reactions_balanced = [{"reactions": "CCCC>>C1CCC1"}]
        self.reactions_unbalanced_products_impute = [{"reactions": "CC>>C"}]
        self.reactions_unbalanced_reactants_impute = [{"reactions": "C>>CC"}]
        self.reactions_invalid = [{"reactions": "InvalidSMILES>>C"}]

    def test_count_atoms_valid(self):
        smiles = "CCO"
        count = CheckCarbonBalance.count_atoms(smiles, "C", {})
        self.assertEqual(count, 2)

    def test_count_atoms_invalid(self):
        smiles = "InvalidSMILES"
        count = CheckCarbonBalance.count_atoms(smiles, "C", {})
        self.assertEqual(count, 0)  # InvalidSMiles will count no Carbon

    def test_check_carbon_balance_balanced(self):
        checker = CheckCarbonBalance(self.reactions_balanced)
        results = checker.check_carbon_balance()
        self.assertEqual(results[0]["carbon_balance_check"], "balanced")

    def test_check_carbon_balance_unbalanced_products_impute(self):
        checker = CheckCarbonBalance(self.reactions_unbalanced_products_impute)
        results = checker.check_carbon_balance()
        self.assertEqual(results[0]["carbon_balance_check"], "products")

    def test_check_carbon_balance_unbalanced_reactants_impute(self):
        checker = CheckCarbonBalance(self.reactions_unbalanced_reactants_impute)
        results = checker.check_carbon_balance()
        self.assertEqual(results[0]["carbon_balance_check"], "reactants")

    def test_check_carbon_balance_invalid(self):
        checker = CheckCarbonBalance(self.reactions_invalid)
        results = checker.check_carbon_balance()
        self.assertEqual(results[0]["carbon_balance_check"], "reactants")


# Run the tests
if __name__ == "__main__":
    unittest.main()
