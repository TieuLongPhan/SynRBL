import unittest

from synrbl.SynRuleImputer import RuleConstraint


class TestRuleConstraint(unittest.TestCase):
    def setUp(self):
        # Define sample reactions for testing
        self.sample_reactions = [
            {"reactants": "N.C=O", "products": "C=N.[O]"},
            {"reactants": "CO", "products": "C=O.[H].[H]"},
            {"reactants": "CBr.CBr", "products": "CC.BrBr"},
        ]
        self.ban_atoms = [
            "[O].[O]",
            "F-F",
            "Cl-Cl",
            "Br-Br",
            "I-I",
            "Cl-Br",
            "Cl-I",
            "Br-I",
        ]
        self.rule_constraint = RuleConstraint(self.sample_reactions, self.ban_atoms)

    def test_oxygen_products(self):
        # Expected outcome for the first reaction
        expected_outcome = {
            "reactants": "N.C=O.[H].[H]",
            "products": "C=N.O",
            "new_reaction": "N.C=O.[H].[H]>>C=N.O",
        }

        # Applying the fit method
        filtered_reactions, _ = self.rule_constraint.fit()

        # Assertions
        self.assertEqual(
            filtered_reactions[0]["new_reaction"], expected_outcome["new_reaction"]
        )

    def test_hydrogen_products(self):
        # Expected outcome for the second reaction
        expected_outcome = {
            "reactants": "CO.[O]",
            "products": "C=O.O",
            "new_reaction": "CO.[O]>>C=O.O",
        }

        # Applying the fit method
        filtered_reactions, _ = self.rule_constraint.fit()

        # Assertions
        self.assertEqual(
            filtered_reactions[1]["new_reaction"], expected_outcome["new_reaction"]
        )

    def test_reaction_banned_atoms(self):
        # Applying the fit method
        filtered_reactions, _ = self.rule_constraint.fit()

        # Assertions
        with self.assertRaises(IndexError):
            _ = filtered_reactions[2]


if __name__ == "__main__":
    unittest.main()
