import unittest

from synrbl.SynRuleImputer import SyntheticRuleMatcher


class TestSyntheticRuleMatcher(unittest.TestCase):
    def setUp(self):
        # Sample rules and data
        self.rule_dict = [
            {"smiles": "C1=CC=CC=C1", "Composition": {"C": 6, "H": 6}},
            {"smiles": "CCO", "Composition": {"C": 2, "H": 6, "O": 1}},
        ]
        self.data_dict = {"C": 6, "H": 6}
        self.matcher = SyntheticRuleMatcher(
            self.rule_dict, self.data_dict, select="best", ranking=False
        )

    def test_match(self):
        # Test matching solutions
        solution = self.matcher.match()
        self.assertIsInstance(solution, list)
        self.assertGreater(len(solution), 0)

    def test_dfs(self):
        # Test depth-first search algorithm
        solution = self.matcher.dfs(self.data_dict, [])
        self.assertIsInstance(solution, list)
        self.assertGreater(len(solution), 0)

    def test_apply_rule(self):
        # Test applying a chemical rule
        rule = {"smiles": "C1=CC=CC=C1", "Composition": {"C": 6, "H": 6}}
        new_data, new_path = self.matcher.apply_rule(self.data_dict, [], rule)
        self.assertIsNotNone(new_data)
        self.assertIsNotNone(new_path)

    def test_can_match(self):
        # Test checking if a chemical rule can be matched
        rule = {"C": 6, "H": 6}
        can_match = self.matcher.can_match(rule, self.data_dict)
        self.assertTrue(can_match)

    def test_rank_solutions(self):
        # Test ranking solutions
        solutions = [[{"smiles": "C1=CC=CC=C1", "Ratio": 1}]]
        ranked_solutions = SyntheticRuleMatcher.rank_solutions(
            solutions, ranking="longest"
        )
        self.assertEqual(solutions, ranked_solutions)

    def test_remove_overlapping_solutions(self):
        # Test removing overlapping solutions
        solutions = [
            [{"smiles": "C1=CC=CC=C1", "Ratio": 1}],
            [{"smiles": "C1=CC=CC=C1", "Ratio": 1}],
        ]
        unique_solutions = SyntheticRuleMatcher.remove_overlapping_solutions(solutions)
        self.assertEqual(len(unique_solutions), 1)


if __name__ == "__main__":
    unittest.main()
