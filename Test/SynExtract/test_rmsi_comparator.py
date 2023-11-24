import unittest
from SynRBL.SynExtract import RSMIComparator  

class TestRSMIComparator(unittest.TestCase):
    def test_compare_dicts_balance(self):
        result = RSMIComparator.compare_dicts({'C': 1, 'H': 4}, {'C': 1, 'H': 4})
        self.assertEqual(result, "Balance")

    def test_compare_dicts_unbalance(self):
        result = RSMIComparator.compare_dicts({'C': 1, 'H': 4}, {'C': 2, 'H': 4})
        self.assertEqual(result, "Products")

    def test_diff_dicts(self):
        result = RSMIComparator.diff_dicts({'C': 1, 'H': 4}, {'C': 1, 'H': 6})
        self.assertEqual(result, {'H': 2})

    def test_run_parallel(self):
        comparator = RSMIComparator([{'C': 1, 'H': 4}], [{'C': 1, 'H': 6}], parallel=False)
        comparison, diff = comparator.run_parallel()
        self.assertEqual(comparison, ["Products"])
        self.assertEqual(diff, [{'H': 2}])

if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
