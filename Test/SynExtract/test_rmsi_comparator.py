import sys
from pathlib import Path
import unittest
import pandas as pd
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
from SynRBL.SynExtract import RSMIComparator  

class TestRSMIComparator(unittest.TestCase):

    def test_check_keys(self):
        # Test check_keys method
        dict1 = {'C': 2, 'H': 6}
        dict2 = {'C': 1}
        self.assertTrue(RSMIComparator.check_keys(dict1, dict2))

        dict3 = {'O': 2}
        self.assertFalse(RSMIComparator.check_keys(dict1, dict3))

    def test_compare_dicts_balanced(self):
        # Test compare_dicts method for a balanced reaction
        reactant = {'C': 2, 'H': 6}
        product = {'C': 2, 'H': 6}
        result = RSMIComparator.compare_dicts(reactant, product)
        self.assertEqual(result, 'Balance')

    def test_compare_dicts_unbalanced(self):
        # Test compare_dicts method for an unbalanced reaction
        reactant = {'C': 2, 'H': 4}
        product = {'C': 2, 'H': 6}
        result = RSMIComparator.compare_dicts(reactant, product)
        self.assertEqual(result, 'Reactants')

    def test_diff_dicts(self):
        # Test diff_dicts method
        reactant = {'C': 2, 'H': 6}
        product = {'C': 2, 'H': 4}
        result = RSMIComparator.diff_dicts(reactant, product)
        self.assertEqual(result, {'H': 2})

    def test_run_parallel(self):
        # Test run_parallel method
        reactants = [{'C': 1, 'H': 4}, {'O': 2}]
        products = [{'C': 1, 'H': 6}, {'O': 1, 'H': 2}]
        comparator = RSMIComparator(reactants, products, n_jobs=2, verbose=0)
        comparison_results, difference_results = comparator.run_parallel(reactants, products)
        self.assertEqual(comparison_results, ['Reactants', 'Both'])
        self.assertEqual(difference_results, [{'H': 2}, {'O': 1}])

if __name__ == '__main__':
    unittest.main()

