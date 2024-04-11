import unittest

from synrbl.SynMCSImputer.SubStructure.substructure_analyzer import SubstructureAnalyzer
from rdkit import Chem


class TestSubstructureAnalyzer(unittest.TestCase):
    def setUp(self):
        self.analyzer = SubstructureAnalyzer()
        self.reactant = Chem.MolFromSmiles("CC(=O)N(C)C1=CC=CC=C1")
        self.product = Chem.MolFromSmiles("CNC1=CC=CC=C1")

    def test_remove_substructure_atoms(self):
        substructures = self.reactant.GetSubstructMatches(self.product)
        # Test case for removing a substructure
        fragment_count = self.analyzer.remove_substructure_atoms(
            self.reactant, substructures[0]
        )
        self.assertEqual(fragment_count, 3)  # break wrong

        fragment_count = self.analyzer.remove_substructure_atoms(
            self.reactant, substructures[1]
        )
        self.assertEqual(fragment_count, 1)  # break right

    def test_sort_substructures_by_fragment_count(self):
        # Test case for sorting substructures
        substructures = self.reactant.GetSubstructMatches(self.product)
        fragment_counts = [3, 1]  # Corresponding fragment counts
        sorted_substructures = self.analyzer.sort_substructures_by_fragment_count(
            substructures, fragment_counts
        )
        print(sorted_substructures)
        self.assertEqual(
            sorted_substructures, [(4, 3, 5, 6, 7, 8, 9, 10), (1, 3, 5, 6, 7, 8, 9, 10)]
        )  # Should be sorted by fragment count

    def test_number_substructures(self):
        substructures = self.reactant.GetSubstructMatches(self.product)
        self.assertEqual(len(substructures), 2)

    def test_identify_optimal_substructure(self):
        # Test case for identifying the optimal substructure

        optimal_substructure = self.analyzer.identify_optimal_substructure(
            self.reactant, self.product
        )
        print(optimal_substructure)
        self.assertEqual(optimal_substructure, (4, 3, 5, 6, 7, 8, 9, 10))
        self.assertNotEqual(
            optimal_substructure, (1, 3, 5, 6, 7, 8, 9, 10)
        )  # Assuming this is the correct substructure


if __name__ == "__main__":
    unittest.main()
