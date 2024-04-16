import unittest

from synrbl.SynMCSImputer.SubStructure.extract_common_mcs import ExtractMCS


class TestExtractMCS(unittest.TestCase):
    def setUp(self):
        self.extractor = ExtractMCS()

    def test_get_num_atoms(self):
        # Test calculation of the number of atoms in a molecule
        smiles = "CCO"
        num_atoms = self.extractor.get_num_atoms(smiles)
        self.assertEqual(num_atoms, 3)

    def test_calculate_total_number_atoms_mcs_parallel(self):
        # Test parallel calculation of total number of atoms in MCS results
        condition = [{"mcs_results": ["CCO", "CC"]}]
        total_atoms = self.extractor.calculate_total_number_atoms_mcs_parallel(
            condition
        )
        self.assertEqual(total_atoms, [5])

    def test_get_popular_elements_from_list(self):
        # Test finding the most popular elements
        elements_list = ["H", "C", "C", "O"]
        popular_elements = self.extractor.get_popular_elements_from_list(elements_list)
        self.assertIn("C", popular_elements)

    def test_get_top_n_common_elements(self):
        # Test finding the top n most common elements
        elements_list = [{"H", "C"}, {"C", "O"}]
        top_elements = self.extractor.get_top_n_common_elements(elements_list, top_n=2)
        self.assertEqual(len(top_elements), 2)

    def test_calculate_corrected_individual_overlap_percentage(self):
        # Test calculating corrected individual overlap percentage
        conditions = [[{"mcs_results": ["CCO"]}], [{"mcs_results": ["CC"]}]]
        (
            overlap_percentages,
            reference_results_list,
        ) = self.extractor.calculate_corrected_individual_overlap_percentage(
            *conditions
        )
        self.assertIsNotNone(overlap_percentages)

    def test_extract_common_mcs_index(self):
        # Test extracting common MCS index
        conditions = [[{"mcs_results": ["CCO"]}], [{"mcs_results": ["CC"]}]]
        (
            threshold_index,
            reference_results_list,
        ) = self.extractor.extract_common_mcs_index(0, 100, *conditions)
        self.assertIsNotNone(threshold_index)


if __name__ == "__main__":
    unittest.main()
