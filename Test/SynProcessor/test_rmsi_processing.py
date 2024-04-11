import unittest
import pandas as pd

from synrbl.SynProcessor import RSMIProcessing


class TestRSMIProcessing(unittest.TestCase):
    def test_can_parse_valid(self):
        """
        Test can_parse function with a valid RSMI string
        """
        rsmi = "CCO>>CCOC"
        self.assertTrue(RSMIProcessing.can_parse(rsmi))

    def test_can_parse_invalid(self):
        """
        Test can_parse function with a invalid RSMI string
        """
        invalid_rsmi = "InvalidString>>C"
        self.assertFalse(RSMIProcessing.can_parse(invalid_rsmi))

    def test_smi_splitter_valid(self):
        """
        Test smi_splitter method with a valid RSMI string
        """
        # Define the valid RSMI string
        rsmi = "CCO>>CCOC"

        # Initialize the RSMIProcessing object with the RSMI string and symbol
        processor = RSMIProcessing(reaction_smiles=rsmi, symbol=">>")

        # Call the smi_splitter method to split the RSMI string into reactants
        # and products
        reactants, products = processor.smi_splitter(rsmi)

        # Assert that the reactants and products are correct
        self.assertEqual(reactants, "CCO")
        self.assertEqual(products, "CCOC")

    def test_smi_splitter_invalid(self):
        """
        Test smi_splitter method with a valid RSMI string
        """
        rsmi = "InvalidString>>C"
        processor = RSMIProcessing(reaction_smiles=rsmi, symbol=">>")
        result = processor.smi_splitter(rsmi)
        self.assertEqual(result, "Can't parse")

    def test_data_splitter_valid(self):
        """
        Test data_splitter method with a DataFrame containing valid RSMI strings
        """
        valid_data = pd.DataFrame({"rsmi": ["CCO>>CCOC", "CC>>C"]})
        processor = RSMIProcessing(data=valid_data, rsmi_col="rsmi", parallel=False)
        processed_data = processor.data_splitter()
        self.assertIn("reactants", processed_data.columns)
        self.assertIn("products", processed_data.columns)
        self.assertEqual(len(processed_data), 2)

    def test_data_splitter_with_invalid(self):
        """
        Test data_splitter method with a DataFrame containing valid RSMI strings
        """
        mixed_data = pd.DataFrame({"rsmi": ["CCO>>C", "CCO>>InvalidString"]})
        processor = RSMIProcessing(data=mixed_data, rsmi_col="rsmi", parallel=False)
        processed_data = processor.data_splitter()
        self.assertIn("reactants", processed_data.columns)
        self.assertIn("products", processed_data.columns)
        self.assertEqual(
            len(processed_data), 1
        )  # Expecting only 1 valid entry to be processed

    def test_data_splitter_parallel_with_valid_and_invalid(self):
        """
        Test data_splitter method with parallel processing enabled
        """

        # A DataFrame containing both valid and invalid RSMI strings
        mixed_data = pd.DataFrame({"rsmi": ["CCO>>CCOC", "InvalidString>>C", "CC>>C"]})
        processor = RSMIProcessing(
            data=mixed_data, rsmi_col="rsmi", parallel=True, n_jobs=2
        )
        processed_data = processor.data_splitter()

        # Ensure that the processed data contains reactants and products columns
        self.assertIn("reactants", processed_data.columns)
        self.assertIn("products", processed_data.columns)

        # Verify that the processed data only includes the valid RSMI strings
        self.assertEqual(
            len(processed_data), 2
        )  # Expecting 2 valid entries to be processed
        self.assertNotIn("InvalidString", processed_data["rsmi"].values)


if __name__ == "__main__":
    unittest.main()
