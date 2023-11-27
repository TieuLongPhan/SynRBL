import sys
from pathlib import Path
import unittest
import pandas as pd
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))

from SynRBL.SynExtract import RSMIProcessing, can_parse   

class TestRSMIProcessing(unittest.TestCase):

    def test_can_parse_valid(self):
        # Test can_parse function with a valid RSMI string
        rsmi = 'CCO>>CCOC'
        self.assertTrue(can_parse(rsmi))

    def test_can_parse_invalid(self):
        # Test can_parse function with an invalid RSMI string
        rsmi = 'InvalidString'
        self.assertFalse(can_parse(rsmi))

    def test_smi_splitter_valid(self):
        # Test smi_splitter method with a valid RSMI string
        rsmi = 'CCO>>CCOC'
        processor = RSMIProcessing(rsmi=rsmi)
        reactants, products = processor.smi_splitter()
        self.assertEqual(reactants, 'CCO')
        self.assertEqual(products, 'CCOC')

    def test_smi_splitter_invalid(self):
        # Test smi_splitter method with an invalid RSMI string
        rsmi = 'InvalidString'
        processor = RSMIProcessing(rsmi=rsmi)
        result = processor.smi_splitter()
        self.assertEqual(result, "Can't parse")

    def test_data_splitter(self):
        # Test data_splitter method with a DataFrame of RSMI strings
        data = pd.DataFrame({'rsmi': ['CCO>>CCOC', 'CC>>C']})
        processor = RSMIProcessing(data=data, rsmi_col='rsmi', parallel=False)
        processed_data = processor.data_splitter()
        self.assertIn('reactants', processed_data.columns)
        self.assertIn('products', processed_data.columns)
        self.assertEqual(len(processed_data), 2)

    # Additional tests for other methods and edge cases can be added here

if __name__ == '__main__':
    unittest.main()
