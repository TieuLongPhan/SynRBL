import unittest
from SynRBL.SynExtract import RSMIProcessing, can_parse  # Replace 'your_module' with the actual module name
import pandas as pd

class TestRSMIProcessing(unittest.TestCase):
    def test_can_parse_valid(self):
        self.assertTrue(can_parse('CC>>CC'))

    def test_can_parse_invalid(self):
        self.assertFalse(can_parse('CC>>', symbol='>>'))

    def test_smi_splitter_valid(self):
        processor = RSMIProcessing(rsmi='CC>>CC')
        self.assertEqual(processor.smi_splitter(), ('CC', 'CC'))

    def test_smi_splitter_invalid(self):
        processor = RSMIProcessing(rsmi='CC>>')
        self.assertEqual(processor.smi_splitter(), "Can't parse")

    def test_data_splitter(self):
        data = pd.DataFrame({'rsmi': ['C>>CC', 'CC>>CCC']})
        processor = RSMIProcessing(data=data, rsmi_col='rsmi', parallel=False)
        processed_data = processor.data_splitter()
        expected_data = pd.DataFrame({'rsmi': ['C>>CC', 'CC>>CCC'], 'reactants': ['C', 'CC'], 'products': ['CC', 'CCC']})
        pd.testing.assert_frame_equal(processed_data, expected_data)

if __name__ == '__main__':
    unittest.main()