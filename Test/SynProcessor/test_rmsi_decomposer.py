import unittest
import pandas as pd

from synrbl.SynProcessor.rsmi_decomposer import RSMIDecomposer


class TestRSMIDecomposer(unittest.TestCase):
    def setUp(self):
        self.decomposer = RSMIDecomposer()

    def test_decompose_valid(self):
        # Test decompose method with a valid SMILES string
        smiles = "CCO"
        composition = RSMIDecomposer.decompose(smiles)
        self.assertEqual(sorted(composition), sorted({"C": 2, "O": 1, "H": 6}))

    def test_decompose_invalid(self):
        # Test decompose method with an invalid SMILES string
        smiles = "InvalidString"
        composition = RSMIDecomposer.decompose(smiles)
        self.assertEqual({}, composition)

    def test_data_decomposer_valid(self):
        # Test data_decomposer method with valid data
        data = pd.DataFrame({"reactants": ["CCO", "CC"], "products": ["C=O", "C=C"]})
        decomposer = RSMIDecomposer(data=data, parallel=False)
        reactants, products = decomposer.data_decomposer()
        self.assertEqual(len(reactants), 2)
        self.assertEqual(len(products), 2)

    def test_data_decomposer_invalid(self):
        # Test data_decomposer method with invalid data
        data = pd.DataFrame(
            {"reactants": ["InvalidString", "CC"], "products": ["C=O", "InvalidString"]}
        )
        decomposer = RSMIDecomposer(data=data, parallel=False)
        reactants, products = decomposer.data_decomposer()
        self.assertEqual(len(reactants), 2)
        self.assertEqual(len(products), 2)
        self.assertEqual({}, reactants[0])
        self.assertEqual({}, products[1])


if __name__ == "__main__":
    unittest.main()
