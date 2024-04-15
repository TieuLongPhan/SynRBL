import unittest
from synrbl.SynChemImputer.molecule_standardizer import (
    MoleculeStandardizer,
)


class TestMoleculeStandardizer(unittest.TestCase):

    def test_enol_transformation(self):
        smiles = "C=C(O)CC"  # But-1-en-2-ol
        atom_indices = [0, 1, 2]  # Indices of C=O and adjacent C
        standardizer = MoleculeStandardizer(smiles)
        result = standardizer.standardize_enol(smiles, atom_indices)
        expected = "CCC(C)=O"  # Expected butanone
        self.assertEqual(result, expected, "Enol transformation failed or incorrect")

    def test_hemiketal_transformation(self):
        # Assuming the structure can form a hemiketal
        smiles = "C(O)(O)"
        atom_indices = [0, 1, 2]  # Indices for hemiketal formation
        standardizer = MoleculeStandardizer(smiles)
        result = standardizer.standardize_hemiketal(smiles, atom_indices)
        expected = "C=O.O"  # Expected hemiketal cyclic structure
        self.assertEqual(
            result, expected, "Hemiketal transformation failed or incorrect"
        )

    def test_MoleculeStandardizer(self):
        smiles = "C(O)(O)C=CO"
        standardizer = MoleculeStandardizer(smiles)
        result = standardizer.fit()
        expected = "O.O=CCC=O"
        self.assertEqual(result, expected, "MoleculeStandardizer failed or incorrect")


# If the script is executed directly, run the tests
if __name__ == "__main__":
    unittest.main()
