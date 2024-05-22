import unittest
from synrbl.SynChemImputer.molecule_standardizer import (
    MoleculeStandardizer,
)


class TestMoleculeStandardizer(unittest.TestCase):

    def test_enol_transformation(self):
        smiles = "C=C(O)CC"  # But-1-en-2-ol
        atom_indices = [0, 1, 2]  # Indices of C=O and adjacent C
        result = MoleculeStandardizer.standardize_enol(smiles, atom_indices)
        expected = "CCC(C)=O"  # Expected butanone
        self.assertEqual(result, expected, "Enol transformation failed or incorrect")

    def test_hemiketal_transformation(self):
        # Assuming the structure can form a hemiketal
        smiles = "C(O)(O)"
        atom_indices = [0, 1, 2]  # Indices for hemiketal formation
        result = MoleculeStandardizer.standardize_hemiketal(smiles, atom_indices)
        expected = "C=O.O"  # Expected hemiketal cyclic structure
        self.assertEqual(
            result, expected, "Hemiketal transformation failed or incorrect"
        )

    def test_hemiketal_transformation_with_aam(self):
        smiles = "[C:1]([O:2])([OH:3])"
        atom_indices = [0, 1, 2]
        result = MoleculeStandardizer.standardize_hemiketal(smiles, atom_indices)
        expected = ["[C:1]=[O:2].[OH2:3]", "[C:1]=[O:3].[OH2:2]"]
        self.assertIn(result, expected)

    def test_MoleculeStandardizer(self):
        smiles = "C(O)(O)C=CO"
        standardizer = MoleculeStandardizer()
        result = standardizer(smiles)
        expected = "O.O=CCC=O"
        self.assertEqual(result, expected, "MoleculeStandardizer failed or incorrect")


# If the script is executed directly, run the tests
if __name__ == "__main__":
    unittest.main()
