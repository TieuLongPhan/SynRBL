import sys
from pathlib import Path
import unittest
from rdkit import Chem
import unittest
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
#from SynRBL.SynCleaning.standardizer_wrapper import normalize_molecule

from SynRBL.SynCleaning.standardizer_wrapper import (  
    normalize_molecule,
    canonicalize_tautomer,
    salts_remover,
    reionize_charges,
    uncharge_molecule,
    assign_stereochemistry,
    fragments_remover,
    remove_hydrogens_and_sanitize,
)

class TestMoleculeFunctions(unittest.TestCase):

    def setUp(self):
        # Example molecule for testing
        self.mol = Chem.MolFromSmiles('CC(=O)O.C[Na]')  # Acetic acid and sodium cation

    def test_normalize_molecule(self):
        normalized_mol = normalize_molecule(self.mol)
        self.assertIsNotNone(normalized_mol)

    def test_canonicalize_tautomer(self):
        try:
            canonical_tautomer = canonicalize_tautomer(self.mol)
            self.assertIsNotNone(canonical_tautomer)
        except:
            self.assertTrue(False)

    def test_salts_remover(self):
        no_salts = salts_remover(self.mol)
        self.assertIsNotNone(no_salts)

    def test_reionize_charges(self):
        reionized = reionize_charges(self.mol)
        self.assertIsNotNone(reionized)

    def test_uncharge_molecule(self):
        uncharged = uncharge_molecule(self.mol)
        self.assertIsNotNone(uncharged)

    def test_assign_stereochemistry(self):
        # This function does not return a molecule, so we test if it runs without errors
        assign_stereochemistry(self.mol)
        self.assertTrue(True)

    def test_fragments_remover(self):
        largest_fragment = fragments_remover(self.mol)
        self.assertIsNotNone(largest_fragment)

    def test_remove_hydrogens_and_sanitize(self):
        no_hydrogens = remove_hydrogens_and_sanitize(self.mol)
        self.assertIsNotNone(no_hydrogens)

if __name__ == '__main__':
    unittest.main()
