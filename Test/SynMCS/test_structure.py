import unittest
import unittest.mock as mock
import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.rdmolfiles as rdmolfiles
from SynRBL.SynMCS.structure import *


class TestCompound(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            Compound(None)
        Compound("C")
        Compound(rdmolfiles.MolFromSmiles("C"))

    def test_resolve_merge(self):
        c = Compound("CCO")
        b = c.add_boundary(1)
        new_smiles = "CC(O)O"
        new_mol = rdmolfiles.MolFromSmiles(new_smiles)
        c.update(new_mol, b)
        self.assertEqual(0, len(c.boundaries))
        self.assertEqual(new_smiles, c.smiles)


    def test_add_boundary(self):
        c = Compound("CCO")
        c.add_boundary(0, "C")
        with self.assertRaises(ValueError):
            c.add_boundary(2, "C")
        c.add_boundary(2)

    def test_boundary_len(self):
        c = Compound("CCO")
        self.assertEqual(0, len(c.boundaries))
        c.add_boundary(0, "C")
        self.assertEqual(1, len(c.boundaries))

    def test_boundary_get(self):
        c = Compound("CCO")
        b = c.add_boundary(1, "C")
        self.assertEqual(1, b.index)
        self.assertEqual('C', b.symbol)


