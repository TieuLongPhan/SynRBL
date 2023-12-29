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

    def test_init_src_mol(self):
        c = Compound("C")
        self.assertEqual(None, c.src_smiles)
        c = Compound("C", src_mol="CCO")
        self.assertEqual("CCO", c.src_smiles)
        c = Compound("C", src_mol=rdmolfiles.MolFromSmiles("CCO"))
        self.assertEqual("CCO", c.src_smiles)


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
        self.assertEqual("C", b.symbol)

    def test_get_boundary_atom(self):
        mol = rdmolfiles.MolFromSmiles("CCO")
        exp_atom = mol.GetAtomWithIdx(2)
        c = Compound(mol)
        b = c.add_boundary(2, "O")
        act_atom = b.get_atom()
        self.assertEqual(exp_atom.GetSymbol(), act_atom.GetSymbol())
        self.assertEqual(exp_atom.GetIdx(), act_atom.GetIdx())

    def test_get_boundary_symbol(self):
        c = Compound("CCO")
        b = c.add_boundary(2)
        self.assertEqual("O", b.symbol)

    def test_add_boundary_with_neighbor(self):
        c = Compound("CCC", src_mol="CC(=O)OCCC")
        b = c.add_boundary(0, neighbor_index=3)
        self.assertEqual("C", b.get_atom().GetSymbol())
        self.assertEqual("O", b.get_neighbor_atom().GetSymbol()) # type: ignore

    def test_add_boundary_with_missing_src(self):
        c = Compound("CCC")
        with self.assertRaises(ValueError):
            c.add_boundary(0, neighbor_index=3)

    def test_add_boundary_with_invalid_neighbor(self):
        c = Compound("CCC", src_mol="CC(=O)OCCC")
        with self.assertRaises(ValueError):
            c.add_boundary(0, "C", 3, "C")

