import unittest
import unittest.mock as mock
import rdkit.Chem.rdchem as rdchem

from .merge import *
from .structure import Compound

class DummyMergeRule:
    def apply(self, mol, atom1, atom2):
        mol.AddBond(atom1.GetIdx(), atom2.GetIdx(), order=rdchem.BondType.SINGLE)
        return mol

class TestMerge(unittest.TestCase):
    def test_merge_with_unequal_number_of_bounds(self):
        rule = DummyMergeRule()
        c1 = Compound("O=Cc1ccccc1C=O")
        b11 = c1.add_boundary(1, "C")
        c1.add_boundary(8, "C")
        c2 = Compound("O")
        b2 = c2.add_boundary(0)
        c3 = Compound("O")
        b3 = c3.add_boundary(0)
        cm = merge(b11, b2, rule) # type: ignore
        self.assertIsNot(None, cm)
        self.assertEqual("O=Cc1ccccc1C(=O)O", cm.smiles) # type: ignore
        self.assertEqual(1, len(cm.boundaries)) # type: ignore
        cm = merge(cm.boundaries[0], b3, rule)
        self.assertEqual("O=C(O)c1ccccc1C(O)=O", cm.smiles)
        self.assertEqual(0, len(cm.boundaries)) 



class TestCompoundCollection(unittest.TestCase):
    def test_merge_two_single_bounds(self):
        c1 = Compound("CC1(C)OBOC1(C)C")
        c1.add_boundary(4, "B")
        c2 = Compound("CCC")
        c2.add_boundary(1, "C")
        merge([c1, c2])
        self.assertTrue(c1.boundaries[0].is_merged)
        self.assertTrue(c2.boundaries[0].is_merged)

    def test_merge_one_compound(self):
        c1 = Compound("CC1(C)OBOC1(C)C")
        c1.add_boundary(4, "B")
        with self.assertRaises(ValueError):
            merge([c1])

    def test_merge_one_single_bound(self):
        c1 = Compound("CC1(C)OBOC1(C)C")
        c1.add_boundary(4, "B")
        c2 = Compound("CCC")
        with self.assertRaises(ValueError):
            merge([c1, c2])
