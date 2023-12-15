import unittest
import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.rdmolfiles as rdmolfiles
from SynRBL.SynMCS.structure import *


class TestReaction(unittest.TestCase):
    def test_init(self):
        smiles = "C1CO1>>OCCO"
        r = Reaction(smiles)
        self.assertEqual(smiles, r.smiles)


class TestReactionCompound(unittest.TestCase):
    def test_init(self):
        c = ReactionCompound("CCO", "O", False, True)
        self.assertEqual("O", c.reactant.GetAtomWithIdx(2).GetSymbol())
        self.assertEqual("O", c.product.GetAtomWithIdx(0).GetSymbol())
        self.assertFalse(c.is_new_reactant)
        self.assertTrue(c.is_new_product)

    def test_add_broken_bond(self):
        c = ReactionCompound("CCO", "O", False, True)
        c.add_broken_bond({"O": 0}, {"C": 1})
        self.assertEqual(1, len(c.boundaries))
        self.assertEqual("O", c.boundaries[0][0].GetSymbol())
        self.assertEqual("C", c.boundaries[0][1].GetSymbol())

    def test_add_broken_bond_without_neighbor(self):
        c = ReactionCompound("O", "O", False, True)
        c.add_broken_bond({"O": 0})
        self.assertEqual(1, len(c.boundaries))
        self.assertEqual("O", c.boundaries[0][0].GetSymbol())
        self.assertEqual(None, c.boundaries[0][1])

    def test_complete_reaction(self):
        r = rdmolfiles.MolFromSmiles("C1CO1")
        p = rdmolfiles.MolFromSmiles("OCCO")
        c = ReactionCompound("O", "O", True, False)
        r, p = c.complete_reaction(r, p)
        self.assertEqual("C1CO1.O", rdmolfiles.MolToSmiles(r))
        self.assertEqual("OCCO", rdmolfiles.MolToSmiles(p))
