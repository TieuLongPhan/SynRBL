import unittest
import unittest.mock as mock
import rdkit.Chem.rdchem as rdchem

import SynRBL.SynMCS.merge as merge
import SynRBL.SynMCS.structure as structure


class DummyMergeRule:
    def can_apply(self, b1, b2):
        return True

    def apply(self, mol, atom1, atom2):
        mol.AddBond(atom1.GetIdx(), atom2.GetIdx(), order=rdchem.BondType.SINGLE)
        return mol


class TestMerge(unittest.TestCase):
    @mock.patch("SynRBL.SynMCS.merge.MergeRule")
    def test_simple_merge(self, m_MergeRule):
        m_MergeRule.get_all = mock.MagicMock(return_value=[DummyMergeRule()])
        c1 = structure.Compound("CC1(C)OBOC1(C)C")
        b1 = c1.add_boundary(4, "B")
        c2 = structure.Compound("CCC")
        b2 = c2.add_boundary(1, "C")
        cm = merge.merge_boundaries(b1, b2)
        self.assertIsNot(None, cm)
        self.assertEqual("CC(C)B1OC(C)(C)C(C)(C)O1", cm.smiles)  # type: ignore
        self.assertEqual(0, len(cm.boundaries))  # type: ignore

    @mock.patch("SynRBL.SynMCS.merge.MergeRule")
    def test_no_rule_found(self, m_MergeRule):
        rule = mock.MagicMock()
        rule.can_apply.return_value = False
        m_MergeRule.get_all = mock.MagicMock(return_value=[rule])
        c1 = structure.Compound("C")
        b1 = c1.add_boundary(0)
        c2 = structure.Compound("O")
        b2 = c2.add_boundary(0)
        cm = merge.merge_boundaries(b1, b2)
        self.assertIs(None, cm)

    @mock.patch("SynRBL.SynMCS.merge.MergeRule")
    def test_merge_with_unequal_number_of_bounds(self, m_MergeRule):
        m_MergeRule.get_all = mock.MagicMock(return_value=[DummyMergeRule()])
        c1 = structure.Compound("O=Cc1ccccc1C=O")
        b11 = c1.add_boundary(1, "C")
        b12 = c1.add_boundary(8, "C")
        c2 = structure.Compound("O")
        b2 = c2.add_boundary(0)
        c3 = structure.Compound("O")
        b3 = c3.add_boundary(0)
        cm = merge.merge_boundaries(b11, b2)
        self.assertIsNot(None, cm)
        self.assertEqual("O=Cc1ccccc1C(=O)O", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.boundaries))  # type: ignore
        self.assertEqual(b12, cm.boundaries[0])  # type: ignore
        cm = merge.merge_boundaries(b12, b3)
        self.assertEqual("O=C(O)c1ccccc1C(=O)O", cm.smiles)  # type: ignore
        self.assertEqual(0, len(cm.boundaries))  # type: ignore


class TestMergeRule(unittest.TestCase):
    def test_phosphor_double_bond(self):
        c1 = structure.Compound("O")
        c2 = structure.Compound("c1ccc(P(c2ccccc2)c2ccccc2)cc1")
        b1 = c1.add_boundary(0, "O")
        b2 = c2.add_boundary(4, "P")
        cm = merge.merge_boundaries(b1, b2)
        self.assertIsNot(None, cm)
        self.assertEqual("c1ccc(P(=O)(c2ccccc2)c2ccccc2)cc1", cm.smiles)  # type: ignore


class TestCompoundCollection(unittest.TestCase):
    def test(self):
        pass
