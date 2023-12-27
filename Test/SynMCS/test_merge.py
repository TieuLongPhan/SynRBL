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


class DummyCompoundRule:
    def __init__(self, can_apply=True, smiles="O"):
        self.__can_apply = can_apply
        self.__smiles = smiles

    def can_apply(self, b):
        return self.__can_apply

    def apply(self):
        comp = structure.Compound(self.__smiles)
        comp.add_boundary(0)
        return comp


class TestMergeBoundary(unittest.TestCase):
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
        self.assertEqual(1, len(cm.rules))  # type: ignore
        self.assertEqual(b12, cm.boundaries[0])  # type: ignore
        cm = merge.merge_boundaries(b12, b3)
        self.assertEqual("O=C(O)c1ccccc1C(=O)O", cm.smiles)  # type: ignore
        self.assertEqual(0, len(cm.boundaries))  # type: ignore
        self.assertEqual(2, len(cm.rules))  # type: ignore


class TestMergeRule(unittest.TestCase):
    def test_phosphor_double_bond(self):
        c1 = structure.Compound("O")
        c2 = structure.Compound("c1ccc(P(c2ccccc2)c2ccccc2)cc1")
        b1 = c1.add_boundary(0, "O")
        b2 = c2.add_boundary(4, "P")
        cm = merge.merge_boundaries(b1, b2)
        self.assertEqual("O=P(c1ccccc1)(c1ccccc1)c1ccccc1", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.rules))  # type: ignore
        self.assertEqual("phosphor double bond", cm.rules[0].name)  # type: ignore

    def test_merge_restriction(self):
        c1 = structure.Compound("Cl")
        c2 = structure.Compound("CCCC[Sn](CCCC)CCCC")
        b1 = c1.add_boundary(0)
        b2 = c2.add_boundary(4, "Sn")
        cm = merge.merge_boundaries(b1, b2)
        self.assertEqual("CCCC[Sn](CCCC)CCCC.Cl", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.rules))  # type: ignore
        self.assertEqual("bond restriction", cm.rules[0].name)  # type: ignore

    def test_default_single_bond(self):
        c1 = structure.Compound("CC1(C)OBOC1(C)C")
        b1 = c1.add_boundary(4, "B")
        c2 = structure.Compound("CCC")
        b2 = c2.add_boundary(1, "C")
        cm = merge.merge_boundaries(b1, b2)
        self.assertEqual("CC(C)B1OC(C)(C)C(C)(C)O1", cm.smiles)  # type: ignore
        self.assertEqual("default single bond", cm.rules[0].name)  # type: ignore


class TestExpansion(unittest.TestCase):
    @mock.patch("SynRBL.SynMCS.merge.CompoundRule")
    def test_simple_expansion(self, m_CompoundRule):
        m_CompoundRule.get_all = mock.MagicMock(return_value=[DummyCompoundRule()])
        c = structure.Compound("C")
        b = c.add_boundary(0)
        cm = merge.expand_boundary(b)
        self.assertEqual("O", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.boundaries))  # type: ignore

    @mock.patch("SynRBL.SynMCS.merge.CompoundRule")
    def test_rule_apply_check(self, m_CompoundRule):
        m_CompoundRule.get_all = mock.MagicMock(
            return_value=[
                DummyCompoundRule(can_apply=False),
                DummyCompoundRule(smiles="C"),
            ]
        )
        c = structure.Compound("C")
        b = c.add_boundary(0)
        cm = merge.expand_boundary(b)
        self.assertEqual("C", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.boundaries))  # type: ignore


class TestExpandRule(unittest.TestCase):
    def test_expand_O_next_to_O_or_N(self):
        c = structure.Compound("O=COCc1ccccc1")
        b = c.add_boundary(1, neighbor_symbol="O")
        cm = merge.expand_boundary(b)
        self.assertEqual("O", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.boundaries))  # type: ignore
        self.assertEqual(1, len(cm.rules))  # type: ignore
        self.assertEqual("append O when next to O or N", cm.rules[0].name)  # type: ignore

    def test_expand_O_to_CC_bond(self):
        c = structure.Compound("C")
        b = c.add_boundary(0, neighbor_symbol="C")
        cm = merge.expand_boundary(b)  # type: ignore
        self.assertEqual("O", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.boundaries))  # type: ignore
        self.assertEqual(1, len(cm.rules))  # type: ignore
        self.assertEqual("append O to C-C bond", cm.rules[0].name)  # type: ignore


class TestMerge(unittest.TestCase):
    def test_merge_1comp_a_1bound(self):
        c1 = structure.Compound("C[Si](C)C(C)(C)C")
        c1.add_boundary(1, neighbor_symbol="O")
        cm = merge.merge(c1)
        self.assertEqual("C[Si](O)(C)C(C)(C)C", cm.smiles)

    def test_merge_1comp_a_2bounds(self):
        c1 = structure.Compound("O=Cc1ccccc1C=O")
        c1.add_boundary(1, "C", neighbor_symbol="N")
        c1.add_boundary(8, "C", neighbor_symbol="N")
        cm = merge.merge(c1)
        self.assertEqual("O=C(O)c1ccccc1C(O)=O", cm.smiles)

    def test_merge_2comp_a_1bound(self):
        c1 = structure.Compound("CC1(C)OBOC1(C)C")
        c1.add_boundary(4, "B", neighbor_symbol="C")
        c2 = structure.Compound("Br")
        c2.add_boundary(0, "Br", neighbor_symbol="C")
        cm = merge.merge([c1, c2])
        self.assertEqual("CC1(C)OB(Br)OC1(C)C", cm.smiles)

    def test_merge_2comp_a_2bounds(self):
        c1 = structure.Compound("O=Cc1ccccc1C=O")
        c1.add_boundary(1, "C")
        c1.add_boundary(8, "C")
        c2 = structure.Compound("O.O")
        c2.add_boundary(0)
        c2.add_boundary(1)
        cm = merge.merge([c1, c2])
        self.assertEqual("O=C(O)c1ccccc1C(=O)O", cm.smiles)
