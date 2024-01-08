import unittest
import unittest.mock as mock
import rdkit.Chem.rdchem as rdchem
import rdkit.Chem.rdmolfiles as rdmolfiles

import SynRBL.SynMCSImputer.merge as merge
import SynRBL.SynMCSImputer.structure as structure


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
    @mock.patch("SynRBL.SynMCSImputer.merge.MergeRule")
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

    @mock.patch("SynRBL.SynMCSImputer.merge.MergeRule")
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

    @mock.patch("SynRBL.SynMCSImputer.merge.MergeRule")
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

    def test_default_single_bond(self):
        c1 = structure.Compound("CC1(C)OBOC1(C)C")
        b1 = c1.add_boundary(4, "B")
        c2 = structure.Compound("CCC")
        b2 = c2.add_boundary(1, "C")
        cm = merge.merge_boundaries(b1, b2)
        self.assertEqual("CC(C)B1OC(C)(C)C(C)(C)O1", cm.smiles)  # type: ignore
        self.assertEqual("default single bond", cm.rules[0].name)  # type: ignore


class TestExpansion(unittest.TestCase):
    @mock.patch("SynRBL.SynMCSImputer.merge.CompoundRule")
    def test_simple_expansion(self, m_CompoundRule):
        m_CompoundRule.get_all = mock.MagicMock(return_value=[DummyCompoundRule()])
        c = structure.Compound("C")
        b = c.add_boundary(0)
        cm = merge.expand_boundary(b)
        self.assertEqual("O", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.boundaries))  # type: ignore

    @mock.patch("SynRBL.SynMCSImputer.merge.CompoundRule")
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
        c = structure.Compound(
            "O=COCc1ccccc1", src_mol="O=C(NCCOc1ccc(-c2cnoc2)cc1)OCc1ccccc1"
        )
        b = c.add_boundary(1, neighbor_index=2, neighbor_symbol="N")
        cm = merge.expand_boundary(b)
        self.assertEqual("O", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.boundaries))  # type: ignore
        self.assertEqual(1, len(cm.rules))  # type: ignore

    def test_expand_O_to_CC_bond(self):
        c = structure.Compound("C", "CC")
        b = c.add_boundary(0, neighbor_index=1)
        cm = merge.expand_boundary(b)  # type: ignore
        self.assertEqual("O", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.boundaries))  # type: ignore
        self.assertEqual(1, len(cm.rules))  # type: ignore


class TestCompounds(unittest.TestCase):
    def test_1(self):
        # broken bond: C (boundary) - O (neighbor)
        # O is part of Ether -> C forms C - I
        compound = structure.Compound("C", src_mol="COc1ccccc1")
        compound.add_boundary(0, neighbor_index=1, neighbor_symbol="O")
        merged = merge.merge(compound)
        self.assertEqual("CI", merged.smiles)

    def test_2(self):
        # broken bond: C (boundary) - S (neighbor)
        # S is part of Thioether -> C forms C - I
        compound = structure.Compound("C", src_mol="CSc1ccccc1")
        compound.add_boundary(0, neighbor_index=1, neighbor_symbol="S")
        merged = merge.merge(compound)
        self.assertEqual("CI", merged.smiles)

    def test_3(self):
        # broken bond: C (boundary) - O (neighbor)
        # O is NOT part of Ether -> C forms C - O
        compound = structure.Compound(
            "CC(C)(C)", src_mol="OC(=O)CONC(=O)NCc1cccc2ccccc12"
        )
        compound.add_boundary(1, neighbor_index=4, neighbor_symbol="O")
        merged = merge.merge(compound)
        self.assertEqual("CC(C)(C)O", merged.smiles)

    def test_4(self):
        # broken bond: C (boundary) - O (neighbor)
        # O is NOT part of Ether -> C forms C - O
        compound = structure.Compound(
            "CC(C)(C)", src_mol="OC(=O)CONC(=O)NCc1cccc2ccccc12"
        )
        compound.add_boundary(1, neighbor_index=4, neighbor_symbol="O")
        merged = merge.merge(compound)
        self.assertEqual("CC(C)(C)O", merged.smiles)

    def test_thioether_break(self):
        # broken bond: C (boundary) - S (neighbor)
        # S is part of Thioether -> C forms C - I
        # Reaction: "CCSC.[H]I>>CCSH.CI"
        compound = structure.Compound("C", src_mol="CCSC")
        compound.add_boundary(0, symbol="C", neighbor_index=2, neighbor_symbol="S")
        merged = merge.merge(compound)
        self.assertIn("C-S Thioether break", [r.name for r in merged.rules])
        self.assertEqual("CI", merged.smiles)

    def test_thioester_break(self):
        # broken bond: C (boundary) - S (neighbor)
        # S is part of Thioester -> C forms C - O
        compound = structure.Compound(
            "CC=O", src_mol="CC(=O)SCC(C)C(=O)N(CC(=O)O)C1CCC1"
        )
        compound.add_boundary(1, symbol="C", neighbor_index=3, neighbor_symbol="S")
        merged = merge.merge(compound)
        self.assertIn("C-S Thioester break", [r.name for r in merged.rules])
        self.assertEqual("CC(=O)O", merged.smiles)

    def test_leave_single_compound_as_is(self):
        s = "CS(C)=O"
        compound = structure.Compound(s, src_mol="C[SH](C)(C)=O")
        compound.add_boundary(1, neighbor_index=0)
        cm = merge.merge(compound)
        self.assertEqual(s, cm.smiles)
        self.assertEqual(0, len(cm.boundaries))

    def test_merge_with_charge(self):
        compound1 = structure.Compound("CNOC", src_mol="CON(C)C(=O)C1CCN(Cc2ccccc2)CC1")
        compound1.add_boundary(1, symbol="N", neighbor_index=4, neighbor_symbol="C")
        compound2 = structure.Compound("[MgH+]", src_mol="C[Mg+]")
        compound2.add_boundary(0, symbol="Mg", neighbor_index=0, neighbor_symbol="C")
        cm = merge.merge([compound1, compound2])
        self.assertEqual("CON(C)[Mg+]", cm.smiles)

    def test_merge_with_explicit_H_1(self):
        compound = structure.Compound(
            "C[SH](=O)=O", src_mol="CS(=O)(=O)Oc1ccc(C(=N)N)cc1C(=O)c1ccccc1"
        )
        compound.add_boundary(1, symbol="S", neighbor_index=4, neighbor_symbol="O")
        cm = merge.merge(compound)
        self.assertEqual("CS(=O)(=O)O", cm.smiles)

    def test_merge_P_with_explicit_H(self):
        compound1 = structure.Compound(
            "CCO[PH](=O)OCC", src_mol="CCOP(=O)(Cc1cccc(C#N)c1)OCC"
        )
        compound1.add_boundary(3, symbol="P", neighbor_index=5, neighbor_symbol="C")
        compound2 = structure.Compound("O", src_mol="CC(C)=O")
        compound2.add_boundary(0, symbol="O", neighbor_index=1, neighbor_symbol="C")
        cm = merge.merge([compound1, compound2])
        self.assertEqual("CCO[P](=O)(=O)OCC", cm.smiles)

    def test_O_forms_alcohol(self):
        compound1 = structure.Compound("C", src_mol="CC(=O)OC")
        compound1.add_boundary(0, symbol="C")
        compound2 = structure.Compound("O", src_mol="O")
        compound2.add_boundary(0, symbol="O")
        cm = merge.merge([compound1, compound2])
        self.assertEqual("CO", cm.smiles)

    def test_5(self):
        # super complicated reaction: OCC(O)CC(O)O.O=CCCC=O>>OC1CC2C=C(CC2O1)C=O
        compound1 = structure.Compound("CCCCO", src_mol="O=CC1=CC2CC(O)OC2C1")
        compound1.add_boundary(0, symbol="C", neighbor_index=2, neighbor_symbol="C")
        compound1.add_boundary(1, symbol="C", neighbor_index=9, neighbor_symbol="C")
        compound1.add_boundary(3, symbol="C", neighbor_index=8, neighbor_symbol="O")
        cm = merge.merge(compound1)
        self.assertEqual("OCC(O)CC(O)O", cm.smiles)
