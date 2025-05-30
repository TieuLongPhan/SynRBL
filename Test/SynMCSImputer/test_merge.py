import unittest
import unittest.mock as mock

import synrbl.SynMCSImputer.merge as merge
from synrbl.SynMCSImputer.structure import Compound, CompoundSet


class DummyMergeRule:
    def can_apply(self, b1, b2):
        return True

    def apply(self, boundary1, boundary2):
        return None


class DummyExpandRule:
    def __init__(self, can_apply=True, smiles="O"):
        self.__can_apply = can_apply
        self.__smiles = smiles

    def can_apply(self, b):
        return self.__can_apply

    def apply(self):
        comp = Compound(self.__smiles)
        comp.add_boundary(0)
        return comp


class TestMergeBoundary(unittest.TestCase):
    @mock.patch("synrbl.SynMCSImputer.merge.MergeRule")
    def test_no_rule_found(self, m_MergeRule):
        rule = mock.MagicMock()
        rule.can_apply.return_value = False
        m_MergeRule.get_all = mock.MagicMock(return_value=[rule])
        c1 = Compound("C")
        b1 = c1.add_boundary(0)
        c2 = Compound("O")
        b2 = c2.add_boundary(0)
        cm = merge.merge_boundaries(b1, b2)
        self.assertIs(None, cm)


class TestMergeRule(unittest.TestCase):
    def test_default_single_bond(self):
        c1 = Compound("CC1(C)OBOC1(C)C")
        b1 = c1.add_boundary(4, "B")
        c2 = Compound("CCC")
        b2 = c2.add_boundary(1, "C")
        cm = merge.merge_boundaries(b1, b2)
        self.assertEqual("CC(C)B1OC(C)(C)C(C)(C)O1", cm.smiles)  # type: ignore
        self.assertEqual("default single bond", cm.rules[0].name)  # type: ignore

    def test_phosphor_bond1(self):
        # If Oxygen comes from COH merge with single bond
        c1 = Compound("O", src_mol="CCO")
        c2 = Compound("BrPBr", src_mol="BrP(Br)Br")
        b1 = c1.add_boundary(0, "O", 1, "C")
        b2 = c2.add_boundary(1, "P", 2, "Br")
        cm = merge.merge_boundaries(b1, b2)
        self.assertEqual(1, len(cm.rules))  # type: ignore
        self.assertEqual("phosphor single bond", cm.rules[0].name)  # type: ignore
        self.assertEqual("OP(Br)Br", cm.smiles)  # type: ignore

    def test_phosphor_bond2(self):
        # If Oxygen comes from C=O and P has no P=O merge with double bond
        c1 = Compound("O", src_mol="CC(=O)")
        c2 = Compound("BrPBr", src_mol="BrP(Br)Br")
        b1 = c1.add_boundary(0, "O", 1, "C")
        b2 = c2.add_boundary(1, "P", 2, "Br")
        cm = merge.merge_boundaries(b1, b2)
        self.assertEqual(1, len(cm.rules))  # type: ignore
        self.assertEqual("phosphor double bond", cm.rules[0].name)  # type: ignore
        self.assertEqual("O=[PH](Br)Br", cm.smiles)  # type: ignore

    def test_phosphor_bond3(self):
        # If Oxygen comes from C=O and P has P=O merge with double bond
        # and change old P=O double bond to single bond
        c1 = Compound("[O:3]", src_mol="[CH3:0][C:1]([CH3:2])=[O:3]")
        c2 = Compound(
            "[P:5](=[O:8])([OH:6])[OH:7]", src_mol="[CH3:4][P:5](=[O:8])([OH:6])[OH:7]"
        )
        b1 = c1.add_boundary(0, "O", 1, "C")
        b2 = c2.add_boundary(0, "P", 0, "C")
        cm = merge.merge_boundaries(b1, b2)
        self.assertEqual(1, len(cm.rules))  # type: ignore
        self.assertEqual("phosphor double bond change", cm.rules[0].name)  # type: ignore
        self.assertEqual("[O:3]=[P:5]([OH:6])([OH:7])[O:8]", cm.smiles)  # type: ignore

    def test_nitrogen_bond1(self):
        c1 = Compound("C", src_mol="C=C")
        c2 = Compound("N#N", src_mol="CS(=O)(=O)N=[N+]=[N-]")
        b1 = c1.add_boundary(0, "C", 1, "C")
        b2 = c2.add_boundary(0, "N", 5, "N")
        cm = merge.merge_boundaries(b1, b2)
        self.assertEqual(1, len(cm.rules))  # type: ignore
        self.assertEqual("nitrogen double bond", cm.rules[0].name)  # type: ignore
        self.assertEqual("C=[N+]=[N-]", cm.smiles)  # type: ignore


class TestExpansion(unittest.TestCase):
    @mock.patch("synrbl.SynMCSImputer.merge.ExpandRule")
    def test_simple_expansion(self, m_ExpandRule):
        m_ExpandRule.get_all = mock.MagicMock(return_value=[DummyExpandRule()])
        c = Compound("C")
        b = c.add_boundary(0)
        cm = merge.expand_boundary(b)
        self.assertEqual("O", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.boundaries))  # type: ignore

    @mock.patch("synrbl.SynMCSImputer.merge.ExpandRule")
    def test_rule_apply_check(self, m_ExpandRule):
        m_ExpandRule.get_all = mock.MagicMock(
            return_value=[
                DummyExpandRule(can_apply=False),
                DummyExpandRule(smiles="C"),
            ]
        )
        c = Compound("C")
        b = c.add_boundary(0)
        cm = merge.expand_boundary(b)
        self.assertEqual("C", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.boundaries))  # type: ignore


class TestExpandRule(unittest.TestCase):
    def test_expand_O_next_to_O_or_N(self):
        c = Compound("O=COCc1ccccc1", src_mol="O=C(NCCOc1ccc(-c2cnoc2)cc1)OCc1ccccc1")
        b = c.add_boundary(1, neighbor_index=2, neighbor_symbol="N")
        cm = merge.expand_boundary(b)
        self.assertEqual("O", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.boundaries))  # type: ignore
        self.assertEqual(1, len(cm.rules))  # type: ignore

    def test_expand_O_to_CC_bond(self):
        c = Compound("C", "CC")
        b = c.add_boundary(0, neighbor_index=1)
        cm = merge.expand_boundary(b)  # type: ignore
        self.assertEqual("O", cm.smiles)  # type: ignore
        self.assertEqual(1, len(cm.boundaries))  # type: ignore
        self.assertEqual(1, len(cm.rules))  # type: ignore


class TestCompounds(unittest.TestCase):
    def test_merge_with_unequal_number_of_bounds(self):
        c1 = Compound("O=Cc1ccccc1C=O")
        b11 = c1.add_boundary(1, "C")
        b12 = c1.add_boundary(8, "C")
        c2 = Compound("O")
        b2 = c2.add_boundary(0)
        c3 = Compound("O")
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

    def test_1(self):
        # broken bond: C (boundary) - O (neighbor)
        # O is part of Ether -> C forms C - I
        cset = CompoundSet()
        compound = cset.add_compound("C", src_mol="COc1ccccc1")
        compound.add_boundary(0, neighbor_index=1, neighbor_symbol="O")
        merged = merge.merge(cset)
        self.assertEqual("CI", merged.smiles)

    def test_2(self):
        # broken bond: C (boundary) - S (neighbor)
        # S is part of Thioether -> C forms C - I
        cset = CompoundSet()
        compound = cset.add_compound("C", src_mol="CSc1ccccc1")
        compound.add_boundary(0, neighbor_index=1, neighbor_symbol="S")
        merged = merge.merge(cset)
        self.assertEqual("CI", merged.smiles)

    def test_3(self):
        # broken bond: C (boundary) - O (neighbor)
        # O is NOT part of Ether -> C forms C - O
        cset = CompoundSet()
        compound = cset.add_compound(
            "CC(C)(C)", src_mol="OC(=O)CONC(=O)NCc1cccc2ccccc12"
        )
        compound.add_boundary(1, neighbor_index=4, neighbor_symbol="O")
        merged = merge.merge(cset)
        self.assertEqual("CC(C)(C)O", merged.smiles)

    def test_4(self):
        # broken bond: C (boundary) - O (neighbor)
        # O is NOT part of Ether -> C forms C - O
        cset = CompoundSet()
        compound = cset.add_compound(
            "CC(C)(C)", src_mol="OC(=O)CONC(=O)NCc1cccc2ccccc12"
        )
        compound.add_boundary(1, neighbor_index=4, neighbor_symbol="O")
        merged = merge.merge(cset)
        self.assertEqual("CC(C)(C)O", merged.smiles)

    def test_thioether_break(self):
        # broken bond: C (boundary) - S (neighbor)
        # S is part of Thioether -> C forms C - I
        # Reaction: "CCSC.[H]I>>CCSH.CI"
        cset = CompoundSet()
        compound = cset.add_compound("C", src_mol="CCSC")
        compound.add_boundary(0, symbol="C", neighbor_index=2, neighbor_symbol="S")
        merged = merge.merge(cset)
        self.assertIn("C-S Thioether break", [r.name for r in merged.rules])
        self.assertEqual("CI", merged.smiles)

    def test_thioester_break(self):
        # broken bond: C (boundary) - S (neighbor)
        # S is part of Thioester -> C forms C - O
        cset = CompoundSet()
        compound = cset.add_compound(
            "CC=O", src_mol="CC(=O)SCC(C)C(=O)N(CC(=O)O)C1CCC1"
        )
        compound.add_boundary(1, symbol="C", neighbor_index=3, neighbor_symbol="S")
        merged = merge.merge(cset)
        self.assertIn("C-S Thioester break", [r.name for r in merged.rules])
        self.assertEqual("CC(=O)O", merged.smiles)

    def test_leave_single_compound_as_is(self):
        s = "CS(C)=O"
        cset = CompoundSet()
        compound = cset.add_compound(s, src_mol="C[SH](C)(C)=O")
        compound.add_boundary(1, neighbor_index=0)
        cm = merge.merge(cset)
        self.assertEqual(s, cm.smiles)
        self.assertEqual(0, len(cm.boundaries))

    def test_merge_with_charge(self):
        cset = CompoundSet()
        compound1 = cset.add_compound("CNOC", src_mol="CON(C)C(=O)C1CCN(Cc2ccccc2)CC1")
        compound1.add_boundary(1, symbol="N", neighbor_index=4, neighbor_symbol="C")
        compound2 = cset.add_compound("[MgH+]", src_mol="C[Mg+]")
        compound2.add_boundary(0, symbol="Mg", neighbor_index=0, neighbor_symbol="C")
        cm = merge.merge(cset)
        self.assertEqual("CO[N](C)[Mg+]", cm.smiles)

    def test_merge_with_explicit_H_1(self):
        cset = CompoundSet()
        compound = cset.add_compound(
            "C[SH](=O)=O", src_mol="CS(=O)(=O)Oc1ccc(C(=N)N)cc1C(=O)c1ccccc1"
        )
        compound.add_boundary(1, symbol="S", neighbor_index=4, neighbor_symbol="O")
        cm = merge.merge(cset)
        self.assertEqual("CS(=O)(=O)O", cm.smiles)

    def test_merge_P_with_explicit_H(self):
        cset = CompoundSet()
        compound1 = cset.add_compound(
            "CCO[PH](=O)OCC", src_mol="CCOP(=O)(Cc1cccc(C#N)c1)OCC"
        )
        compound1.add_boundary(3, symbol="P", neighbor_index=5, neighbor_symbol="C")
        compound2 = cset.add_compound("O", src_mol="CC(C)=O")
        compound2.add_boundary(0, symbol="O", neighbor_index=1, neighbor_symbol="C")
        cm = merge.merge(cset)
        self.assertEqual("CCOP(=O)(O)OCC", cm.smiles)

    def test_O_forms_alcohol(self):
        cset = CompoundSet()
        compound1 = cset.add_compound("C", src_mol="CC(=O)OC")
        compound1.add_boundary(0, symbol="C")
        compound2 = cset.add_compound("O", src_mol="O")
        compound2.add_boundary(0, symbol="O")
        cm = merge.merge(cset)
        self.assertEqual("CO", cm.smiles)

    def test_5(self):
        # super complicated reaction: OCC(O)CC(O)O.O=CCCC=O>>OC1CC2C=C(CC2O1)C=O
        cset = CompoundSet()
        compound1 = cset.add_compound("CCCCO", src_mol="O=CC1=CC2CC(O)OC2C1")
        compound1.add_boundary(0, symbol="C", neighbor_index=2, neighbor_symbol="C")
        compound1.add_boundary(1, symbol="C", neighbor_index=9, neighbor_symbol="C")
        compound1.add_boundary(3, symbol="C", neighbor_index=8, neighbor_symbol="O")
        cm = merge.merge(cset)
        self.assertEqual("OCC(O)CC(O)O", cm.smiles)

    def test_merge_expansion_of_two_compounds_with_unequal_nr_of_bonds(self):
        cset = CompoundSet()
        compound1 = cset.add_compound(
            "Cl", src_mol="COc1ccc(N(C)c2nc(CCl)nc3ccccc23)cc1Cl"
        )
        compound1.add_boundary(0, symbol="Cl", neighbor_index=11, neighbor_symbol="C")
        compound2 = cset.add_compound("O=Cc1ccccc1C=O", src_mol="O=C1NC(=O)c2ccccc21")
        compound2.add_boundary(1, symbol="C", neighbor_index=2, neighbor_symbol="N")
        compound2.add_boundary(8, symbol="C", neighbor_index=2, neighbor_symbol="N")
        compound1.rules = ["r1"]
        compound2.rules = ["r2"]
        cm = merge.merge(cset)
        self.assertEqual("Cl.O=C(O)c1ccccc1C(=O)O", cm.smiles)
        self.assertIn("r1", cm.rules)
        self.assertIn("r2", cm.rules)

    def test_6(self):
        cset = CompoundSet()
        compound1 = cset.add_compound("Br", src_mol="Clc1ccccc1CBr")
        compound2 = cset.add_compound("N", src_mol="N#CC1CC1")
        compound1.add_boundary(0, symbol="Br", neighbor_index=7, neighbor_symbol="C")
        compound2.add_boundary(0, symbol="N", neighbor_index=1, neighbor_symbol="C")
        cm = merge.merge(cset)
        self.assertEqual("Br.N", cm.smiles)

    def test_ignore_water_in_passthrough(self):
        cset = CompoundSet()
        compound1 = cset.add_compound("C", src_mol="C")
        compound1.add_boundary(0, symbol="C")
        compound2 = cset.add_compound("O", src_mol="O")
        compound2.add_boundary(0, symbol="O")
        cset.add_compound("O", src_mol="O")
        cm = merge.merge(cset)
        self.assertEqual("remove_water_catalyst", cm.rules[0].name)
        self.assertEqual("CO", cm.smiles)

    def test_catalyst_passthrough(self):
        cset = CompoundSet()
        cset.add_compound("C", src_mol="C")
        cm = merge.merge(cset)
        self.assertEqual(0, len(cm.rules))
        self.assertEqual("C", cm.smiles)
