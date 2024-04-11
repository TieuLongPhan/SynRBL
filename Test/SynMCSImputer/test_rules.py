import unittest

from synrbl.SynMCSImputer.structure import Compound
from synrbl.SynMCSImputer.rules import (
    Action,
    Property,
    FunctionalGroupProperty,
    FunctionalGroupCompoundProperty,
    BoundarySymbolProperty,
    NeighborSymbolProperty,
    BoundaryCondition,
    PatternProperty,
    ChangeBondAction,
    CountBoundariesCompoundProperty,
    AddBoundaryAction,
    SmilesCompoundProperty,
)


class TestProperty(unittest.TestCase):
    def __test_prop(self, config, in_p=[], in_n=[]):
        prop = Property(config)
        self.assertTrue(all(e in prop.pos_values for e in in_p))
        self.assertTrue(all(e in prop.neg_values for e in in_n))
        self.assertTrue(all(e not in prop.pos_values for e in in_n))
        self.assertTrue(all(e not in prop.neg_values for e in in_p))

    def test_parsing(self):
        self.__test_prop("A", in_p=["A"])
        self.__test_prop(["A", "B"], in_p=["A", "B"])
        self.__test_prop("!A", in_n=["A"])
        self.__test_prop(["!A", "!B"], in_n=["A", "B"])
        self.__test_prop(["!A", "B"], in_p=["B"], in_n=["A"])

    def test_parsing_none(self):
        prop = Property(None)
        self.assertEqual(0, len(prop.neg_values))
        self.assertEqual(0, len(prop.pos_values))

    def test_passthrough_behaviour(self):
        prop = Property(None)
        self.assertTrue(prop("A"))

    def test_allow_none(self):
        prop = Property(allow_none=True)
        self.assertTrue(prop(None))

    def test_dont_allow_non(self):
        prop = Property(allow_none=False)
        with self.assertRaises(ValueError):
            prop(None)

    def test_check_property(self):
        prop = Property("A")
        self.assertTrue(prop("A"))
        self.assertFalse(prop("B"))
        prop = Property("!A")
        self.assertTrue(prop("B"))
        self.assertFalse(prop("A"))
        prop = Property(["A", "B"])
        self.assertTrue(prop("A"))
        self.assertTrue(prop("B"))
        self.assertFalse(prop("C"))
        prop = Property(["!A", "!B"])
        self.assertFalse(prop("A"))
        self.assertFalse(prop("B"))
        self.assertTrue(prop("C"))

    def test_invalid_config_type(self):
        with self.assertRaises(ValueError):
            Property({"test": 0})  # type: ignore


class TestFunctionalGroupProperty(unittest.TestCase):
    def test_init(self):
        p = FunctionalGroupProperty(["ether", "!ester"])
        self.assertIn("ether", p.pos_values)
        self.assertIn("ester", p.neg_values)

    def test_check_with_incomplete_boundary(self):
        p = FunctionalGroupProperty(["ester"])
        c = Compound("CCC")
        b = c.add_boundary(0)
        self.assertFalse(p(b))

    def test_pos_successful_check(self):
        p = FunctionalGroupProperty(["ester"])
        c = Compound("CCC", src_mol="CC(=O)OCCC")
        b = c.add_boundary(0, neighbor_index=3)
        result = p(b)
        self.assertEqual(True, result)

    def test_pos_fail_check(self):
        p = FunctionalGroupProperty(["amin"])
        c = Compound("CCC", src_mol="CC(=O)OCCC")
        b = c.add_boundary(0, neighbor_index=3)
        result = p(b)
        self.assertEqual(False, result)

    def test_neg_successful_check(self):
        p = FunctionalGroupProperty(["!ester"])
        c = Compound("CCC", src_mol="CC(=O)OCCC")
        b = c.add_boundary(0, neighbor_index=3)
        result = p(b)
        self.assertEqual(False, result)

    def test_neg_fail_check(self):
        p = FunctionalGroupProperty(["!amin"])
        c = Compound("CCC", src_mol="CC(=O)OCCC")
        b = c.add_boundary(0, neighbor_index=3)
        result = p(b)
        self.assertEqual(True, result)

    def test_defaults_to_true(self):
        p = FunctionalGroupProperty()
        c = Compound("C")
        b = c.add_boundary(0)
        self.assertTrue(p(b))

    def test_check_with_missing_src(self):
        p = FunctionalGroupProperty(["ester"])
        c = Compound("CC(=O)OCCC")
        b = c.add_boundary(3)
        result = p(b)
        self.assertFalse(result)


class TestBoundarySymbolProperty(unittest.TestCase):
    def test_defaults_to_true(self):
        p = BoundarySymbolProperty()
        c = Compound("C")
        b = c.add_boundary(0)
        self.assertTrue(p(b))

    def test_match_atom(self):
        p = BoundarySymbolProperty("C")
        c = Compound("C")
        b = c.add_boundary(0)
        self.assertTrue(p(b))

    def test_antipattern_match_atom(self):
        p = BoundarySymbolProperty("!C")
        c = Compound("C")
        b = c.add_boundary(0)
        self.assertFalse(p(b))


class TestNeighborSymbolProperty(unittest.TestCase):
    def test_defaults_to_true(self):
        p = NeighborSymbolProperty()
        c = Compound("O", "CCO")
        b = c.add_boundary(0, neighbor_index=1)
        self.assertTrue(p(b))

    def test_match_atom(self):
        p = NeighborSymbolProperty("C")
        c = Compound("O", "CCO")
        b = c.add_boundary(0, neighbor_index=1)
        self.assertTrue(p(b))

    def test_antipattern_match_atom(self):
        p = NeighborSymbolProperty("!C")
        c = Compound("O", "CCO")
        b = c.add_boundary(0, neighbor_index=1)
        self.assertFalse(p(b))


class TestBoundaryCondition(unittest.TestCase):
    def __check_cond(
        self, cond, smiles, idx, expected_result, neighbor=None, src_smiles=None
    ):
        c = Compound(smiles, src_mol=src_smiles)
        b = c.add_boundary(idx, neighbor_index=neighbor)
        actual_result = cond(b)
        self.assertEqual(expected_result, actual_result)

    def test_positive_check(self):
        cond = BoundaryCondition(atom=["C", "O"])
        self.__check_cond(cond, "CO", 0, True)
        self.__check_cond(cond, "CO", 1, True)
        self.__check_cond(cond, "[Na+].[Cl-]", 0, False)

    def test_negative_check(self):
        cond = BoundaryCondition(atom=["!Si", "!Cl"])
        self.__check_cond(cond, "C=[Si](C)C", 0, True)
        self.__check_cond(cond, "CO", 1, True)
        self.__check_cond(cond, "[Na+]", 0, True)
        self.__check_cond(cond, "C=[Si](C)C", 1, False)
        self.__check_cond(cond, "[Na+].[Cl-]", 1, False)

    def test_positive_check_with_neighbors(self):
        cond = BoundaryCondition(atom="C", neighbor_atom=["O", "N"])
        self.__check_cond(cond, "C", 0, True, src_smiles="CO", neighbor=1)

    def test_pattern_match(self):
        cond = BoundaryCondition(src_pattern="P=O")
        self.__check_cond(cond, "Cl", 0, True, src_smiles="O=P(Cl)Cl", neighbor=1)

    def test_unseccessful_pattern_match(self):
        cond = BoundaryCondition(src_pattern="P=O")
        self.__check_cond(cond, "Cl", 0, False, src_smiles="O=C(Cl)Cl", neighbor=1)


class TestPatternProperty(unittest.TestCase):
    def test_successful_src_match(self):
        cond = PatternProperty("P=O", use_src_mol=True)
        comp = Compound("Cl", src_mol="O=P(Cl)Cl")
        boundary = comp.add_boundary(0, neighbor_index=1, neighbor_symbol="P")
        result = cond(boundary)
        self.assertTrue(result)

    def test_successful_match(self):
        cond = PatternProperty("P=O", use_src_mol=False)
        comp = Compound("O=P(Cl)Cl")
        boundary = comp.add_boundary(0)
        result = cond(boundary)
        self.assertTrue(result)

    def test_unsuccessful_match(self):
        cond = PatternProperty("P=O", use_src_mol=True)
        comp = Compound("Cl", src_mol="O=C(Cl)Cl")
        boundary = comp.add_boundary(0, neighbor_index=0, neighbor_symbol="O")
        result = cond(boundary)
        self.assertFalse(result)

    def test_successful_match_with_anchor(self):
        cond = PatternProperty("P=O", use_src_mol=True)
        comp = Compound("Cl", src_mol="O=P(Cl)(Cl)Cl")
        boundary = comp.add_boundary(0, neighbor_index=1, neighbor_symbol="P")
        result = cond(boundary)
        self.assertTrue(result)

    def test_uninitialized(self):
        cond = PatternProperty(use_src_mol=True)
        comp = Compound("Cl", src_mol="O=P(Cl)(Cl)Cl")
        boundary = comp.add_boundary(0, neighbor_index=1, neighbor_symbol="P")
        result = cond(boundary)
        self.assertTrue(result)

    def test_antipattern_match(self):
        cond = PatternProperty("!P=O", use_src_mol=True)
        comp = Compound("BrPBr", src_mol="BrP(Br)Br")
        boundary = comp.add_boundary(1, "P", neighbor_index=2, neighbor_symbol="Br")
        result = cond(boundary)
        self.assertTrue(result)


class TestChangeBondAction(unittest.TestCase):
    def test_invalid_init(self):
        with self.assertRaises(ValueError):
            ChangeBondAction("CCO", "single")
        with self.assertRaises(ValueError):
            ChangeBondAction("C.C", "single")
        with self.assertRaises(ValueError):
            ChangeBondAction(pattern="O=P")
        with self.assertRaises(ValueError):
            ChangeBondAction(bond="single")

    def test_build(self):
        action = Action.build("change_bond", **{"pattern": "CC", "bond": "single"})
        self.assertTrue(isinstance(action, ChangeBondAction))

    def test_change_double_to_single_bond(self):
        action = ChangeBondAction("O=P", "single")
        comp = Compound("O=P(O)O", src_mol="O=P(C)(O)O")
        boundary = comp.add_boundary(1, "P")
        action(boundary)
        self.assertEqual("OP(O)O", boundary.compound.smiles)


class TestNrBoundariesCompoundProperty(unittest.TestCase):
    def test_pos_check(self):
        prop = CountBoundariesCompoundProperty("0")
        comp = Compound("O=P(O)O", src_mol="O=P(C)(O)O")
        self.assertTrue(prop(comp))

    def test_neg_check(self):
        prop = CountBoundariesCompoundProperty("1")
        comp = Compound("O=P(O)O", src_mol="O=P(C)(O)O")
        self.assertFalse(prop(comp))


class TestFunctionalGroupCompoundProperty(unittest.TestCase):
    def test_single_fg_pos_check(self):
        prop = FunctionalGroupCompoundProperty("alcohol")
        comp = Compound("CO")
        self.assertTrue(prop(comp))

    def test_single_fg_neg_check(self):
        prop = FunctionalGroupCompoundProperty("ether")
        comp = Compound("CO")
        self.assertFalse(prop(comp))

    def test_multiple_fgs(self):
        prop1 = FunctionalGroupCompoundProperty("alcohol")
        prop2 = FunctionalGroupCompoundProperty("ether")
        comp = Compound("COCCO")
        self.assertTrue(prop1(comp))
        self.assertTrue(prop2(comp))


class TestSmilesCompoundProperty(unittest.TestCase):
    def test_atom_mapped_smiles(self):
        prop = SmilesCompoundProperty("O")
        comp = Compound("[OH2:1]")
        self.assertTrue(prop(comp))


class TestAddBoundaryAction(unittest.TestCase):
    def test_1(self):
        action = AddBoundaryAction("alcohol", "CO", "1")
        comp = Compound("COCCO")
        action(comp)
        self.assertEqual(1, len(comp.boundaries))  # type: ignore
        self.assertEqual(4, comp.boundaries[0].index)  # type: ignore
        self.assertEqual("O", comp.boundaries[0].symbol)  # type: ignorte
