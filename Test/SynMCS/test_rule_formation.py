import unittest
import rdkit.Chem.rdmolfiles as rdmolfiles
from SynRBL.SynMCS.rule_formation import *

class TestProperty(unittest.TestCase):
    def __test_prop(self, config, in_p=[], in_n=[], dtype: type[str | int] = str):
        prop = Property(config, dtype=dtype)
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
        self.__test_prop(
            ["!-1", "!1", "-2", "2"], in_p=[-2, 2], in_n=[-1, 1], dtype=int
        )
        self.__test_prop(["1", "!2"], in_p=[1], in_n=[2], dtype=int)

    def test_parsing_none(self):
        prop = Property(None)
        self.assertEqual(0, len(prop.neg_values))
        self.assertEqual(0, len(prop.pos_values))

    def test_passthrough_behaviour(self):
        prop = Property(None)
        self.assertTrue(prop.check("A"))

    def test_allow_none(self):
        prop = Property(allow_none=True)
        self.assertTrue(prop.check(None))

    def test_dont_allow_non(self):
        prop = Property(allow_none=False)
        with self.assertRaises(ValueError):
            prop.check(None)

    def test_check_value_error(self):
        prop = Property("1", dtype=int)
        with self.assertRaises(ValueError):
            prop.check("2")

    def test_check_property(self):
        prop = Property("A")
        self.assertTrue(prop.check("A"))
        self.assertFalse(prop.check("B"))
        prop = Property("!A")
        self.assertTrue(prop.check("B"))
        self.assertFalse(prop.check("A"))
        prop = Property(["A", "B"])
        self.assertTrue(prop.check("A"))
        self.assertTrue(prop.check("B"))
        self.assertFalse(prop.check("C"))
        prop = Property(["!A", "!B"])
        self.assertFalse(prop.check("A"))
        self.assertFalse(prop.check("B"))
        self.assertTrue(prop.check("C"))

    def test_invalid_config_type(self):
        with self.assertRaises(ValueError):
            Property(8) # type: ignore

class TestAtomCondition(unittest.TestCase):
    def __check_cond(self, cond, smiles, idx, expected_result, neighbor=None):
        mol = rdmolfiles.MolFromSmiles(smiles)
        actual_result = cond.check(mol.GetAtomWithIdx(idx), neighbor)
        self.assertEqual(expected_result, actual_result)

    def test_positive_check(self):
        cond = AtomCondition(atom=["C", "O"])
        self.__check_cond(cond, "CO", 0, True)
        self.__check_cond(cond, "CO", 1, True)
        self.__check_cond(cond, "[Na+].[Cl-]", 0, False)

    def test_negative_check(self):
        cond = AtomCondition(atom=["!Si", "!Cl"])
        self.__check_cond(cond, "C=[Si](C)C", 0, True)
        self.__check_cond(cond, "CO", 1, True)
        self.__check_cond(cond, "[Na+]", 0, True)
        self.__check_cond(cond, "C=[Si](C)C", 1, False)
        self.__check_cond(cond, "[Na+].[Cl-]", 1, False)

    def test_positive_check_with_neighbors(self):
        cond = AtomCondition(atom="C", neighbors=["O", "N"])
        self.__check_cond(cond, "CO", 0, True, neighbor="O")

    def test_invalid_neighbor(self):
        cond = AtomCondition(atom="C", neighbors=["O"])
        with self.assertRaises(ValueError):
            self.__check_cond(cond, "CO", 0, True, neighbor=["O"])


class TestAction(unittest.TestCase):
    def test_remove_H_action(self):
        action = ActionSet()
