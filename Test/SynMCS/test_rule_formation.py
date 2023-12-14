import unittest
from SynRBL.SynMCS.rule_formation import *


class TestProperty(unittest.TestCase):
    def __test_prop(self, config, in_p=[], in_n=[], dtype=str):
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
        prop = Property(["!-1", "!1", "-2", "2"], dtype=int)
        self.assertTrue(all(v in prop.neg_values for v in [-1, 1]))
        self.assertTrue(all(v in prop.pos_values for v in [-2, 2]))
        prop = Property([1, "!2"], dtype=int)
        self.assertIn(1, prop.pos_values)
        self.assertIn(2, prop.neg_values)

    def test_parsing_none(self):
        prop = Property(None)
        self.assertEqual(0, len(prop.neg_values))
        self.assertEqual(0, len(prop.pos_values))

    def test_passthrough_behaviour(self):
        prop = Property(None)
        self.assertTrue(prop.check("A"))

    def test_allow_none(self):
        prop = Property("A", allow_none=True)
        self.assertTrue(prop.check(None))

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
