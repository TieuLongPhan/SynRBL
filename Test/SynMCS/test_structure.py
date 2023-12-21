import unittest
import unittest.mock as mock
import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.rdmolfiles as rdmolfiles
from SynRBL.SynMCS.structure import *


class TestCompound(unittest.TestCase):
    def test_add_boundary(self):
        c = Compound("CCO")
        c.add_boundary(0, "C")
        with self.assertRaises(ValueError):
            c.add_boundary(2, "C")
        c.add_boundary(2)

    def test_boundary_len(self):
        c = Compound("CCO")
        self.assertEqual(0, c.boundary_len)
        c.add_boundary(0, "C")
        self.assertEqual(1, c.boundary_len)

    def test_boundary_get(self):
        c = Compound("CCO")
        c.add_boundary(1, "C")
        b = c.get_boundary(0)
        self.assertEqual(1, b.index)
        self.assertEqual('C', b.symbol)


class TestCompoundCollection(unittest.TestCase):
    def test_merge_two_single_bounds(self):
        cc = CompoundCollection()
        c1 = Compound("CC1(C)OBOC1(C)C")
        c1.add_boundary(4, "B")
        c2 = Compound("CCC")
        c2.add_boundary(1, "C")
        cc.compounds.extend([c1, c2])
        cc.merge()
        self.assertTrue(c1.get_boundary(0).is_merged)
        self.assertTrue(c2.get_boundary(0).is_merged)

    def test_merge_one_compound(self):
        cc = CompoundCollection()
        c1 = Compound("CC1(C)OBOC1(C)C")
        c1.add_boundary(4, "B")
        cc.compounds.append(c1)
        with self.assertRaises(ValueError):
            cc.merge()

    def test_merge_one_single_bound(self):
        cc = CompoundCollection()
        c1 = Compound("CC1(C)OBOC1(C)C")
        c1.add_boundary(4, "B")
        c2 = Compound("CCC")
        cc.compounds.extend([c1, c2])
        with self.assertRaises(ValueError):
            cc.merge()

    # TODO probably remove 
    def test_gml_without_compounds(self):
        cc = CompoundCollection()
        ml = cc.get_merge_list()
        self.assertEqual(0, len(ml))

    def test_gml_with_one_compound_without_bounds(self):
        cc = CompoundCollection()
        c = Compound("O")
        c.boundaries = []
        cc.compounds.append(c)
        ml = cc.get_merge_list()
        self.assertEqual(0, len(ml))

    def test_gml_with_one_compound_with_one_bound(self):
        cc = CompoundCollection()
        c1 = mock.MagicMock()
        b1 = mock.MagicMock()
        c1.boundaries = [b1]
        cc.compounds.append(c1)
        with self.assertRaises(ValueError):
            cc.get_merge_list()

    def test_gml_with_two_compounds_with_one_bound(self):
        cc = CompoundCollection()
        c1 = mock.MagicMock()
        c2 = mock.MagicMock()
        b1 = mock.MagicMock()
        b2 = mock.MagicMock()
        c1.boundaries = [b1]
        c2.boundaries = [b2]
        cc.compounds.extend([c1, c2])

        ml = cc.get_merge_list()
        self.assertListEqual([(0, 0, 1, 0)], ml)

    def test(self):
        exp_result = "CC1(C)OB(-C(C)C)OC1(C)C"
        c1 = Compound("CC1(C)OBOC1(C)C")
        c1.add_boundary(4, "B")
        c2 = Compound("CCC")
        c2.add_boundary(1, "C")
        # TODO
