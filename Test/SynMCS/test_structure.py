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


