import unittest
import rdkit.Chem.rdmolfiles as rdmolfiles

from synrbl.SynMCSImputer.utils import check_atom_dict, InvalidAtomDict


class TestCheckAtoms(unittest.TestCase):
    def test_valid_atom_dict(self):
        mol = rdmolfiles.MolFromSmiles("CCO")
        check_atom_dict(mol, {"O": 2})
        check_atom_dict(mol, [{"C": 0}, {"O": 2}])

    def test_invalid_atom_dict(self):
        mol = rdmolfiles.MolFromSmiles("CCO")
        with self.assertRaises(InvalidAtomDict):
            check_atom_dict(mol, {"N": 2})
