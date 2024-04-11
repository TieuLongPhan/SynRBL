import unittest
import rdkit.Chem.rdmolfiles as rdmolfiles
import synrbl.SynMCSImputer.structure as structure


class TestCompound(unittest.TestCase):
    def test_init(self):
        with self.assertRaises(ValueError):
            structure.Compound(None)
        structure.Compound("C")
        structure.Compound(rdmolfiles.MolFromSmiles("C"))

    def test_init_src_mol(self):
        c = structure.Compound("C")
        self.assertEqual(None, c.src_smiles)
        c = structure.Compound("C", src_mol="CCO")
        self.assertEqual("CCO", c.src_smiles)
        c = structure.Compound("C", src_mol=rdmolfiles.MolFromSmiles("CCO"))
        self.assertEqual("CCO", c.src_smiles)

    def test_resolve_merge(self):
        c = structure.Compound("CCO")
        b = c.add_boundary(1)
        new_smiles = "CC(O)O"
        new_mol = rdmolfiles.MolFromSmiles(new_smiles)
        c.update(new_mol, b)
        self.assertEqual(0, len(c.boundaries))
        self.assertEqual(new_smiles, c.smiles)

    def test_add_boundary(self):
        c = structure.Compound("CCO")
        c.add_boundary(0, "C")
        with self.assertRaises(ValueError):
            c.add_boundary(2, "C")
        c.add_boundary(2)

    def test_boundary_len(self):
        c = structure.Compound("CCO")
        self.assertEqual(0, len(c.boundaries))
        c.add_boundary(0, "C")
        self.assertEqual(1, len(c.boundaries))

    def test_boundary_get(self):
        c = structure.Compound("CCO")
        b = c.add_boundary(1, "C")
        self.assertEqual(1, b.index)
        self.assertEqual("C", b.symbol)

    def test_get_boundary_atom(self):
        mol = rdmolfiles.MolFromSmiles("CCO")
        exp_atom = mol.GetAtomWithIdx(2)
        c = structure.Compound(mol)
        b = c.add_boundary(2, "O")
        act_atom = b.get_atom()
        self.assertEqual(exp_atom.GetSymbol(), act_atom.GetSymbol())
        self.assertEqual(exp_atom.GetIdx(), act_atom.GetIdx())

    def test_get_boundary_symbol(self):
        c = structure.Compound("CCO")
        b = c.add_boundary(2)
        self.assertEqual("O", b.symbol)

    def test_add_boundary_with_neighbor(self):
        c = structure.Compound("CCC", src_mol="CC(=O)OCCC")
        b = c.add_boundary(0, neighbor_index=3)
        self.assertEqual("C", b.get_atom().GetSymbol())
        self.assertEqual("O", b.get_neighbor_atom().GetSymbol())  # type: ignore

    def test_add_boundary_with_missing_src(self):
        c = structure.Compound("CCC")
        with self.assertRaises(ValueError):
            c.add_boundary(0, neighbor_index=3)

    def test_add_boundary_with_invalid_neighbor(self):
        c = structure.Compound("CCC", src_mol="CC(=O)OCCC")
        with self.assertRaises(ValueError):
            c.add_boundary(0, "C", 3, "C")
