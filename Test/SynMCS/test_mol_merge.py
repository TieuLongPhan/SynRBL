import unittest
from rdkit import Chem
from SynRBL.SynMCS.mol_merge import *
import matplotlib.pyplot as plt


class TestAtomCondition(unittest.TestCase):
    def __check_cond(self, cond, smiles, idx, expected_result, neighbor=None):
        mol = Chem.MolFromSmiles(smiles)
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


class TestProperty(unittest.TestCase):
    def test_parsing(self):
        prop = Property("A")
        self.assertIn("A", prop.pos_values)
        self.assertNotIn("A", prop.neg_values)
        prop = Property(["A", "B"])
        self.assertIn("A", prop.pos_values)
        self.assertIn("B", prop.pos_values)
        prop = Property("!A")
        self.assertIn("A", prop.neg_values)
        self.assertNotIn("A", prop.pos_values)
        prop = Property(["!A", "!B"])
        self.assertIn("A", prop.neg_values)
        self.assertIn("B", prop.neg_values)
        prop = Property(["!A", "B"])
        self.assertIn("A", prop.neg_values)
        self.assertIn("B", prop.pos_values)
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


class TestMergeMols(unittest.TestCase):
    def __test_merge(self, smiles1, smiles2, idx1, idx2, expected_result):
        mol1 = Chem.MolFromSmiles(smiles1)
        mol2 = Chem.MolFromSmiles(smiles2)
        merge_result = merge_mols(
            mol1, mol2, idx1, idx2, mol1_track=[idx1], mol2_track=[idx2]
        )
        # print(merge_result['rule'].name)
        mmol = Chem.RemoveHs(merge_result["mol"])
        actual_result = Chem.MolToSmiles(mmol)
        self.assertEqual(Chem.CanonSmiles(expected_result), actual_result)

    def test_merge_with_default_single_bond(self):
        self.__test_merge("CC1(C)OBOC1(C)C", "CCC", 4, 1, "CC1(C)OB(-C(C)C)OC1(C)C")

    def test_merge_to_silicium_radical(self):
        self.__test_merge("O", "C[Si](C)C(C)(C)C", 0, 1, "C[Si](O)(C)C(C)(C)C")

    def test_merge_with_phosphor_double_bond(self):
        self.__test_merge(
            "O", "CC(C)[P+](c1ccccc1)c1ccccc1", 0, 3, "CC(C)P(=O)(c1ccccc1)c1ccccc1"
        )

    def test_merge_radical_halogen_exchange(self):
        self.__test_merge("Cl", "CCCC[Sn](CCCC)CCCC", 0, 4, "CCCC[Sn+](CCCC)CCCC.[Cl-]")

    def test_merge_halogen_bond_restriction(self):
        pass
        # self.__test_merge('', '', 0, 0, '')


class TestMergeExpand(unittest.TestCase):
    def __test_expand(self, smiles, bounds, neighbors, expected_result):
        mol = Chem.MolFromSmiles(smiles)
        merge_result = merge_expand(mol, bounds, neighbors)
        # print(merge_result['compound_rules'][0].name)
        mmol = Chem.RemoveHs(merge_result["mol"])
        actual_result = Chem.MolToSmiles(mmol)
        self.assertEqual(Chem.CanonSmiles(expected_result), actual_result)

    def test_expand_O_next_to_O_or_N(self):
        self.__test_expand("O=COCc1ccccc1", [1], ["O"], "O=C(O)OCc1ccccc1")

    def test_expand_multiple_bounds(self):
        self.__test_expand("O=Cc1ccccc1C=O", [1, 8], ["O", "O"], "O=C(O)c1ccccc1C(O)=O")

    def test_expand(self):
        self.__test_expand("C[Si](C)C(C)(C)C", [1], ["O"], "C[Si](O)(C)C(C)(C)C")

    def test_leave_O_bound_as_is(self):
        self.__test_expand("CC(C)(C)OC(=O)O", [7], ["C"], "CC(C)(C)OC(=O)O")


class TestMerge(unittest.TestCase):
    def __test_merge(self, smiles, bounds, neighbors, expected_results):
        mols = [Chem.MolFromSmiles(s) for s in smiles]
        merge_result = merge(mols, bounds, neighbors)
        mmols = [Chem.RemoveHs(m["mol"]) for m in merge_result]
        for exp, act in zip(expected_results, mmols):
            self.assertEqual(Chem.CanonSmiles(exp), Chem.MolToSmiles(act))

    def test_merge_two_compounds(self):
        self.__test_merge(
            ["CC1(C)OBOC1(C)C", "Br"],
            [[{"B": 4}], [{"Br": 0}]],
            [[{"C": 5}], [{"C": 13}]],
            ["CC1(C)OB(Br)OC1(C)C"],
        )

    def test_split_and_expand(self):
        self.__test_merge(
            ["C.O"], [[{"C": 0}, {"O": 1}]], [[{"O": 1}, {"C": 2}]], ["CO", "O"]
        )

    def test_merge_expand_multiple(self):
        self.__test_merge(
            ["O=Cc1ccccc1C=O"],
            [[{"C": 1}, {"C": 8}]],
            [[{"N": 9}, {"N": 11}]],
            ["O=C(O)c1ccccc1C(O)=O"],
        )

    def test_invalid_structure(self):
        with self.assertRaises(NoCompoundError):
            self.__test_merge(
                ["NBr.O"],
                [[{"N": 0}, {"O": 2}, {"N": 0}]],
                [[{"C": 1}, {"C": 4}, {"C": 4}]],
                ["NBr", "O"],
            )
