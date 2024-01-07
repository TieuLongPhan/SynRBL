#import unittest
#from rdkit import Chem
#from SynRBL.SynMCS.mol_merge import *
#import matplotlib.pyplot as plt
#
#
#
#
#class TestMergeMols(unittest.TestCase):
#    def __test_merge(self, smiles1, smiles2, idx1, idx2, expected_result):
#        mol1 = Chem.MolFromSmiles(smiles1)
#        mol2 = Chem.MolFromSmiles(smiles2)
#        merge_result = merge_mols(
#            mol1, mol2, idx1, idx2, mol1_track=[idx1], mol2_track=[idx2]
#        )
#        # print(merge_result['rule'].name)
#        mmol = Chem.RemoveHs(merge_result["mol"])
#        actual_result = Chem.MolToSmiles(mmol)
#        self.assertEqual(Chem.CanonSmiles(expected_result), actual_result)
#
#    def test_merge_with_default_single_bond(self):
#        self.__test_merge("CC1(C)OBOC1(C)C", "CCC", 4, 1, "CC1(C)OB(-C(C)C)OC1(C)C")
#
#    def test_merge_to_silicium_radical(self):
#        self.__test_merge("O", "C[Si](C)C(C)(C)C", 0, 1, "C[Si](O)(C)C(C)(C)C")
#
#    def test_merge_with_phosphor_double_bond(self):
#        self.__test_merge(
#            "O",
#            "c1ccc(P(c2ccccc2)c2ccccc2)cc1",
#            0,
#            4,
#            "c1ccc(P(=O)(c2ccccc2)c2ccccc2)cc1",
#        )
#
#    def test_merge_radical_halogen_exchange(self):
#        self.__test_merge("Cl", "CCCC[Sn](CCCC)CCCC", 0, 4, "CCCC[Sn+](CCCC)CCCC.[Cl-]")
#
#    def test_merge_halogen_bond_restriction(self):
#        pass
#        # self.__test_merge('', '', 0, 0, '')
#
#
#class TestMergeExpand(unittest.TestCase):
#    def __test_expand(self, smiles, bounds, neighbors, expected_result):
#        mol = Chem.MolFromSmiles(smiles)
#        merge_result = merge_expand(mol, bounds, neighbors)
#        # print(merge_result['compound_rules'][0].name)
#        mmol = Chem.RemoveHs(merge_result["mol"])
#        actual_result = Chem.MolToSmiles(mmol)
#        self.assertEqual(Chem.CanonSmiles(expected_result), actual_result)
#
#    def test_expand_O_next_to_O_or_N(self):
#        self.__test_expand("O=COCc1ccccc1", [1], ["O"], "O=C(O)OCc1ccccc1")
#
#    def test_expand_multiple_bounds(self):
#        self.__test_expand("O=Cc1ccccc1C=O", [1, 8], ["O", "O"], "O=C(O)c1ccccc1C(O)=O")
#
#    def test_expand(self):
#        self.__test_expand("C[Si](C)C(C)(C)C", [1], ["O"], "C[Si](O)(C)C(C)(C)C")
#
#    def test_leave_O_bound_as_is(self):
#        self.__test_expand("CC(C)(C)OC(=O)O", [7], ["C"], "CC(C)(C)OC(=O)O")
#
#    def test_leave_NN_bound_as_is(self):
#        self.__test_expand("[N-]=[NH2+]", [0], ["N"], "[N-]=[NH2+]")
#
#
#class TestMerge(unittest.TestCase):
#    def __test_merge(self, smiles, bounds, neighbors, expected_results):
#        mols = [Chem.MolFromSmiles(s) for s in smiles]
#        merge_result = merge(mols, bounds, neighbors)
#        mmols = [Chem.RemoveHs(m["mol"]) for m in merge_result]
#        for exp, act in zip(expected_results, mmols):
#            self.assertEqual(Chem.CanonSmiles(exp), Chem.MolToSmiles(act))
#
#    def test_merge_two_compounds(self):
#        self.__test_merge(
#            ["CC1(C)OBOC1(C)C", "Br"],
#            [[{"B": 4}], [{"Br": 0}]],
#            [[{"C": 5}], [{"C": 13}]],
#            ["CC1(C)OB(Br)OC1(C)C"],
#        )
#
#    def test_split_and_expand(self):
#        self.__test_merge(
#            ["C.O"], [[{"C": 0}, {"O": 1}]], [[{"O": 1}, {"C": 2}]], ["CO", "O"]
#        )
#
#    def test_merge_expand_multiple(self):
#        self.__test_merge(
#            ["O=Cc1ccccc1C=O"],
#            [[{"C": 1}, {"C": 8}]],
#            [[{"N": 9}, {"N": 11}]],
#            ["O=C(O)c1ccccc1C(O)=O"],
#        )
#
#    # def test_invalid_structure(self):
#    #    with self.assertRaises(NoCompoundError):
#    #        self.__test_merge(
#    #            ["NBr.O"],
#    #            [[{"N": 0}, {"O": 2}, {"N": 0}]],
#    #            [[{"C": 1}, {"C": 4}, {"C": 4}]],
#    #            ["NBr", "O"],
#    #        )
