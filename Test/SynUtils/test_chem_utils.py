import unittest
import itertools
from SynRBL.SynUtils.chem_utils import remove_atom_mapping, normalize_smiles

class TestRemoveAtomMapping(unittest.TestCase):
    def test_single_simple_smiles(self):
        smiles = "[O:0][CH:1][CH3:2][Br:13][OH:101]"
        exp_smiles = "OCCBrO"
        result = remove_atom_mapping(smiles)
        self.assertEqual(exp_smiles, result)

    def test_smiles(self):
        smiles = "[C:0]([CH:1]1[NH:2][O:3]1)(=[O:4])[OH:5]"
        exp_smiles = "C(C1NO1)(=O)O"
        result = remove_atom_mapping(smiles)
        self.assertEqual(exp_smiles, result)

    def test_two_compounds(self):
        smiles = "[OH2:0].[CH3:1][CH2:2][O:3]"
        exp_smiles = "O.CCO"
        result = remove_atom_mapping(smiles)
        self.assertEqual(exp_smiles, result)

    def test_reaction(self):
        smiles = "[CH3:1][CH:2]=[O:3]>>[CH3:1][C:2](=[O:3])[OH:4]"
        exp_smiles = "CC=O>>CC(=O)O"
        result = remove_atom_mapping(smiles)
        self.assertEqual(exp_smiles, result)

    def test_keep_atom_brackets(self):
        smiles = "C[SiH:3](C)[C:5]"
        exp_smiles = "C[SiH](C)C"
        result = remove_atom_mapping(smiles)
        self.assertEqual(exp_smiles, result)

    def test_keep_radical(self):
        smiles = "[C:10][Si+:5](C)C"
        exp_smiles = "C[Si+](C)C"
        result = remove_atom_mapping(smiles)
        self.assertEqual(exp_smiles, result)

class TestNormalizeReaction(unittest.TestCase):
    def test_remove_mappings(self):
        smiles = "[CH3:1][CH:2]=[O:3]>>[CH3:1][C:2](=[O:3])[OH:4]"
        exp_smiles = "CC=O>>CC(=O)O"
        result = normalize_smiles(smiles)
        self.assertEqual(exp_smiles, result)
    
    def test_order_compounds(self):
        smiles = "[C:0][O:1].OCO.O>>CO.O.OCO"
        exp_smiles = "OCO.CO.O>>OCO.CO.O"
        result = normalize_smiles(smiles)
        self.assertEqual(exp_smiles, result)
    
    def test_order_equal_len_compounds(self):
        smiles = ["CCO", "CC=O", "CCN", "CCCO", "CCS"]
        exp_smiles = "CCCO.CC=O.CCS.CCO.CCN"
        for s in itertools.permutations(smiles): 
            result = normalize_smiles(".".join(s))
            self.assertEqual(exp_smiles, result)

    def test_canonial_smiles(self):
        smiles = "CC(O)=O"
        exp_smiles = "CC(=O)O"
        result = normalize_smiles(smiles)
        self.assertEqual(exp_smiles, result)