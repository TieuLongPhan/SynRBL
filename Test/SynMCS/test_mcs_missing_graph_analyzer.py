import sys
from pathlib import Path
import unittest
import pandas as pd
from rdkit.Chem import rdFMCS
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
from SynRBL.SynMCS.mcs_missing_graph_analyzer import MCSMissingGraphAnalyzer  

class TestMCSMissingGraphAnalyzer(unittest.TestCase):

    def setUp(self):
        self.analyzer = MCSMissingGraphAnalyzer()

    def test_get_smiles(self):
        # Test extraction of SMILES strings from reaction dictionary
        reaction_dict = {'reactants': 'CCO', 'products': 'CCOCC'}
        reactants, products = self.analyzer.get_smiles(reaction_dict)
        self.assertEqual(reactants, 'CCO')
        self.assertEqual(products, 'CCOCC')

    def test_convert_smiles_to_molecule(self):
        # Test conversion of SMILES string to molecule
        smiles = 'CCO'
        mol = self.analyzer.convert_smiles_to_molecule(smiles)
        self.assertIsNotNone(mol)

    def test_find_maximum_common_substructure(self):
        # Test finding MCS between two molecules
        mol1 = self.analyzer.convert_smiles_to_molecule('CCO')
        mol2 = self.analyzer.convert_smiles_to_molecule('CCOC')
        mcs_mol = self.analyzer.find_maximum_common_substructure(mol1, mol2, params=None)
        self.assertIsNotNone(mcs_mol)

    def test_IterativeMCSReactionPairs(self):
        # Test the iterative MCS reaction pairs method
        reactants = ['CCO', 'CC']
        products = 'CCOCC'
        params = rdFMCS.MCSParameters()
        params.AtomTyper = rdFMCS.AtomCompare.CompareElements
        params.BondTyper = rdFMCS.BondCompare.CompareOrder
        params.Timeout = 60
        params.BondCompareParameters.RingMatchesRingOnly = True
        params.BondCompareParameters.CompleteRingsOnly = True
        reactant_mols = [self.analyzer.convert_smiles_to_molecule(smiles) for smiles in reactants]
        product_mol = self.analyzer.convert_smiles_to_molecule(products)
        mcs_list, sorted_reactants = self.analyzer.IterativeMCSReactionPairs(reactant_mols, product_mol, params)
        self.assertIsNotNone(mcs_list)
        self.assertEqual(len(sorted_reactants), len(reactants))

    def test_fit(self):
        # Test the fit method
        reaction_dict = {'reactants': 'CCO.CC', 'products': 'CCOCC'}
        mcs_list, sorted_reactants, product_mol = self.analyzer.fit(reaction_dict)
        self.assertIsNotNone(mcs_list)
        self.assertEqual(len(sorted_reactants), 2)
        self.assertIsNotNone(product_mol)

if __name__ == '__main__':
    unittest.main()
