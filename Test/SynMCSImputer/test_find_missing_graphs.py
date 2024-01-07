import unittest
from rdkit import Chem
import sys
from pathlib import Path
import unittest
import pandas as pd
from rdkit.Chem import rdFMCS
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
from SynRBL.SynMCS.MissingGraph.find_missing_graphs import FindMissingGraphs

class TestFindMissingGraphs(unittest.TestCase):

    def setUp(self):
        self.finder = FindMissingGraphs()

    def test_find_missing_parts_pairs(self):
        # Test finding missing parts pairs
        mol = Chem.MolFromSmiles('CCO')
        mcs_mol = Chem.MolFromSmiles('CC')
        missing_parts, boundary_atoms, nearest_neighbors = self.finder.find_missing_parts_pairs([mol], [mcs_mol])
        self.assertIsNotNone(missing_parts)
        self.assertIsNotNone(boundary_atoms)
        self.assertIsNotNone(nearest_neighbors)

    def test_find_single_graph(self):
        # Test finding single graph
        reactant_smiles = 'CCO'
        product_smiles = 'CCO.C'
        reactant_mol = [Chem.MolFromSmiles(reactant_smiles)]
        product_mol = Chem.MolFromSmiles(product_smiles)
        missing_parts, boundary_atoms, nearest_neighbors  = self.finder.find_missing_parts_pairs([product_mol], reactant_mol)
        self.assertIsNotNone(missing_parts)
        self.assertIsNotNone(boundary_atoms)
        self.assertIsNotNone(nearest_neighbors)

if __name__ == '__main__':
    unittest.main()
