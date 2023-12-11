import unittest

from rdkit import Chem

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
        product_mol = [Chem.MolFromSmiles(product_smiles)]
        result = self.finder.find_single_graph([product_mol], reactant_mol)
        self.assertIsNotNone(result)
        self.assertIn('smiles', result)
        self.assertIn('boundary_atoms_products', result)
        self.assertIn('nearest_neighbor_products', result)

    def test_find_single_graph_parallel(self):
        # Test finding single graph in parallel
        reactant_smiles = ['CCO']
        product_smiles = ['CCO.C']
        reactant_mols = [Chem.MolFromSmiles(smiles) for smiles in reactant_smiles]
        product_mols = [Chem.MolFromSmiles(smiles) for smiles in product_smiles]
        results = self.finder.find_single_graph_parallel(product_mols, reactant_mols)
        self.assertIsNotNone(results)
        for result in results:
            self.assertIn('smiles', result)
            self.assertIn('boundary_atoms_products', result)
            self.assertIn('nearest_neighbor_products', result)

if __name__ == '__main__':
    unittest.main()
