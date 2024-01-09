import unittest
import sys
from pathlib import Path
root_dir = Path(__file__).parents[3]
sys.path.append(str(root_dir))
import unittest
from rdkit import Chem
from SynRBL.SynMCSImputer.MissingGraph.find_graph_dict import find_single_graph, find_single_graph_parallel, find_graph_dict, convert_smiles_to_mols, smiles_to_mol_parallel  
from unittest.mock import MagicMock

class TestFindGraphFunctions(unittest.TestCase):

    def setUp(self):
        # Example molecules for testing
        self.mcs_mol_list = [Chem.MolFromSmiles("CC"), Chem.MolFromSmiles("C")]
        self.sorted_reactants_mol_list = [Chem.MolFromSmiles("CCO"), Chem.MolFromSmiles("CO")]

    def test_find_single_graph(self):
        result = find_single_graph(self.mcs_mol_list, self.sorted_reactants_mol_list)
        self.assertIsInstance(result, dict)
        self.assertIn('smiles', result)
        self.assertIn('boundary_atoms_products', result)
        self.assertIn('nearest_neighbor_products', result)
        self.assertIn('issue', result)

    def test_find_single_graph_parallel(self):
        result = find_single_graph_parallel(self.mcs_mol_list, self.sorted_reactants_mol_list, n_jobs=2)
        self.assertIsInstance(result, list)
        for item in result:
            self.assertIsInstance(item, dict)
            self.assertIn('smiles', item)
            self.assertIn('boundary_atoms_products', item)
            self.assertIn('nearest_neighbor_products', item)
            self.assertIn('issue', item)

    # def test_find_graph_dict(self):
    #     # This function requires mocking the I/O operations
    #     # Assume we have a MagicMock that simulates load_database and save_database
    #     with self.assertRaises(Exception):  # Adjust based on the expected exception or result
    #         find_graph_dict("test_path", "save_path", save=False, n_jobs=2, use_findMCS=True)

    def test_convert_smiles_to_mols(self):
        smiles_list = ["CC", "C"]
        result = convert_smiles_to_mols(smiles_list)
        self.assertIsInstance(result, list)
        for mol in result:
            self.assertIsInstance(mol, Chem.Mol)

    def test_smiles_to_mol_parallel(self):
        smiles_lists = [["CC", "C"], ["CO", "CCO"]]
        result = smiles_to_mol_parallel(smiles_lists, n_jobs=4)
        self.assertIsInstance(result, list)
        for sublist in result:
            for mol in sublist:
                self.assertIsInstance(mol, Chem.Mol or type(None))  

if __name__ == '__main__':
    unittest.main()
