import unittest
import copy

from rdkit import Chem
from synrbl.SynMCSImputer.SubStructure.substructure_analyzer import SubstructureAnalyzer
from synrbl.SynMCSImputer.MissingGraph.find_missing_graphs import FindMissingGraphs


class TestFindMissingGraphs(unittest.TestCase):
    def setUp(self):
        self.fm = FindMissingGraphs()
        self.mol = Chem.MolFromSmiles("CC(=O)N(C)C1=CC=CC=C1")
        self.mcs_mol = Chem.MolFromSmarts("[#6]-[#7]-[#6]1:[#6]:[#6]:[#6]:[#6]:[#6]:1")

    def test_find_missing_parts_pairs(self):
        # Assuming that the MCS is a part of the molecule
        (
            missing_parts,
            boundary_atoms,
            nearest_neighbors,
        ) = self.fm.find_missing_parts_pairs([self.mol], [self.mcs_mol])

        # Check if missing_parts contains valid molecules
        self.assertTrue(
            all(isinstance(part, Chem.Mol) or part is None for part in missing_parts)
        )
        # Check boundary atoms and nearest neighbors lists
        self.assertIsInstance(boundary_atoms, list)
        self.assertIsInstance(nearest_neighbors, list)

    def test_map_parent_to_child(self):
        # Example to map parent to child atom indices
        parent_mol = self.mol
        child_mol = self.mcs_mol
        atoms_to_remove = set()
        analyzer = SubstructureAnalyzer()
        substructure_match = analyzer.identify_optimal_substructure(
            parent_mol=parent_mol, child_mol=child_mol
        )
        if substructure_match:
            atoms_to_remove.update(substructure_match)

        left_number = []
        for i in range(parent_mol.GetNumAtoms()):
            if i not in substructure_match:
                left_number.append(i)
        print(left_number)

        missing_part = Chem.RWMol(parent_mol)
        for idx in sorted(atoms_to_remove, reverse=True):
            missing_part.RemoveAtom(idx)
        missing_part_old = copy.deepcopy(missing_part)
        key_base = left_number  # Example indices in the parent molecule

        atom_mapping = self.fm.map_parent_to_child(
            missing_part_old, missing_part, key_base
        )
        self.assertIsInstance(atom_mapping, dict)
        self.assertTrue(all(isinstance(idx, int) for idx in atom_mapping.values()))

    def test_is_mapping_correct(self):
        # Example molecule and mapping for testing
        mol = self.mol
        symbol_to_index = {"C": 1, "O": 2}  # Example mapping

        is_correct = self.fm.is_mapping_correct(mol, symbol_to_index)
        self.assertIsInstance(is_correct, bool)


if __name__ == "__main__":
    unittest.main()
