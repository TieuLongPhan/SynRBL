import copy

from rdkit import Chem
from rdkit.rdBase import BlockLogs
from synrbl.SynMCSImputer.SubStructure.substructure_analyzer import SubstructureAnalyzer
from synrbl.SynMCSImputer.MissingGraph.molcurator import MoleculeCurator
from typing import List, Dict, Optional, Tuple


class FindMissingGraphs:
    """
    A class for finding missing parts, boundary atoms, and nearest neighbors in
    a list of reactant molecules.

    Usage:
    1. Create an instance of the class.
    2. Use the class methods to find missing parts, boundary atoms, and nearest
    neighbors for a list of molecules.

    Example:
    ```
    fm = FindMissingGraphs()
    missing_results = fm.find_single_graph(mcs_mol_list, sorted_reactants_mol_list)
    ```

    Note: This class requires the RDKit library to be installed.

    Attributes:
    None

    Methods:
    - find_missing_parts_pairs: Analyze a list of molecules and identify
        missing parts, boundary atoms, and nearest neighbors.
    - find_single_graph: Find missing parts, boundary atoms, and nearest
        neighbors for a list of reactant molecules.
    - find_single_graph_parallel: Find missing parts, boundary atoms, and
        nearest neighbors in parallel for a list of reactant molecules.
    """

    def __init__(self):
        pass

    @staticmethod
    def find_missing_parts_pairs(
        mol_list: List[Chem.Mol],
        mcs_list: Optional[List[Chem.Mol]] = None,
        substructure_optimize: bool = True,
    ) -> Tuple[
        Optional[List[Chem.Mol]], List[List[Dict[str, int]]], List[List[Dict[str, int]]]
    ]:
        """
        This function analyzes each molecule in a given list and identifies the
        parts that are not present in the corresponding Maximum Common
        Substructure (MCS). It also finds the boundary atoms and nearest
        neighbors for each molecule.

        Parameters:
        - mol_list (list of rdkit.Chem.Mol): The list of RDKit molecule objects
            to analyze.
        - mcs_list (list of rdkit.Chem.Mol or None): List of RDKit molecule
            objects representing MCS,
        corresponding to each molecule in mol_list. If None, MCS will be
            calculated using RDKit's rdFMCS.
        - use_findMCS (bool): Whether to use RDKit's rdFMCS to find MCS and
            remove it from mol_list.
        - params (rdkit.Chem.rdFMCS.MCSParameters): Parameters for RDKit's rdFMCS.

        Returns:
        Tuple containing:
        - list of rdkit.Chem.Mol or None: RDKit molecule objects representing
            the missing parts of each molecule, or None if no missing parts
            are found.
        - list of lists: Each sublist contains the boundary atoms of the
            corresponding molecule.
        - list of lists: Each sublist contains the nearest neighbors of the
            corresponding molecule.
        """
        block = BlockLogs()
        missing_parts_list = []
        boundary_atoms_lists = []
        nearest_neighbor_lists = []

        for mol, mcs_mol in zip(mol_list, mcs_list):
            atoms_to_remove = set()
            boundary_atoms_list = []
            nearest_neighbor_list = []

            if mcs_mol.GetNumAtoms() > 0:
                # Finding substructure matches
                if Chem.MolToSmiles(mcs_mol) == "O":
                    # hardcode special case OC(O)OH
                    smarts_pattern = "[OH]"
                    smarts_mol = Chem.MolFromSmarts(smarts_pattern)
                    substructure_match = mol.GetSubstructMatch(smarts_mol)
                    # if no substructure match is found, use rdkit's
                    # substructure matching for mcs_mol, this not special case
                    if not substructure_match:
                        substructure_match = mol.GetSubstructMatch(mcs_mol)
                else:
                    if substructure_optimize:
                        analyzer = SubstructureAnalyzer()
                        substructure_match = analyzer.identify_optimal_substructure(
                            parent_mol=mol, child_mol=mcs_mol
                        )
                    else:
                        substructure_match = mol.GetSubstructMatch(mcs_mol)

                if substructure_match:
                    atoms_to_remove.update(substructure_match)

                left_number = []
                for i in range(mol.GetNumAtoms()):
                    if i not in substructure_match:
                        left_number.append(i)

                # Creating the molecule of missing parts
                missing_part = Chem.RWMol(mol)
                for idx in sorted(atoms_to_remove, reverse=True):
                    missing_part.RemoveAtom(idx)

                missing_part_old = copy.deepcopy(missing_part)

                if missing_part is not None:
                    missing_part_smiles = Chem.MolToSmiles(missing_part)
                    try:
                        missing_part = Chem.MolFromSmiles(
                            missing_part_smiles, sanitize=False
                        )
                        Chem.SanitizeMol(missing_part)

                    except Exception:
                        missing_part = MoleculeCurator.manual_kekulize(
                            missing_part_smiles
                        )

                    missing_part = MoleculeCurator.add_hydrogens_to_radicals(
                        missing_part
                    )
                    atom_mapping = FindMissingGraphs.map_parent_to_child(
                        missing_part_old, missing_part, left_number
                    )

                else:
                    index_mapping = {
                        idx: i
                        for i, idx in enumerate(
                            sorted(set(range(mol.GetNumAtoms())) - atoms_to_remove)
                        )
                    }

                boundary_atoms = []
                nearest_atoms = []

                # Identifying boundary atoms and nearest neighbors
                for atom_idx in substructure_match:
                    # display(mol)
                    if atom_idx < mol.GetNumAtoms():
                        atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                        neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                        # Loop through neighbors to find boundary atoms
                        # and nearest neighbors
                        for neighbor in neighbors:
                            if neighbor.GetIdx() not in substructure_match:
                                nearest_atoms.append({atom_symbol: atom_idx})

                                if missing_part:
                                    renumerate_idx = atom_mapping.get(
                                        neighbor.GetIdx(), -1
                                    )
                                else:
                                    renumerate_idx = index_mapping.get(
                                        neighbor.GetIdx(), -1
                                    )
                                if renumerate_idx != -1:
                                    boundary_atoms.append(
                                        {neighbor.GetSymbol(): renumerate_idx}
                                    )
            else:
                missing_part = None
            if boundary_atoms:
                boundary_atoms_list.append(boundary_atoms)
            if nearest_atoms:
                nearest_neighbor_list.append(nearest_atoms)

            try:
                Chem.SanitizeMol(missing_part)
                if missing_part.GetNumAtoms() > 0:
                    missing_part = MoleculeCurator.standardize_diazo_charge(
                        missing_part
                    )
                    missing_parts_list.append(missing_part)
                    boundary_atoms_lists.extend(boundary_atoms_list)
                    nearest_neighbor_lists.extend(nearest_neighbor_list)
                else:
                    missing_parts_list.append(None)
                    boundary_atoms_lists.append(None)
                    nearest_neighbor_lists.append(None)
            except Exception:
                missing_parts_list.append(None)
                boundary_atoms_lists.append(None)
                nearest_neighbor_lists.append(None)

        del block
        return missing_parts_list, boundary_atoms_lists, nearest_neighbor_lists

    @staticmethod
    def map_parent_to_child(parent_mol, child_mol, key_base):
        # Get atom indices in the parent molecule that match the entire child molecule
        parent_mcs_indices = parent_mol.GetSubstructMatch(child_mol)

        # Create a mapping of parent atom indices to child atom indices
        atom_mapping = {}
        for child_idx, parent_idx in enumerate(parent_mcs_indices):
            map_key = key_base[parent_idx]
            atom_mapping[map_key] = child_idx

        return atom_mapping

    @staticmethod
    def is_mapping_correct(mol, symbol_to_index):
        # Convert the molecule to a dictionary of atom indices to symbols
        molecule_dict = {atom.GetIdx(): atom.GetSymbol() for atom in mol.GetAtoms()}

        # Check if the mappings are consistent
        for symbol, index in symbol_to_index.items():
            if index not in molecule_dict:
                return False
            if molecule_dict[index] != symbol:
                return False

        return True
