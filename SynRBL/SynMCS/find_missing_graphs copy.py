from rdkit import Chem
from rdkit.Chem import rdFMCS
from joblib import Parallel, delayed

class FindMissingGraphs:
    """
    A class for finding missing parts, boundary atoms, and nearest neighbors in a list of reactant molecules.

    Usage:
    1. Create an instance of the class.
    2. Use the class methods to find missing parts, boundary atoms, and nearest neighbors for a list of molecules.

    Example:
    ```
    fm = FindMissingGraphs()
    missing_results = fm.find_single_graph(mcs_mol_list, sorted_reactants_mol_list)
    ```

    Note: This class requires the RDKit library to be installed.

    Attributes:
    None

    Methods:
    - find_missing_parts_pairs: Analyze a list of molecules and identify missing parts, boundary atoms, and nearest neighbors.
    - find_single_graph: Find missing parts, boundary atoms, and nearest neighbors for a list of reactant molecules.
    - find_single_graph_parallel: Find missing parts, boundary atoms, and nearest neighbors in parallel for a list of reactant molecules.
    """

    def __init__(self):
        pass

    @staticmethod
    def find_missing_parts_pairs(mol_list, mcs_list=None, use_findMCS=False, params=None):
        """
        This function analyzes each molecule in a given list and identifies the parts that are not 
        present in the corresponding Maximum Common Substructure (MCS). It also finds the boundary 
        atoms and nearest neighbors for each molecule.

        Parameters:
        - mol_list (list of rdkit.Chem.Mol): The list of RDKit molecule objects to analyze.
        - mcs_list (list of rdkit.Chem.Mol or None): List of RDKit molecule objects representing MCS, 
        corresponding to each molecule in mol_list. If None, MCS will be calculated using RDKit's rdFMCS.
        - use_findMCS (bool): Whether to use RDKit's rdFMCS to find MCS and remove it from mol_list.
        - params (rdkit.Chem.rdFMCS.MCSParameters): Parameters for RDKit's rdFMCS.

        Returns:
        Tuple containing:
        - list of rdkit.Chem.Mol or None: RDKit molecule objects representing the missing parts 
        of each molecule, or None if no missing parts are found.
        - list of lists: Each sublist contains the boundary atoms of the corresponding molecule.
        - list of lists: Each sublist contains the nearest neighbors of the corresponding molecule.
        """
        missing_parts_list = []
        boundary_atoms_lists = []
        nearest_neighbor_lists = []

        for mol, mcs_mol in zip(mol_list, mcs_list):
            atoms_to_remove = set()
            boundary_atoms_list = []
            nearest_neighbor_list = []

            if use_findMCS:
                # Calculate MCS using RDKit's rdFMCS
                mcs = rdFMCS.FindMCS([mol, mcs_mol], params)
                mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
            
            try:
                if mcs_mol:
                    # Special case handling (e.g., single oxygen atom)
                    if Chem.MolToSmiles(mcs_mol) == 'O':
                        smarts_pattern = '[OH]'
                        smarts_mol = Chem.MolFromSmarts(smarts_pattern)
                        substructure_match = mol.GetSubstructMatch(smarts_mol)
                        rw_mol = Chem.RWMol(mol)
                        rw_mol.RemoveAtom(substructure_match[0])
                        missing_part = rw_mol.GetMol()
                        boundary_atoms = [{'O': 0}]
                        nearest_atoms = [{'O': 0}]
                    else:
                        raise ValueError
            except:
                if mcs_mol:
                    # Finding substructure matches
                    substructure_match = mol.GetSubstructMatch(mcs_mol)
                    if substructure_match:
                        atoms_to_remove.update(substructure_match)

                    # Creating the molecule of missing parts
                    missing_part = Chem.RWMol(mol)
                    for idx in sorted(atoms_to_remove, reverse=True):
                        missing_part.RemoveAtom(idx)

                    # re-index
                    try:
                        missing_part = Chem.MolFromSmiles(Chem.MolToSmiles(missing_part))
                    except:
                        missing_part = missing_part
                    atom_mapping = FindMissingGraphs.map_parent_to_child(mol, missing_part)
                    
                    # Mapping indices from original to missing part molecule
                    #index_mapping = {idx: i for i, idx in enumerate(sorted(set(range(mol.GetNumAtoms())) - atoms_to_remove))}

                    boundary_atoms = []
                    nearest_atoms = []

                    # Identifying boundary atoms and nearest neighbors
                    for atom_idx in substructure_match:
                        if atom_idx < mol.GetNumAtoms():
                            atom_symbol = mol.GetAtomWithIdx(atom_idx).GetSymbol()
                            neighbors = mol.GetAtomWithIdx(atom_idx).GetNeighbors()
                            # Loop through neighbors to find boundary atoms and nearest neighbors
                            for neighbor in neighbors:
                                if neighbor.GetIdx() not in substructure_match:
                                    nearest_atoms.append({atom_symbol: atom_idx})
                                    #renumerate_idx = index_mapping.get(neighbor.GetIdx(), -1)
                                    renumerate_idx = atom_mapping.get(neighbor.GetIdx(), -1)
                                    if renumerate_idx != -1:
                                        boundary_atoms.append({neighbor.GetSymbol(): renumerate_idx})

                if boundary_atoms:
                    boundary_atoms_list.append(boundary_atoms)
                if nearest_atoms:
                    nearest_neighbor_list.append(nearest_atoms)

            try:
                Chem.SanitizeMol(missing_part)
                if missing_part.GetNumAtoms() > 0:
                    missing_parts_list.append(missing_part)
                    boundary_atoms_lists.extend(boundary_atoms_list)
                    nearest_neighbor_lists.extend(nearest_neighbor_list)
                else:
                    #missing_parts_list.append(None)
                    boundary_atoms_lists.extend([])
                    nearest_neighbor_lists.extend([])
            except:
                #missing_parts_list.append(None)
                boundary_atoms_lists.extend([])
                nearest_neighbor_lists.extend([])

        return missing_parts_list, boundary_atoms_lists, nearest_neighbor_lists
    

    @staticmethod
    def find_single_graph(mcs_mol_list, sorted_reactants_mol_list, use_findMCS=True):
        """
        Find missing parts, boundary atoms, and nearest neighbors for a list of reactant molecules
        using a corresponding list of MCS (Maximum Common Substructure) molecules.

        Parameters:
        - mcs_mol_list (list of rdkit.Chem.Mol): List of RDKit molecule objects representing the MCS,
        corresponding to each molecule in sorted_reactants_mol_list.
        - sorted_reactants_mol_list (list of rdkit.Chem.Mol): The list of RDKit molecule objects to analyze.

        Returns:
        - Dictionary containing:
        - 'smiles' (list of list of str): SMILES representations of the missing parts for each molecule.
        - 'boundary_atoms_products' (list of list of dict): Lists of boundary atoms for each molecule.
        - 'nearest_neighbor_products' (list of list of dict): Lists of nearest neighbors for each molecule.
        - 'issue' (list): Any issues encountered during processing.
        """
        missing_results = {'smiles': [], 'boundary_atoms_products': [], 'nearest_neighbor_products': [], 'issue': []}
        for i in zip(sorted_reactants_mol_list, mcs_mol_list):
            try:
                mols, boundary_atoms_products, nearest_neighbor_products = FindMissingGraphs.find_missing_parts_pairs(i[0], i[1], use_findMCS=use_findMCS)
                missing_results['smiles'].append([Chem.MolToSmiles(mol) for mol in mols])
                missing_results['boundary_atoms_products'].append(boundary_atoms_products)
                missing_results['nearest_neighbor_products'].append(nearest_neighbor_products)
                missing_results['issue'].append([])
            except Exception as e:
                missing_results['smiles'].append([])
                missing_results['boundary_atoms_products'].append([])
                missing_results['nearest_neighbor_products'].append([])
                missing_results['issue'].append(str(e))
        return missing_results

    @staticmethod
    def find_single_graph_parallel(mcs_mol_list, sorted_reactants_mol_list, n_jobs=-1, use_findMCS=True):
        """
        Find missing parts, boundary atoms, and nearest neighbors for a list of reactant molecules
        using a corresponding list of MCS (Maximum Common Substructure) molecules in parallel.

        Parameters:
        - mcs_mol_list (list of rdkit.Chem.Mol): List of RDKit molecule objects representing the MCS,
        corresponding to each molecule in sorted_reactants_mol_list.
        - sorted_reactants_mol_list (list of rdkit.Chem.Mol): The list of RDKit molecule objects to analyze.
        - n_jobs (int): The number of parallel jobs to run. Default is -1, which uses all available CPU cores.

        Returns:
        - List of dictionaries, where each dictionary contains:
        - 'smiles' (list of str): SMILES representations of the missing parts for each molecule.
        - 'boundary_atoms_products' (list of dict): Lists of boundary atoms for each molecule.
        - 'nearest_neighbor_products' (list of dict): Lists of nearest neighbors for each molecule.
        - 'issue' (str): Any issues encountered during processing.
        """
        def process_single_pair(reactant_mol, mcs_mol, use_findMCS=True):
            try:
                mols, boundary_atoms_products, nearest_neighbor_products = FindMissingGraphs.find_missing_parts_pairs(reactant_mol, mcs_mol, use_findMCS=use_findMCS)
                return {
                    'smiles': [Chem.MolToSmiles(mol) for mol in mols],
                    'boundary_atoms_products': boundary_atoms_products,
                    'nearest_neighbor_products': nearest_neighbor_products,
                    'issue': ''
                }
            except Exception as e:
                return {
                    'smiles': [],
                    'boundary_atoms_products': [],
                    'nearest_neighbor_products': [],
                    'issue': str(e)
                }

        results = Parallel(n_jobs=n_jobs)(delayed(process_single_pair)(reactant_mol, mcs_mol, use_findMCS=use_findMCS) for reactant_mol, mcs_mol in zip(sorted_reactants_mol_list, mcs_mol_list))
        return results

    
    @staticmethod
    def map_parent_to_child(parent_mol, child_mol):
        # Get atom indices in the parent molecule that match the entire child molecule
        parent_mcs_indices = parent_mol.GetSubstructMatch(child_mol)

        # Create a mapping of parent atom indices to child atom indices
        atom_mapping = {}
        for child_idx, parent_idx in enumerate(parent_mcs_indices):
            atom_mapping[parent_idx] = child_idx

        return atom_mapping
