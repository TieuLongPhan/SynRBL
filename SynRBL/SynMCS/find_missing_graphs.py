from rdkit import Chem
def find_missing_parts_pairs(mol_list, mcs_list):
    """
    This function analyzes each molecule in a given list and identifies the parts that are not 
    present in the corresponding Maximum Common Substructure (MCS). It also finds the boundary 
    atoms and nearest neighbors for each molecule.

    Parameters:
    - mol_list (list of rdkit.Chem.Mol): The list of RDKit molecule objects to analyze.
    - mcs_list (list of rdkit.Chem.Mol): List of RDKit molecule objects representing MCS, 
    corresponding to each molecule in mol_list.

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
                # Finding substructure matches
                substructure_match = mol.GetSubstructMatch(mcs_mol)
                if substructure_match:
                    atoms_to_remove.update(substructure_match)

                # Creating the molecule of missing parts
                missing_part = Chem.RWMol(mol)
                for idx in sorted(atoms_to_remove, reverse=True):
                    missing_part.RemoveAtom(idx)

                # Mapping indices from original to missing part molecule
                index_mapping = {idx: i for i, idx in enumerate(sorted(set(range(mol.GetNumAtoms())) - atoms_to_remove))}

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
                                renumerate_idx = index_mapping.get(neighbor.GetIdx(), -1)
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