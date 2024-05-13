from typing import List, Tuple
from rdkit import Chem
from rdkit.Chem.rdchem import Mol


class SubstructureAnalyzer:
    """
    A class for analyzing substructures within molecules using RDKit.

    Methods:
    remove_substructure_atoms: Removes a specified substructure from a
        molecule and returns the number of resulting fragments.
    sort_substructures_by_fragment_count: Sorts a list of substructures based
        on the number of fragments resulting from their removal.
    identify_optimal_substructure: Identifies the most relevant substructure
        from a list of potential substructures based on fragment count.
    """

    def __init__(self):
        pass

    def remove_substructure_atoms(
        self, parent_mol: Mol, substructure: Tuple[int, ...]
    ) -> int:
        """
        Removes specified atoms (substructure) from a molecule and returns the
        number of resulting fragments.

        Parameters:
        parent_mol (Mol): The parent molecule.
        substructure (Tuple[int, ...]): Indices of atoms in the substructure.

        Returns:
        int: The number of fragments resulting from the removal of the
            substructure.
        """
        rw_mol = Chem.RWMol(parent_mol)
        for atom_idx in sorted(substructure, reverse=True):
            if atom_idx < rw_mol.GetNumAtoms():
                rw_mol.RemoveAtom(atom_idx)
        new_mol = rw_mol.GetMol()
        return len(Chem.GetMolFrags(new_mol))

    def sort_substructures_by_fragment_count(
        self, substructures: List[Tuple[int, ...]], fragment_counts: List[int]
    ) -> List[Tuple[int, ...]]:
        """
        Sorts a list of substructures based on the number of fragments
        resulting from their removal.

        Parameters:
        substructures (List[Tuple[int, ...]]): List of substructures
            represented by atom indices.
        fragment_counts (List[int]): List of fragment counts corresponding to
            each substructure.

        Returns:
        List[Tuple[int, ...]]: Sorted list of substructures based on fragment
            counts.
        """
        paired_list = list(zip(substructures, fragment_counts))
        paired_list.sort(key=lambda x: x[1])
        return [pair[0] for pair in paired_list]

    def identify_optimal_substructure(
        self, parent_mol: Mol, child_mol: Mol, maxNodes: int = 200
    ) -> Tuple[int, ...]:
        """
        Identifies the most relevant substructure within a parent molecule
        given a child molecule, with a timeout feature for the
        substructure matching process. If the primary matching process times out,
        a fallback search is attempted with a maximum of one match.

        Parameters:
        parent_mol (Mol): The parent molecule.
        child_mol (Mol): The child molecule whose substructures are to be analyzed.
        timeout_sec (int): Timeout in seconds for the substructure search process.

        Returns:
        Tuple[int, ...]: The atom indices of the identified substructure
        in the parent molecule.

        Returns:
        Tuple[int, ...]: The atom indices of the identified substructure in
            the parent molecule.
        """

        if child_mol.GetNumAtoms() <= maxNodes:
            substructures = parent_mol.GetSubstructMatches(child_mol)
        else:
            substructures = parent_mol.GetSubstructMatches(child_mol, maxMatches=1)

        if len(substructures) > 1:
            fragment_counts = [
                self.remove_substructure_atoms(parent_mol, substructure)
                for substructure in substructures
            ]
            sorted_substructures = self.sort_substructures_by_fragment_count(
                substructures, fragment_counts
            )
            return sorted_substructures[0]
        else:
            return substructures[0] if substructures else ()
