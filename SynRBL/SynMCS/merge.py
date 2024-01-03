import rdkit.Chem.rdchem as rdchem
import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.rdmolfiles as rdmolfiles

from .rules import MergeRule, CompoundRule
from .structure import Boundary, Compound

class NoCompoundRule(Exception):
    def __str__(self):
        return "No compound rule found." 

class AtomTracker:
    """
    A class to track atoms through the merge process.
    """

    def __init__(self, indices):
        """
        A class to track atoms through the merge process. After instantiation
        call the add_atoms method to initialize the tracker with the atom
        objects.

        Arguments:
            indices (list[int]): A list of atom indices to track.
        """
        self.__track_dict = {}
        if indices is not None:
            for idx in indices:
                self.__track_dict[str(idx)] = {}

    def add_atoms(self, mol, offset=0):
        """
        Add atom objects to the tracker. This is a necessary initialization
        step.

        Arguments:
            mol (rdkit.Chem.Mol): The molecule in which to track atoms.
            offset (int, optional): The atom index offset.
        """
        atoms = mol.GetAtoms()
        for k in self.__track_dict.keys():
            self.__track_dict[k]["atom"] = atoms[int(k) + offset]

    def to_dict(self):
        """
        Convert the tracker into a mapping dictionary.

        Returns:
            dict: A dictionary where keys are the old indices and the values
                represent the atom indices in the new molecule.
        """
        sealed_dict = {}
        for k in self.__track_dict.keys():
            sealed_dict[k] = self.__track_dict[k]["atom"].GetIdx()
        return sealed_dict


def merge_two_mols(mol1, mol2, idx1, idx2, rule, mol1_track=None, mol2_track=None):
    """
    Merge two molecules. How the molecules are merge is defined by merge rules.

    Arguments:
        mol1 (rdkit.Chem.Mol): First molecule to merge.
        mol2 (rdkit.Chem.Mol): Second molecule to merge.
        idx1 (int): Atom index in mol1 where the new bond is formed.
        idx2 (int): Atom index in mol2 where the new bond is formed.
        rule (TODO): The merge rule to apply.
        mol1_track (list[int], optional): A list of atom indices in mol1 that
            should be tracked during merging. The index mapping is part of the
            result with key 'aam1'.
        mol2_track (list[int], optional): A list of atom indices in mol2 that
            should be tracked during merging. The index mapping is part of the
            result with key 'aam2'.

    Returns:
        dict: A dictionary with the merged molecule at key 'mol' and optional
            atom index mappings at 'aam1' and 'aam2'.
    """
    mol1_tracker = AtomTracker(mol1_track)
    mol2_tracker = AtomTracker(mol2_track)

    mol = rdchem.RWMol(rdmolops.CombineMols(mol1, mol2))

    mol2_offset = len(mol1.GetAtoms())
    mol1_tracker.add_atoms(mol)
    mol2_tracker.add_atoms(mol, offset=mol2_offset)

    atom1 = mol.GetAtoms()[idx1]
    atom2 = mol.GetAtoms()[mol2_offset + idx2]

    rule.apply(mol, atom1, atom2)

    rdmolops.SanitizeMol(mol)

    return {
        "mol": mol,
        "aam1": mol1_tracker.to_dict(),
        "aam2": mol2_tracker.to_dict(),
    }


def expand_boundary(boundary: Boundary) -> Compound:
    compound = None
    for rule in CompoundRule.get_all():
        if rule.can_apply(boundary):
            compound = rule.apply()
            compound.rules.append(rule)
            break
    if compound is None:
        raise NoCompoundRule()
    return compound


def merge_boundaries(boundary1: Boundary, boundary2: Boundary) -> Compound | None:
    for rule in MergeRule.get_all():
        if not rule.can_apply(boundary1, boundary2):
            continue
        result = merge_two_mols(
            boundary1.compound.mol,
            boundary2.compound.mol,
            boundary1.index,
            boundary2.index,
            rule,
        )
        boundary1.compound.update(result["mol"], boundary1)
        boundary1.compound.rules.extend(boundary2.compound.rules)
        boundary1.compound.rules.append(rule)
        return boundary1.compound
    return None


def _merge_one_compound(compound: Compound) -> Compound:
    merged_compound = compound
    while len(merged_compound.boundaries) > 0:
        boundary1 = merged_compound.boundaries[0]
        compound2 = expand_boundary(boundary1)
        if len(compound2.boundaries) != 1:
            raise NotImplementedError(
                "Compound expansion and merge is only supported for "
                + "compounds with a single boundary atom."
            )
        merged_compound = merge_boundaries(boundary1, compound2.boundaries[0])
        if merged_compound is None:
            raise ValueError("No merge rule found.")
    return merged_compound


def _merge_two_compounds(compound1: Compound, compound2: Compound) -> Compound:
    boundaries1 = compound1.boundaries
    boundaries2 = compound2.boundaries
    if len(boundaries1) != 1:
        raise NotImplementedError("Can only merge compounds with single boundary atom.")
    if len(boundaries1) != len(boundaries2):
        raise NotImplementedError(
            "Can only merge compounds with the same number of boundaries."
        )
    merged_compound = merge_boundaries(boundaries1[0], boundaries2[0])
    if merged_compound is None:
        raise ValueError("No merge rule found.")
    return merged_compound


def merge(compounds: Compound | list[Compound]) -> Compound:
    if isinstance(compounds, Compound):
        compounds = list([compounds])
    if len(compounds) == 1:
        return _merge_one_compound(compounds[0])
    elif len(compounds) == 2:
        return _merge_two_compounds(compounds[0], compounds[1])
    else:
        raise NotImplementedError()
