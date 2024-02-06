import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdchem as rdchem
import rdkit.Chem.rdmolops as rdmolops


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


def is_carbon_balanced(reaction_smiles):
    def _cnt_C(mol):
        return len([a for a in mol.GetAtoms() if a.GetSymbol() == "C"])

    smiles_token = reaction_smiles.split(">>")
    reactant_smiles = smiles_token[0]
    product_smiles = smiles_token[1]
    reactant = rdmolfiles.MolFromSmiles(reactant_smiles)
    product = rdmolfiles.MolFromSmiles(product_smiles)
    reactant_Cs = _cnt_C(reactant)
    product_Cs = _cnt_C(product)
    return reactant_Cs == product_Cs


class InvalidAtomDict(Exception):
    """
    Exception if the atom dictionary is invalid.
    """

    def __init__(self, expected, actual, index, smiles):
        """
        Exception if the atom dictionary is invalid.

        Arguments:
            expected (str): The expected atom at the given index.
            actual (str): The actual atom at the index.
            index (int): The atom index in the molecule.
            smiles (str): The SMILES representation of the molecule.
        """
        super().__init__(
            (
                "Atom dict is invalid for molecule '{}'. "
                + "Expected atom '{}' at index {} but found '{}'."
            ).format(smiles, expected, index, actual)
        )


def check_atom_dict(mol, atom_dict):
    """
    Check if the atom dict matches the actual molecule. If the atom dictionary
    is not valid a InvalidAtomDict exception is raised.

    Arguments:
        mol (rdkit.Chem.Mol): The molecule on which the atom dictionary is
            checked.
        atom_dict (dict, list[dict]): The atom dictionary or a list of atom
            dictionaries to check on the molecule.
    """
    if isinstance(atom_dict, list):
        for e in atom_dict:
            check_atom_dict(mol, e)
    elif isinstance(atom_dict, dict):
        sym, idx = next(iter(atom_dict.items()))
        actual_sym = mol.GetAtomWithIdx(idx).GetSymbol()
        if actual_sym != sym:
            raise InvalidAtomDict(sym, actual_sym, idx, rdmolfiles.MolToSmiles(mol))
    else:
        raise ValueError("atom_dict must be either a list or a dict.")


def merge_two_mols(mol1, mol2, idx1, idx2, bond_type, mol1_track=None, mol2_track=None):
    """
    Merge two molecules. How the molecules are merge is defined by merge rules.

    Arguments:
        mol1 (rdkit.Chem.Mol): First molecule to merge.
        mol2 (rdkit.Chem.Mol): Second molecule to merge.
        idx1 (int): Atom index in mol1 where the new bond is formed.
        idx2 (int): Atom index in mol2 where the new bond is formed.
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

    if bond_type is not None:
        mol.AddBond(atom1.GetIdx(), atom2.GetIdx(), order=bond_type)

    return {
        "mol": mol,
        "aam1": mol1_tracker.to_dict(),
        "aam2": mol2_tracker.to_dict(),
    }
