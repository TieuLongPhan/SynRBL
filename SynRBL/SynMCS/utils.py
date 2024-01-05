import rdkit.Chem.rdmolfiles as rdmolfiles


def carbon_equality_check(reaction_smiles):
    def _cnt_C(mol):
        return len([a for a in mol.GetAtoms() if a.GetSymbol() == "C"])

    smiles_token = reaction_smiles.split(">>")
    reactant_smiles = smiles_token[0]
    product_smiles = smiles_token[1]
    reactant = rdmolfiles.MolFromSmiles(reactant_smiles)
    product = rdmolfiles.MolFromSmiles(product_smiles)
    reactant_Cs = _cnt_C(reactant)
    product_Cs = _cnt_C(product)
    if reactant_Cs != product_Cs:
        raise RuntimeError(
            (
                "Failed to impute the correct structure. "
                + "Carbon atom count in reactants and products does not match. "
                + "(Reactants={}, Products={})"
            ).format(reactant_Cs, product_Cs)
        )


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
