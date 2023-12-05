import numpy as np
from rdkit import Chem


def _plot_mols(mols, includeAtomNumbers=False, titles=None):
    if type(mols) is not list:
        mols = [mols]
    fig, ax = plt.subplots(1, len(mols))
    for i, mol in enumerate(mols):
        a = ax
        if len(mols) > 1:
            a = ax[i]
        if includeAtomNumbers:
            for atom in mol.GetAtoms():
                atom.SetProp('atomLabel', str(atom.GetIdx()))
        mol_img = Draw.MolToImage(mol)
        if titles is not None and i < len(titles):
            a.set_title(titles[i])
        a.axis('off')
        a.imshow(mol_img)


def _remove_Hs(mol, idx, n):
    cnt = 0
    for n_atom in mol.GetAtoms()[idx].GetNeighbors():
        if n_atom.GetAtomicNum() == 1:
            mol.RemoveAtom(n_atom.GetIdx())
            cnt += 1
            if cnt == n:
                return
    raise RuntimeError(f"Could not remove {n} neighboring H atoms.")


def _count_H(atom):
    return len([a for a in atom.GetNeighbors() if a.GetAtomicNum() == 1])


def _validate_bond_type(min_Hs, bond_type):
    if bond_type is None and min_Hs > 1:
        print(("[WARNING] Merge molecules with single bond where " + 
               "double bond would be possible"))
    bond_type = 'single' if bond_type is None else bond_type
    if min_Hs == 1 and bond_type not in ['single']:
        raise ValueError(f"Invalid bond type '{bond_type}'")
    elif min_Hs == 2 and bond_type not in ['single', 'double']:
        raise ValueError(f"Invalid bond type '{bond_type}'")
    elif min_Hs == 3 and bond_type not in ['single', 'double', 'triple']:
        raise ValueError(f"Invalid bond type '{bond_type}'")
    elif min_Hs == 0 or min_Hs > 3:
        raise ValueError("Invalid number of H atoms.")

    if bond_type == 'single':
        return (1, Chem.rdchem.BondType.SINGLE)
    elif bond_type == 'double':
        return (2, Chem.rdchem.BondType.DOUBLE)
    elif bond_type == 'triple':
        return (3, Chem.rdchem.BondType.TRIPLE)
    else:
        raise ValueError(f"Invalid bond type '{bond_type}'")


def _rule__append_O_to_C_nextto_O_or_N(mol, idx):
    atom = mol.GetAtoms()[idx]
    if atom.GetSymbol() != 'C':
        return None
    for atom in mol.GetAtoms()[idx].GetNeighbors():
        if atom.GetSymbol() in ['O', 'N']:
            return {'mol': Chem.MolFromSmiles('O'), 'bound': {'O': 0}}
    return None


_compound_rules = [_rule__append_O_to_C_nextto_O_or_N]


def _try_get_compound(mol, idx):
    for rule in _compound_rules:
        mol2 = rule(mol, idx)
        if mol2 is not None:
            return mol2
    raise RuntimeError("Not able to identify second compound for merge.")


def merge_two_mols(mol1, mol2, atom_idx1, atom_idx2, bond_type=None):
    """ 
    Merge two molecules at the given atom indices. 
    This operation neglacts the byproduct in the form of bond_type * 2H 
    in the returned molecule.

    Args:
        mol1 (rdkit.Chem.rdchem.Mol): First molecule to merge
        mol2 (rdkit.Chem.rdchem.Mol): Second molecule to merge
        atom_idx1 (int): Merge index in first molecule.
        atom_idx2 (int): Merge index in second molecule.
        bond_type (str): Bond type used to merge the molecules.
            If not set a single bond is used. 
            Possible Values: [None, 'single', 'double', 'triple']

    Returns:
       rdkit.Chem.rdchem.Mol: Merged molecule.
    """

    mol1 = Chem.AddHs(mol1)
    mol2 = Chem.AddHs(mol2)
    mol1_atoms = mol1.GetAtoms()
    mol2_atoms = mol2.GetAtoms()
    min_Hs = np.min([
        _count_H(mol1_atoms[atom_idx1]), 
        _count_H(mol2_atoms[atom_idx2])
    ])
    bond_nr, bond_type = _validate_bond_type(min_Hs, bond_type)
    idx1, idx2 = (atom_idx1, len(mol1_atoms) + atom_idx2 - bond_nr)
    mol = Chem.RWMol(Chem.CombineMols(mol1, mol2))
    _remove_Hs(mol, idx1, bond_nr)
    _remove_Hs(mol, idx2, bond_nr)
    mol.AddBond(idx1, idx2, order=bond_type) 
    Chem.SanitizeMol(mol)
    return mol 


def merge_expand(mol, bounds):
    _mmol = mol
    for idx in bounds:
        mol2 = _try_get_compound(mol, idx)
        _mmol = merge_two_mols(_mmol, mol2['mol'], idx, list(mol2['bound'].values())[0])
    return _mmol
    

if __name__ == "__main__":
    # Used for testing
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw 

    plot = True
   
    def _test_merge_two_mols(mol1, mol2, idx1, idx2, result):
        _mol1 = Chem.MolFromSmiles(mol1)
        _mol2 = Chem.MolFromSmiles(mol2)
        _mmol = Chem.RemoveHs(merge_two_mols(_mol1, _mol2, idx1, idx2))
        is_correct = Chem.MolToSmiles(_mmol) == Chem.CanonSmiles(result)
        if plot:
            _plot_mols([_mol1, _mol2, _mmol], titles=['input1', 'input2', 'merged'], 
                       includeAtomNumbers=False)
            plt.show()
        if not is_correct:
            _rmol = Chem.MolFromSmiles(result)
            _plot_mols([_rmol, _mmol], titles=['expected', 'actual'], 
                       includeAtomNumbers=False)
            plt.show()
        assert is_correct, (
                f"Expected: {Chem.MolToSmiles(_mmol)} " + 
                f"Actual: {Chem.CanonSmiles(result)}")
    
    def _test_merge_expand(mol, bounds, result):
        _mol = Chem.MolFromSmiles(mol)
        _mmol = Chem.RemoveHs(merge_expand(_mol, bounds))
        is_correct = Chem.MolToSmiles(_mmol) == Chem.CanonSmiles(result)
        if plot:
            _plot_mols([_mol, _mmol], titles=['input', 'expaneded'], 
                       includeAtomNumbers=False)
            plt.show()
        if not is_correct:
            _rmol = Chem.MolFromSmiles(result)
            _plot_mols([_rmol, _mmol], titles=['expected', 'actual'], 
                       includeAtomNumbers=False)
            plt.show()
        assert is_correct, (
                f"Expected: {Chem.MolToSmiles(_mmol)} " + 
                f"Actual: {Chem.CanonSmiles(result)}")

    _test_merge_two_mols('CC1(C)OBOC1(C)C', 'CCC', 4, 1, 
                         'CC1(C)OB(-C(C)C)OC1(C)C')
    _test_merge_expand('O=COCc1ccccc1', [1], 'O=C(O)OCc1ccccc1')
    _test_merge_expand('O=Cc1ccccc1C=O', [1, 8], 'O=C(O)c1ccccc1C(O)=O')
