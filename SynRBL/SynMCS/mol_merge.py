import numpy as np
from rdkit import Chem


def plot_mols(mols, includeAtomNumbers=False, titles=None, figsize=None):
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw

    if type(mols) is not list:
        mols = [mols]
    fig, ax = plt.subplots(1, len(mols), figsize=figsize)
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


def _remove_Hs(mol, atom, n):
    cnt = 0
    for n_atom in atom.GetNeighbors():
        if n_atom.GetAtomicNum() == 1:
            mol.RemoveAtom(n_atom.GetIdx())
            cnt += 1
            if cnt == n:
                break
    if cnt == n:
        print("Removed {} H atoms.".format(cnt))
    else:
        raise RuntimeError(f"Could not remove {n} neighboring H atoms.")

def _count_H(atom, include_radicals=False):
    cnt = len([a for a in atom.GetNeighbors() if a.GetAtomicNum() == 1])
    if include_radicals:
        cnt += atom.GetNumRadicalElectrons()
    return cnt 


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
        raise ValueError(f"Invalid number of H atoms ({min_Hs}).")

    if bond_type == 'single':
        return (1, Chem.rdchem.BondType.SINGLE)
    elif bond_type == 'double':
        return (2, Chem.rdchem.BondType.DOUBLE)
    elif bond_type == 'triple':
        return (3, Chem.rdchem.BondType.TRIPLE)
    else:
        raise ValueError(f"Invalid bond type '{bond_type}'")


def _rule__append_O_to_C_nextto_O_or_N(mol, idx, neighbors):
    atom = mol.GetAtoms()[idx]
    if atom.GetSymbol() != 'C':
        return None
    for n_atom in neighbors:
        if n_atom in ['O', 'N']:
            return {'mol': Chem.MolFromSmiles('O'), 'bound': {'O': 0}}
    return None

def _rule__append_O_to_Si(mol, idx, neighbors):
    atom = mol.GetAtoms()[idx]
    if atom.GetSymbol() != 'Si':
        return None
    return {'mol': Chem.MolFromSmiles('O'), 'bound': {'O': 0}}

_compound_rules = [
        _rule__append_O_to_C_nextto_O_or_N,
        _rule__append_O_to_Si]


def _try_get_compound(mol, idx, neighbors):
    for rule in _compound_rules:
        mol2 = rule(mol, idx, neighbors)
        if mol2 is not None:
            return mol2
    raise RuntimeError("Not able to identify second compound for merge.")


def merge_mols(mol1, mol2, idx1, idx2, mol1_track=None, mol2_track=None):
    bond_type = Chem.rdchem.BondType.SINGLE
    bond_nr = 1
      
    def _setup_track_dict(track_ids):
        track_dict = {}
        if track_ids is not None:
            for id in track_ids:
                track_dict[str(id)] = {}
        return track_dict

    def _add_track_atoms(track_dict, mol, offset=0):
        atoms = mol.GetAtoms()
        for k in track_dict.keys():
            track_dict[k]['atom'] = atoms[int(k) + offset]

    def _seal_track_dict(track_dict):
        sealed_dict = {}
        for k in track_dict.keys():
            sealed_dict[k] = track_dict[k]['atom'].GetIdx()
        return sealed_dict

    mol1_track_dict = _setup_track_dict(mol1_track)
    mol2_track_dict = _setup_track_dict(mol2_track)

    mol1 = Chem.AddHs(mol1)
    mol2 = Chem.AddHs(mol2)
    mol = Chem.RWMol(Chem.CombineMols(mol1, mol2))
    mol2_offset = len(mol1.GetAtoms())
    _add_track_atoms(mol1_track_dict, mol)
    _add_track_atoms(mol2_track_dict, mol, offset=mol2_offset)
    atom1 = mol.GetAtoms()[idx1]
    atom2 = mol.GetAtoms()[mol2_offset + idx2]
    _remove_Hs(mol, atom1, bond_nr - atom1.GetNumRadicalElectrons()) 
    _remove_Hs(mol, atom2, bond_nr - atom2.GetNumRadicalElectrons())
    mol.AddBond(atom1.GetIdx(), atom2.GetIdx(), order=bond_type) 
    Chem.SanitizeMol(mol)
    return {'mol': mol, 
            'aam1': _seal_track_dict(mol1_track_dict), 
            'aam2': _seal_track_dict(mol2_track_dict)}


def merge_expand(mol, bound_idx, neighbors=None):
    _mmol = mol
    _mol2 = _try_get_compound(mol, bound_idx, neighbors)
    _mmol = merge_two_mols(_mmol, _mol2['mol'], bound_idx, 
                           list(_mol2['bound'].values())[0])
    return _mmol

if __name__ == "__main__":
    # Used for testing
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw 

    plot = True
   
    def _test_merge_mols(mol1, mol2, idx1, idx2, result):
        _mol1 = Chem.MolFromSmiles(mol1)
        _mol2 = Chem.MolFromSmiles(mol2)
        merge_result = merge_mols(_mol1, _mol2, idx1, idx2, 
                                  mol1_track=[idx1], mol2_track=[idx2])
        _mmol = Chem.RemoveHs(merge_result['mol'])
        is_correct = Chem.MolToSmiles(_mmol) == Chem.CanonSmiles(result)
        if plot:
            plot_mols([_mol1, _mol2, _mmol], titles=['input1', 'input2', 'merged'], 
                      includeAtomNumbers=False)
            plt.show()
        if not is_correct:
            _rmol = Chem.MolFromSmiles(result)
            plot_mols([_rmol, _mmol], titles=['expected', 'actual'], 
                      includeAtomNumbers=False)
            plt.show()
        assert is_correct, (f"Expected: {Chem.MolToSmiles(_mmol)} " + 
                            f"Actual: {Chem.CanonSmiles(result)}")
    
    def _test_merge_expand(mol, bound_idx, neighbors, result):
        _mol = Chem.MolFromSmiles(mol)
        _mmol = Chem.RemoveHs(merge_expand(_mol, bound_idx, neighbors))
        is_correct = Chem.MolToSmiles(_mmol) == Chem.CanonSmiles(result)
        if plot:
            plot_mols([_mol, _mmol], titles=['input', 'expaneded'], 
                      includeAtomNumbers=False)
            plt.show()
        if not is_correct:
            _rmol = Chem.MolFromSmiles(result)
            plot_mols([_rmol, _mmol], titles=['expected', 'actual'], 
                      includeAtomNumbers=True)
            plt.show()
        assert is_correct, (f"Expected: {Chem.MolToSmiles(_mmol)} " + 
                            f"Actual: {Chem.CanonSmiles(result)}")

    _test_merge_mols('CC1(C)OBOC1(C)C', 'CCC', 4, 1, 'CC1(C)OB(-C(C)C)OC1(C)C')

    #_test_merge_expand('O=COCc1ccccc1', 1, ['O'], 'O=C(O)OCc1ccccc1')

    #_test_merge_expand('O=Cc1ccccc1C=O', 1, ['O'], 'O=C(O)c1ccccc1C=O')
    #_test_merge_expand('O=C(O)c1ccccc1C=O', [1, 8], [['O'], ['O']], 'O=C(O)c1ccccc1C(O)=O')

    #_test_merge_expand('C[Si](C)C(C)(C)C', 1, ['O'], 'C[Si](O)(C)C(C)(C)C')

