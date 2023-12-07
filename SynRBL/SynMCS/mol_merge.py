import numpy as np
from rdkit import Chem

class NoCompoundError(Exception):
    def __init__(self, boundary_atom, nearest_neighbor):
        self.boundary_atom = boundary_atom
        self.nearest_neighbor = nearest_neighbor

        super().__init__("Could not identify second compound for merge. " + 
                         "(Boundary Atom: {}, Nearest Neighbor: {})".format(
                             self.boundary_atom, self.nearest_neighbor))


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
    if cnt != n:
        raise RuntimeError("Could not remove {} neighboring H atoms from {}.".format(n, atom.GetSymbol()))

def _count_Hs(atom):
    return len([n for n in atom.GetNeighbors() if n.GetSymbol() == 'H'])

_append_compound_rules = [
        {'condition': {'atom': ['!O'], 'neighbors': [['O'], ['N']]}, 
         'result': {'smiles': 'O', 'index': 0}}, 
        {'condition': {'atom': ['Si']}, 'result': {'smiles': 'O', 'index': 0}},
        {'condition': {'atom': ['C'], 'neighbors': [['C']]},
         'result': {'smiles': 'O', 'index': 0}}]

def _apply_append_compound_rule(rule, mol, idx, neighbors):
    assert isinstance(rule, dict)
    cond = rule.get('condition')
    result = rule['result']
    if cond is not None:
        cond_atom = cond.get('atom')
        if cond_atom is not None and mol.GetAtoms()[idx].GetSymbol() not in cond_atom:
            return None
        cond_neighbors = cond.get('neighbors')
        if cond_neighbors is not None:
            valid_neighbor_set = False
            for neighbor_set in cond_neighbors:
                found_all = True
                for n_atom in neighbor_set:
                    if neighbors is None or n_atom not in neighbors:
                        found_all = False
                        break
                if found_all:
                    valid_neighbor_set = True
                    break
            if not valid_neighbor_set:
                return None
    return (Chem.MolFromSmiles(result['smiles']), result['index'])

_merge_rules = [
        {'condition1': {'atom': ['P'], 'rad_e': 1, 'charge': 1}, 
         'condition2': {'nr_Hs': [2, 3, 4], 'rad_e': 0, 'charge': 0},
         'action1': ['removeRadE', 'discharge'], 'action2': ['removeH', 'removeH'], 'bond': 'double'},
        {'condition1': {'atom': ['Si'], 'rad_e': 1}, 
         'condition2': {'nr_Hs': [1, 2, 3, 4], 'rad_e': 0}, 
         'action1': ['removeRadE'], 'action2': ['removeH'], 'bond': 'single'},
        {'condition1': {'atom': ['Sn', 'Zn', 'Cu', 'Fl', 'Mg'], 'rad_e': 1, 'charge': 0}, 
         'condition2': {'atom': ['F', 'Cl', 'Br', 'I', 'At', 'O', 'N'], 'nr_Hs': [1, 2, 3, 4], 'rad_e': 0, 'charge': 0},
         'action1': ['removeRadE', 'chargePos'], 'action2': ['removeH', 'chargeNeg']},
        {'condition1': {'nr_Hs': [1, 2, 3, 4], 'rad_e': 0}, 'condition2': {'nr_Hs': [1, 2, 3, 4], 'rad_e': 0}, 
         'action1': ['removeH'], 'action2': ['removeH'], 'bond': 'single'}]

def _check_atom_cond(condition_atoms, atom_sym):
    return False

def _apply_merge_rule(rule, mol, atom1, atom2):
    def _check_condition(cond, atom):
        if cond is None:
            return True
        cond_atoms = cond.get('atom')
        if cond_atoms is not None:
            if not _check_atom_cond(cond_atoms, atom.GetSymbol()):
                return False
        nr_Hs = cond.get('nr_Hs')
        if nr_Hs is not None and _count_Hs(atom) not in nr_Hs:
            return False
        rad_e = cond.get('rad_e')
        if rad_e is not None and atom.GetNumRadicalElectrons() != rad_e:
            return False
        return True
            
    def _apply_action(action, mol, atom):
        if action is None:
            return
        if action == 'removeH':
            _remove_Hs(mol, atom, 1)
        elif action == 'removeRadE':
            atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() - 1) 
        elif action == 'discharge':
            atom.SetFormalCharge(0)
        elif action == 'chargePos':
            atom.SetFormalCharge(1)
        elif action == 'chargeNeg':
            atom.SetFormalCharge(-1)
        else:
            raise NotImplementedError(f"Action '{action}' is not implemented.")

    def _get_bond_type(bond):
        if bond is None:
            return None
        elif bond == 'single':
            return Chem.rdchem.BondType.SINGLE
        elif bond == 'double':
            return Chem.rdchem.BondType.DOUBLE
        else:
            raise NotImplementedError(f"Bond type '{bond}' is not implemented.")

    cond1 = rule.get('condition1')
    cond2 = rule.get('condition2')
    action1 = rule.get('action1', [])
    action2 = rule.get('action2', [])
    bond_type = _get_bond_type(rule.get('bond'))
    if not _check_condition(cond1, atom1) or not _check_condition(cond2, atom2):
        return None
    for a in action1:
        _apply_action(a, mol, atom1)
    for a in action2:
        _apply_action(a, mol, atom2)
    if bond_type is not None:
        mol.AddBond(atom1.GetIdx(), atom2.GetIdx(), order=bond_type) 
    return mol

def _apply_merge_rule_sym(rule, mol, atom1, atom2):
    rule_result = _apply_merge_rule(rule, mol, atom1, atom2)
    if rule_result is None:
        #cond1 = rule.get('condition1')
        #cond2 = rule.get('condition2')
        #action1 = rule.get('action1', [])
        #action2 = rule.get('action2', [])
        #rule['condition1'] = cond2
        #rule['condition2'] = cond1
        #rule['action1'] = action2
        #rule['action2'] = action1
        rule_result = _apply_merge_rule(rule, mol, atom2, atom1)
    return rule_result

def _try_get_compound(mol, idx, neighbors):
    for rule in _append_compound_rules:
        mol2 = _apply_append_compound_rule(rule, mol, idx, neighbors)
        if mol2 is not None:
            return {'mol': mol2[0], 'bound_idx': mol2[1]}
    raise NoCompoundError(mol.GetAtoms()[idx].GetSymbol(), neighbors)

def merge_mols(mol1, mol2, idx1, idx2, mol1_track=None, mol2_track=None):
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
    found_merge_rule = False
    for rule in _merge_rules:
        rule_result = _apply_merge_rule_sym(rule, mol, atom1, atom2)
        if rule_result is not None:
            found_merge_rule = True
            break
    if found_merge_rule:
        Chem.SanitizeMol(mol)
        return {'mol': mol, 
                'aam1': _seal_track_dict(mol1_track_dict), 
                'aam2': _seal_track_dict(mol2_track_dict)}
    else:
        raise RuntimeError(("Not able to find merge rule for atoms '{}' and '{}'" + 
                           " of compounds '{}' and '{}'.").format(atom1.GetSymbol(), 
                                                                 atom2.GetSymbol(), 
                                                                 Chem.MolToSmiles(Chem.RemoveHs(mol1)), 
                                                                 Chem.MolToSmiles(Chem.RemoveHs(mol2))))

def merge_expand(mol, bound_indices, neighbors=None):
    if not isinstance(bound_indices, list):
        raise ValueError('bound_indices must be of type list')
    l = len(bound_indices)
    if neighbors is None:
        neighbors = [None for _ in range(l)]
    if len(neighbors) != l:
        raise ValueError('neighbors list must be of same length as bound_indices. ' + 
                         '(bound_indices={}, neighbors={})'.format(bound_indices, neighbors))
    mol1 = mol
    for i in range(l):
        mol2 = _try_get_compound(mol1, bound_indices[i], neighbors[i])
        mol = merge_mols(mol1, mol2['mol'], bound_indices[i], mol2['bound_idx'], mol1_track=bound_indices)
        bound_indices = [mol['aam1'][str(idx)] for idx in bound_indices]
        mol1 = mol['mol']
    return mol

if __name__ == "__main__":
    # Used for testing
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw 

    assert _check_atom_cond(['C'], 'C')
    assert _check_atom_cond(['O', 'C'], 'C')
    assert not _check_atom_cond(['!O'], 'C')
    assert _check_atom_cond(['!O', 'C'], 'C')
    assert not _check_atom_cond(['!O'], 'O')
    
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
        merge_result = merge_expand(_mol, bound_idx, neighbors)
        _mmol = Chem.RemoveHs(merge_result['mol'])
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

    if True:
        _test_merge_mols('CC1(C)OBOC1(C)C', 'CCC', 4, 1, 'CC1(C)OB(-C(C)C)OC1(C)C')
        _test_merge_mols('O', 'C[Si](C)C(C)(C)C', 0, 1, 'C[Si](O)(C)C(C)(C)C')
        _test_merge_mols('O', 'CC(C)[P+](c1ccccc1)c1ccccc1', 0, 3, 'CC(C)P(=O)(c1ccccc1)c1ccccc1')  
        _test_merge_expand('O=COCc1ccccc1', [1], [['O']], 'O=C(O)OCc1ccccc1')
        _test_merge_expand('O=Cc1ccccc1C=O', [1, 8], [['O'], ['O']], 'O=C(O)c1ccccc1C(O)=O')
        _test_merge_expand('C[Si](C)C(C)(C)C', [1], [['O']], 'C[Si](O)(C)C(C)(C)C')
        _test_merge_mols('Cl', 'CCCC[Sn](CCCC)CCCC', 0, 4, 'CCCC[Sn+](CCCC)CCCC.[Cl-]')
    #plot_mols(Chem.MolFromSmiles('CCCC[Sn+](CCCC)CCCC.[Cl-]'), includeAtomNumbers=False)
    #plt.show()
