import json
from importlib.resources import files
from rdkit import Chem
import SynRBL.SynMCS
from rdkit.Chem import rdmolops

class NoCompoundError(Exception):
    def __init__(self, boundary_atom, nearest_neighbor):
        self.boundary_atom = boundary_atom
        self.nearest_neighbor = nearest_neighbor

        super().__init__("Could not identify second compound for merge. " + 
                         "(Boundary Atom: {}, Nearest Neighbor: {})".format(
                             self.boundary_atom, self.nearest_neighbor))

class MergeError(Exception):
    def __init__(self, boundary_atom1, boundary_atom2, mol1, mol2):
        super().__init__(("Not able to find merge rule for atoms '{}' and '{}'" + 
                           " of compounds '{}' and '{}'.")
                           .format(boundary_atom1.GetSymbol(), 
                                   boundary_atom2.GetSymbol(), 
                                   Chem.MolToSmiles(Chem.RemoveHs(mol1)), 
                                   Chem.MolToSmiles(Chem.RemoveHs(mol2))))

class SubstructureError(Exception):
    def __init__(self):
        super().__init__('Substructure mismatch.')


class AtomCondition:
    def __init__(self, atom=None, rad_e=None, charge=None, neighbors=None, **kwargs):
        atom = kwargs.get('atom', atom)
        rad_e = kwargs.get('rad_e', rad_e)
        charge = kwargs.get('charge', charge)
        neighbors = kwargs.get('neighbors', neighbors)
        atom = atom if atom is None or isinstance(atom, list) else [atom]
        self.__rad_e = rad_e if rad_e is None or isinstance(rad_e, list) else [rad_e]
        self.__charge = charge if charge is None or isinstance(charge, list) else [charge]
        self.__neighbors = neighbors

        self.__atoms = None
        self.__neg_atoms = None
        if atom is not None:
            for a in atom:
                if '!' in a:
                    if self.__neg_atoms is None:
                        self.__neg_atoms = []
                    self.__neg_atoms.append(a.replace('!', ''))
                else:
                    if self.__atoms is None:
                        self.__atoms = []
                    self.__atoms.append(a)

    def check(self, atom, neighbor=None):
        if self.__atoms is not None:
            found = False
            for a in self.__atoms:
                if a == atom.GetSymbol():
                    found = True
                    break
            if not found:
                return False
        if self.__neg_atoms is not None:
            for a in self.__neg_atoms:
                if a == atom.GetSymbol():
                    return False
        if self.__rad_e is not None:
            if atom.GetNumRadicalElectrons() not in self.__rad_e:
                return False
        if self.__charge is not None:
            if atom.GetFormalCharge() not in self.__charge:
                return False
        if self.__neighbors is not None:
            if neighbor is None:
                return False
            valid_neighbor_set = False
            for neighbor_set in self.__neighbors:
                found_all = True
                for n_atom in neighbor_set:
                    if neighbor is None or n_atom != neighbor:
                        found_all = False
                        break
                if found_all:
                    valid_neighbor_set = True
                    break
            if not valid_neighbor_set:
                return False
        return True

class Action:
    def __init__(self, action=None):
        self.__action = action if action is None or isinstance(action, list) else [action]
    
    @staticmethod
    def apply(action_name, mol, atom):
        #print('Apply action:', action_name)
        if action_name == 'removeH':
            found_H = False
            for n_atom in atom.GetNeighbors():
                if n_atom.GetAtomicNum() == 1:
                    mol.RemoveAtom(n_atom.GetIdx())
                    found_H = True
                    break
            if not found_H:
                raise RuntimeError("Could not remove any more neighboring H atoms from {}.".format(atom.GetSymbol()))
        elif action_name == 'removeRadE':
            atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() - 1) 
        elif action_name == 'addRadE':
            atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + 1) 
        elif action_name == 'chargeNeg':
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        elif action_name == 'chargePos':
            atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        else:
            raise NotImplementedError(f"Action '{action_name}' is not implemented.")
        
    def __call__(self, mol, atom):
        if self.__action is None:
            return
        for a in self.__action:
            Action.apply(a, mol, atom)

class MergeRule:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'unnamed')
        self.cond1 = AtomCondition(**kwargs.get('condition1', {}))
        self.cond2 = AtomCondition(**kwargs.get('condition2', {}))
        self.action1 = Action(kwargs.get('action1', []))
        self.action2 = Action(kwargs.get('action2', []))
        self.bond = kwargs.get('bond', None)
        self.sym = kwargs.get('sym', True)
    
    @staticmethod
    def get_bond_type(bond):
        if bond is None:
            return None
        elif bond == 'single':
            return Chem.rdchem.BondType.SINGLE
        elif bond == 'double':
            return Chem.rdchem.BondType.DOUBLE
        else:
            raise NotImplementedError(f"Bond type '{bond}' is not implemented.")

    def __can_apply(self, atom1, atom2):
        return self.cond1.check(atom1) and self.cond2.check(atom2)

    def __apply(self, mol, atom1, atom2):
        if not self.__can_apply(atom1, atom2):
            raise ValueError('Can not apply merge rule.')
        #print("Apply merge rule '{}'.".format(self.name))
        self.action1(mol, atom1)
        self.action2(mol, atom2)
        bond_type = MergeRule.get_bond_type(self.bond)
        if bond_type is not None:
            mol.AddBond(atom1.GetIdx(), atom2.GetIdx(), order=bond_type) 
        return mol

    def can_apply(self, atom1, atom2):
        if self.sym:
            return self.__can_apply(atom1, atom2) or self.__can_apply(atom2, atom1)
        else:
            return self.__can_apply(atom1, atom2)

    def apply(self, mol, atom1, atom2):
        if self.sym:
            if self.__can_apply(atom1, atom2):
                return self.__apply(mol, atom1, atom2)
            else:
                return self.__apply(mol, atom2, atom1)
        else:
            return self.__apply(mol, atom1, atom2)

class AtomTracker:
    def __init__(self, ids):
        self.__track_dict = {}
        if ids is not None:
            for id in ids:
                self.__track_dict[str(id)] = {}

    def add_atoms(self, mol, offset=0):
        atoms = mol.GetAtoms()
        for k in self.__track_dict.keys():
            self.__track_dict[k]['atom'] = atoms[int(k) + offset]

    def to_dict(self):
        sealed_dict = {}
        for k in self.__track_dict.keys():
            sealed_dict[k] = self.__track_dict[k]['atom'].GetIdx()
        return sealed_dict

class CompoundRule:
    def __init__(self, **kwargs):
        self.name = kwargs.get('name', 'unnamed')
        self.cond = AtomCondition(**kwargs.get('condition', {}))
        self.compound = kwargs.get('compound', None)
    
    def can_apply(self, atom, neighbor):
        return self.cond.check(atom, neighbor=neighbor)

    def apply(self, atom, neighbor):
        if not self.can_apply(atom, neighbor):
            raise ValueError('Can not apply compound rule.')
        result = None
        if self.compound is not None and all(k in self.compound.keys() for k in ('smiles', 'index')):
            result = {'mol': Chem.MolFromSmiles(self.compound['smiles']), 
                    'index': self.compound['index']}
        #print("Apply compound rule '{}'.".format(self.name))
        return result

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

_merge_rules = None
_compound_rules = None

def get_merge_rules() -> list[MergeRule]:
    global _merge_rules
    if _merge_rules is None:
        json_data = files(SynRBL.SynMCS).joinpath('merge_rules.json').read_text()
        _merge_rules = [MergeRule(**c) for c in json.loads(json_data)]
    return _merge_rules

def get_compound_rules() -> list[CompoundRule]:
    global _compound_rules
    if _compound_rules is None:
        json_data = files(SynRBL.SynMCS).joinpath('compound_rules.json').read_text()
        _compound_rules = [CompoundRule(**c) for c in json.loads(json_data)]
    return _compound_rules

def get_compound(atom, neighbor):
    for rule in get_compound_rules():
        if rule.can_apply(atom, neighbor):
            return rule.apply(atom, neighbor), rule
    raise NoCompoundError(atom.GetSymbol(), neighbor)

def merge_mols(mol1, mol2, idx1, idx2, mol1_track=None, mol2_track=None):
    mol1_tracker = AtomTracker(mol1_track)
    mol2_tracker = AtomTracker(mol2_track)

    mol1 = Chem.AddHs(mol1)
    mol2 = Chem.AddHs(mol2)
    mol = Chem.RWMol(Chem.CombineMols(mol1, mol2))
    mol2_offset = len(mol1.GetAtoms())
    mol1_tracker.add_atoms(mol)
    mol2_tracker.add_atoms(mol, offset=mol2_offset)
    atom1 = mol.GetAtoms()[idx1]
    atom2 = mol.GetAtoms()[mol2_offset + idx2]
    merge_rule = None
    for rule in get_merge_rules():
        if not rule.can_apply(atom1, atom2):
            continue
        rule.apply(mol, atom1, atom2)
        merge_rule = rule
        break
    if merge_rule:
        Chem.SanitizeMol(mol)
        return {'mol': mol, 
                'rule': merge_rule,
                'aam1': mol1_tracker.to_dict(), 
                'aam2': mol2_tracker.to_dict()}
    else:
        raise MergeError(atom1, atom2, mol1, mol2)

def merge_expand(mol, bound_indices, neighbors=None):
    if not isinstance(bound_indices, list):
        raise ValueError('bound_indices must be of type list')
    l = len(bound_indices)
    if neighbors is None:
        neighbors = [None for _ in range(l)]
    if len(neighbors) != l:
        raise ValueError('neighbors list must be of same length as bound_indices. ' + 
                         '(bound_indices={}, neighbors={})'.format(
                             bound_indices, neighbors))
    mol1 = mol
    used_compound_rules = []
    used_merge_rules = []
    for i in range(l):
        atom = mol1.GetAtoms()[bound_indices[i]]
        mol2, rule = get_compound(atom, neighbors[i])
        used_compound_rules.append(rule)
        if mol2 is not None:
            mol = merge_mols(mol1, mol2['mol'], 
                             bound_indices[i], mol2['index'], 
                             mol1_track=bound_indices)
            bound_indices = [mol['aam1'][str(idx)] for idx in bound_indices]
            mol1 = mol['mol']
            used_merge_rules.append(mol['rule'])
        else:
            mol = {'mol': mol1}
    mol['compound_rules'] = used_compound_rules
    mol['merge_rules'] = used_merge_rules
    return mol

def _ad2t(atom_dict):
    """ Atom dict to tuple. """
    assert isinstance(atom_dict, dict) and len(atom_dict) == 1
    return next(iter(atom_dict.items()))

def _adl2t(atom_dict_list):
    """ Atom dict list to symbol and indices lists tuple. """
    sym_list = []
    idx_list = []
    for a in atom_dict_list:
        sym, idx = _ad2t(a)
        sym_list.append(sym)
        idx_list.append(idx)
    return sym_list, idx_list

def _split_mol(mol, bounds, neighbors):
    assert isinstance(bounds, list) and len(bounds) > 0 and isinstance(bounds[0], dict)
    assert isinstance(neighbors, list) and len(neighbors) == len(bounds) and isinstance(neighbors[0], dict)
    frags = list(rdmolops.GetMolFrags(mol, asMols = True))
    offsets = [0]
    for i, f in enumerate(frags):
        offsets.append(offsets[i] + len(f.GetAtoms()))
    _bounds = [[] for _ in range(len(frags))]
    _neighbors = [[] for _ in range(len(frags))]
    for b, n in zip(bounds, neighbors):
        sym, idx = _ad2t(b)
        for i in range(len(offsets) - 1):
            if idx >= offsets[i] and idx < offsets[i + 1]:
                _bounds[i].append({sym: idx - offsets[i]})
                _neighbors[i].append(n)
    return frags, _bounds, _neighbors

def merge(mols, bounds, neighbors):
    merged_mols = []
    if len(mols) == 1:
        mols, bounds, neighbors = _split_mol(mols[0], bounds[0], neighbors[0])
        for m, b, n in zip(mols, bounds, neighbors):
            _, indices = _adl2t(b)
            nsyms, _ = _adl2t(n)
            merged_mol = merge_expand(m, indices, nsyms)
            merged_mols.append(merged_mol)
    elif len(mols) == 2:
        mol1, mol2 = mols[0], mols[1]
        if len(bounds[0]) != 1 or len(bounds[1]) != 1:
            raise SubstructureError()
        _, idx1 = _ad2t(bounds[0][0])
        _, idx2 = _ad2t(bounds[1][0])
        merged_mol = merge_mols(mol1, mol2, idx1, idx2)
        merged_mols.append(merged_mol)
    else:
        raise NotImplementedError('Merging of {} molecules is not supported.'.format(len(mols)))
    for m in merged_mols:
        if 'rule' in m.keys(): del m['rule']
        if 'aam1' in m.keys(): del m['aam1']
        if 'aam2' in m.keys(): del m['aam2']
    return merged_mols

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

    def _test_merge(frags, bounds, neighbors, result):
        mols = [Chem.MolFromSmiles(f) for f in frags]
        _mmol = [Chem.RemoveHs(m['mol']) for m in merge(mols, bounds, neighbors)]
        all_correct = True
        for m, r in zip(_mmol, result):
            is_correct = Chem.MolToSmiles(m) == Chem.CanonSmiles(r)
            all_correct = all_correct and is_correct
            if not is_correct:
                print(f"Expected: {Chem.CanonSmiles(r)} Actual: {Chem.MolToSmiles(m)}")
            if not is_correct:
                _rmol = Chem.MolFromSmiles(r)
                if plot:
                    plot_mols([_rmol, m], titles=['expected', 'actual'], 
                              includeAtomNumbers=True)
                    plt.show()
        if plot:
            plot_mols(_mmol, includeAtomNumbers=False)
            plt.show()
        assert all_correct, "Merge result is not as expected."

    if True:
        _test_merge_mols('CC1(C)OBOC1(C)C', 'CCC', 4, 1, 'CC1(C)OB(-C(C)C)OC1(C)C')
        _test_merge_mols('O', 'C[Si](C)C(C)(C)C', 0, 1, 'C[Si](O)(C)C(C)(C)C')
        _test_merge_mols('O', 'CC(C)[P+](c1ccccc1)c1ccccc1', 0, 3, 'CC(C)P(=O)(c1ccccc1)c1ccccc1')  
        _test_merge_mols('Cl', 'CCCC[Sn](CCCC)CCCC', 0, 4, 'CCCC[Sn+](CCCC)CCCC.[Cl-]')
        _test_merge_expand('O=COCc1ccccc1', [1], ['O'], 'O=C(O)OCc1ccccc1')
        _test_merge_expand('O=Cc1ccccc1C=O', [1, 8], ['O', 'O'], 'O=C(O)c1ccccc1C(O)=O')
        _test_merge_expand('C[Si](C)C(C)(C)C', [1], ['O'], 'C[Si](O)(C)C(C)(C)C')
        _test_merge_expand('CC(C)(C)OC(=O)O', [7], ['C'], 'CC(C)(C)OC(=O)O')

    _test_merge(['CC1(C)OBOC1(C)C', 'Br'], 
                [[{'B': 4}], [{'Br': 0}]], 
                [[{'C': 5}], [{'C': 13}]], 
                ['CC1(C)OB(Br)OC1(C)C'])
    try:
        _test_merge(['NBr.O'], [[{'N': 1}, {'O': 0}, {'N': 1}]], [[{'C': 1}, {'C': 4}, {'C': 4}]], ['NBr', 'O'])
    except NoCompoundError: 
        pass
    _test_merge(['C.O'], [[{'C': 0}, {'O': 1}]], [[{'O': 1}, {'C': 2}]], ['CO', 'O'])
    _test_merge(['O=Cc1ccccc1C=O'], [[{'C': 1}, {'C': 8}]], [[{'N': 9}, {'N': 11}]], ['O=C(O)c1ccccc1C(O)=O'])
    #plot_mols(Chem.MolFromSmiles('CCCC[Sn+](CCCC)CCCC.[Cl-]'), includeAtomNumbers=False)
    #plt.show()



