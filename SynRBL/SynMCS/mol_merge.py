import json
from importlib.resources import files
from rdkit import Chem
import SynRBL.SynMCS
from rdkit.Chem import rdmolops


class NoCompoundError(Exception):
    """
    Exception if no compound rule is found to expand a molecule.
    """
    def __init__(self, boundary_atom, neighbor_atom):
        """
        Exception when no compound rule is found to expand a molecule.

        Arguments:
            boundary_atom (str): Atom symbol of boundary atom.
            neighbor_atom (str): Atom symbol of boundary neighbor.        
        """
        super().__init__(("Could not identify second compound for merge. " + 
                         "(boundary atom: {}, neighbor atom: {})")
                         .format(boundary_atom, neighbor_atom))


class MergeError(Exception):
    """
    Exception if no merge rule is found to combine two molecules.
    """
    def __init__(self, boundary_atom1, boundary_atom2, mol1, mol2):
        """
        Exception when no merge rule is found to combine two molecules.

        Arguments:
            boundary_atom1 (str): Atom symbol of first boundary atom.
            boundary_atom2 (str): Atom symbol of second boundary atom.
            mol1 (rdkit.Chem.Mol): Part one of molecule to be merged.
            mol2 (rdkit.Chem.Mol): Part two of molecule to be merged.
        """
        super().__init__(("Not able to find merge rule for atoms '{}' and " + 
                          "'{}' of compounds '{}' and '{}'.")
                           .format(boundary_atom1.GetSymbol(), 
                                   boundary_atom2.GetSymbol(), 
                                   Chem.MolToSmiles(Chem.RemoveHs(mol1)), 
                                   Chem.MolToSmiles(Chem.RemoveHs(mol2))))


class SubstructureError(Exception):
    """
    Exception if the structure of bound configuration to be merged is invalid.
    """
    def __init__(self):
        """
        Exception if the structure of bound configuration to be merged is 
        invalid.
        """
        super().__init__('Substructure mismatch.')


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
        super().__init__(("Atom dict is invalid for molecule '{}'. " + 
                         "Expected atom '{}' at index {} but found '{}'.")
                        .format(smiles, expected, index, actual))

class Property:
    """
    Generic property for dynamic rule configuration.
    """
    def __init__(self, value: str | list[str], dtype=str, allow_none=False):
        """
        Generic property for dynamic rule configuration.

        Arguments:
            value (str | list[str]): Configuration for this property. This is 
                a list of acceptable and forbiden values. 
                e.g.: ['A', 'B'] -> check is true if value is A or B 
                      ['!C', '!D'] -> check is true if value is not C and not D 
            dtype (optional): Datatype of the property. Must implement 
                conversion from string as constructor (e.g.: int).
            allow_none (bool): Flag if a check with value None is valid or not.
        """
        self.__dtype = dtype
        self.allow_none = allow_none
        self.neg_values = []
        self.pos_values = []
        if value is not None:
            if not isinstance(value, list):
                value = [value]
            for i, s in enumerate(value):
                value[i] = str(s)
            for item in value:
                if not isinstance(item, str):
                    raise ValueError('value must be str or a list of strings.')
                if len(item) > 0 and item[0] == '!':
                    self.neg_values.append(dtype(item[1:]))
                else:
                    self.pos_values.append(dtype(item))

    def check(self, value):
        """
        Check if the property is true for the given value.

        Arguments:
            value: The value to check. It must be of the same datatype as 
                specified in the constructor.

        Returns:
            bool: True if the value fulfills the property, false otherwise.
        """
        if self.allow_none and value is None:
            return True
        if not isinstance(value, self.__dtype):
            raise ValueError("value must be of type '{}'.".format(
                self.__dtype))
        if len(self.pos_values) > 0:
            found = False
            for pv in self.pos_values: 
                if pv == value:
                    found = True
                    break
            if not found:
                return False
        if len(self.neg_values) > 0:
            for nv in self.neg_values:
                if nv == value:
                    return False
        return True


class AtomCondition:
    """
    Atom condition class to check if a rule is applicable to a specific 
    molecule. Property configs (atom, rad_e, ...) can be prefixed with '!' 
    to negate the check. See SynRBL.SynMCS.mol_merge.Property for more 
    information.

    Example:
        Check if atom is Carbon and has Oxygen or Nitrogen as neighbor. 
        >>> cond = AtomCondition(atom=['C'], neighbors=['O', 'N'])
        >>> mol = rdkit.Chem.MolFromSmiles('CO')
        >>> cond.check(mol.GetAtomFromIdx(0), neighbor='O')
        True

    Attributes:
        atom (SynRBL.SynMCS.mol_merge.Property): Atom property 
        rad_e (SynRBL.SynMCS.mol_merge.Property): Radical electron property
        charge (SynRBL.SynMCS.mol_merge.Property): Charge porperty
        neighbors (SynRBL.SynMCS.mol_merge.Property): Neighbors property
    """
    def __init__(self, atom=None, rad_e=None, charge=None, neighbors=None, 
                 **kwargs):
        """
        Atom condition class to check if a rule is applicable to a specific 
        molecule. Property configs (atom, rad_e, ...) can be prefixed with '!' 
        to negate the check. See SynRBL.SynMCS.mol_merge.Property for more 
        information.

        Arguments:
            atom: Atom property configuration. 
            rad_e: Radical electron property configuration.
            charge: Charge porperty configuration.
            neighbors: Neighbors property configuration.
        """
        atom = kwargs.get('atom', atom)
        rad_e = kwargs.get('rad_e', rad_e)
        charge = kwargs.get('charge', charge)
        neighbors = kwargs.get('neighbors', neighbors)
        
        self.atom = Property(atom)
        self.rad_e = Property(rad_e, dtype=int)
        self.charge = Property(charge, dtype=int)
        self.neighbors = Property(neighbors, allow_none=True)

    def check(self, atom, neighbor=None):
        """
        Check if the atom meets the condition.

        Arguments:
            atom (rdkit.Chem.Atom): Atom the condition should be checked for.
            neighbor (str): A boundary atom.

        Returns:
            bool: True if the atom fulfills the condition, false otherwise.
        """
        return all([self.atom.check(atom.GetSymbol()),
                self.rad_e.check(atom.GetNumRadicalElectrons()),
                self.charge.check(atom.GetFormalCharge()),
                self.neighbors.check(neighbor)])


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
        r = self.cond.check(atom, neighbor=neighbor)
        return r

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
    if len(mols) == 0:
        return
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

def _check_atoms(mol, atom_dict):
    if isinstance(atom_dict, list):
        for e in atom_dict:
            _check_atoms(mol, e)
    elif isinstance(atom_dict, dict):
        sym, idx = next(iter(atom_dict.items()))
        actual_sym = mol.GetAtomWithIdx(idx).GetSymbol()
        if actual_sym != sym:
            raise InvalidAtomDict(sym, actual_sym, idx, Chem.MolToSmiles(mol))
    else:
        raise ValueError('atom_dict must be either a list or a dict.')
            

def _ad2t(atom_dict):
    """ 
    Convert atom dict to symbol and index tuple. 

    Args:
        atom_dict (dict): Atom dictionary in the for of {<symbol>: <index>}

    Returns:
        (str, int): Atom dictionary as tuple (<symbol>, <index>}
    """
    if not isinstance(atom_dict, dict) or len(atom_dict) != 1:
        raise ValueError('atom_dict must be of type {<symbol>: <index>}')
    return next(iter(atom_dict.items()))

def _adl2t(atom_dict_list):
    """ Split atom dict into symbol and indices lists. """
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
            _check_atoms(m, b)
            _, indices = _adl2t(b)
            nsyms, _ = _adl2t(n)
            merged_mol = merge_expand(m, indices, nsyms)
            merged_mols.append(merged_mol)
    elif len(mols) == 2:
        mol1, mol2 = mols[0], mols[1]
        _check_atoms(mol1, bounds[0])
        _check_atoms(mol2, bounds[1])
        if len(bounds[0]) != 1 or len(bounds[1]) != 1:
            raise SubstructureError()
        _, idx1 = _ad2t(bounds[0][0])
        _, idx2 = _ad2t(bounds[1][0])
        merged_mol = merge_mols(mol1, mol2, idx1, idx2)
        merged_mols.append(merged_mol)
    elif len(mols) > 2:
        raise NotImplementedError('Merging of {} molecules is not supported.'.format(len(mols)))
    for m in merged_mols:
        if 'rule' in m.keys(): del m['rule']
        if 'aam1' in m.keys(): del m['aam1']
        if 'aam2' in m.keys(): del m['aam2']
    return merged_mols
