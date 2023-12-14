import json
from importlib.resources import files
from rdkit import Chem
import SynRBL.SynMCS
from rdkit.Chem import rdmolops
from SynRBL.SynMCS.rule_formation import Property


_merge_rules = None
_compound_rules = None


class NoMoreHsError(Exception):
    """
    Exception if an atom has no more Hydrogen atoms that can be removed.
    """

    def __init__(self, atom):
        """
        Exception if an atom has no more Hydrogen atoms that can be removed.
        """
        super().__init__(
            "Could not remove any more neighboring H atoms from {}.".format(atom)
        )


class NoCompoundRuleError(Exception):
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
        super().__init__(
            (
                "Could not identify second compound for merge. "
                + "(boundary atom: {}, neighbor atom: {})"
            ).format(boundary_atom, neighbor_atom)
        )


class NoMergeRuleError(Exception):
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
        super().__init__(
            (
                "Not able to find merge rule for atoms '{}' and "
                + "'{}' of compounds '{}' and '{}'."
            ).format(
                boundary_atom1.GetSymbol(),
                boundary_atom2.GetSymbol(),
                Chem.MolToSmiles(Chem.RemoveHs(mol1)),
                Chem.MolToSmiles(Chem.RemoveHs(mol2)),
            )
        )


class MergeError(Exception):
    """
    Exception if merge failed.
    """

    def __init__(self, rule_name, inner_msg=None):
        """
        Exception if merge failed.
        """

        msg = "Merge rule '{}' failed.".format(rule_name)
        if inner_msg is not None:
            msg += " {}".format(inner_msg)
        super().__init__(msg)


class SubstructureError(Exception):
    """
    Exception if the structure of bound configuration to be merged is invalid.
    """

    def __init__(self):
        """
        Exception if the structure of bound configuration to be merged is
        invalid.
        """
        super().__init__("Substructure mismatch.")


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

    def __init__(self, atom=None, rad_e=None, charge=None, neighbors=None, **kwargs):
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
        atom = kwargs.get("atom", atom)
        rad_e = kwargs.get("rad_e", rad_e)
        charge = kwargs.get("charge", charge)
        neighbors = kwargs.get("neighbors", neighbors)

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
        return all(
            [
                self.atom.check(atom.GetSymbol()),
                self.rad_e.check(atom.GetNumRadicalElectrons()),
                self.charge.check(atom.GetFormalCharge()),
                self.neighbors.check(neighbor),
            ]
        )


class Action:
    """
    Class to configure a set of actions to perform on an atom.
    """

    def __init__(self, action=None):
        self.__action = action
        if action is not None and not isinstance(action, list):
            self.__action = [action]

    @staticmethod
    def apply(action_name, mol, atom):
        """
        Apply an action to an atom.

        Arguments:
            action_name (str): The name of the action to apply.
            mol (rdkit.Chem.Mol): The molecule object where the action should
                be applied.
            atom (rdkit.Chem.Atom): The atom where the action should be
                applied.
        """
        if action_name == "removeH":
            found_H = False
            for n_atom in atom.GetNeighbors():
                if n_atom.GetAtomicNum() == 1:
                    mol.RemoveAtom(n_atom.GetIdx())
                    found_H = True
                    break
            if not found_H:
                raise NoMoreHsError(atom.GetSymbol())
        elif action_name == "removeRadE":
            atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() - 1)
        elif action_name == "addRadE":
            atom.SetNumRadicalElectrons(atom.GetNumRadicalElectrons() + 1)
        elif action_name == "chargeNeg":
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)
        elif action_name == "chargePos":
            atom.SetFormalCharge(atom.GetFormalCharge() + 1)
        else:
            raise NotImplementedError(
                ("Action '{}' is not implemented.").format(action_name)
            )

    def __call__(self, mol, atom):
        """
        Apply the configured actions to the atom in the molecule.

        Arguments:
            mol (rdkit.Chem.Mol): The molecule object where the action should
                be applied.
            atom (rdkit.Chem.Atom): The atom where the action should be
                applied.
        """
        if self.__action is None:
            return
        for a in self.__action:
            Action.apply(a, mol, atom)


class MergeRule:
    """
    Class for defining a merge rule between two compounds. If boundary atom1
    meets condition1 and boundary atom2 meets condition2 action1 is applied to
    atom1, action2 is applied to atom2 and a bond is formed between atom1 and
    atom2. A merge rule can be configured by providing a suitable dictionary.
    Examples on how to configure merge rules can be found in
    SynRBL/SynMCS/merge_rules.json file.

    Example:
        The following example shows a complete default configuration for a merge rule.
        config = {
            "name": "unnamed",
            "condition1": {
                "atom": None,
                "rad_e": None,
                "charge": None,
                "neighbors": None,
                "compound": None
            },
            "condition2": {
                "atom": None,
                "rad_e": None,
                "charge": None,
                "neighbors": None,
                "compound": None
            },
            "action1": [],
            "action2": [],
            "bond": None,
            "sym": True
        }

    Attributes:
        name (str, optional): A descriptive name for the rule. This attribute
            is just for readability and does not serve a functional purpose.
        condition1 (SynRBL.SynMCS.mol_merge.AtomCondition, optional): Condition
            for the first boundary atom.
        condition2 (SynRBL.SynMCS.mol_merge.AtomCondition, optional): Condition
            for the second boundary atom.
        action1 (SynRBL.SynMCS.mol_merge.Action, optional): Actions to performe
            on the first boundary atom.
        action2 (SynRBL.SynMCS.mol_merge.Action, optional): Actions to performe
            on the second boundary atom.
        bond (str, optional): The bond type to form between the two compounds.
        sym (bool): If the rule is symmetric. If set to True order of condition
            and passed compounds does not matter. Default: True
    """

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "unnamed")
        self.condition1 = AtomCondition(**kwargs.get("condition1", {}))
        self.condition2 = AtomCondition(**kwargs.get("condition2", {}))
        self.action1 = Action(kwargs.get("action1", []))
        self.action2 = Action(kwargs.get("action2", []))
        self.bond = kwargs.get("bond", None)
        self.sym = kwargs.get("sym", True)

    @staticmethod
    def get_bond_type(bond):
        """
        Convert bond name into the correct rdkit type.

        Arguments:
            bond (str): Bond type name. E.g.: single, double

        Returns:
            rdkit.Chem.rdchem.BondType: The rdkit bond type.
        """
        if bond is None:
            return None
        elif bond == "single":
            return Chem.rdchem.BondType.SINGLE
        elif bond == "double":
            return Chem.rdchem.BondType.DOUBLE
        else:
            raise NotImplementedError("Bond type '{}' is not implemented.".format(bond))

    def __can_apply(self, atom1, atom2):
        return self.condition1.check(atom1) and self.condition2.check(atom2)

    def __apply(self, mol, atom1, atom2):
        if not self.__can_apply(atom1, atom2):
            raise ValueError("Can not apply merge rule.")
        self.action1(mol, atom1)
        self.action2(mol, atom2)
        bond_type = MergeRule.get_bond_type(self.bond)
        if bond_type is not None:
            mol.AddBond(atom1.GetIdx(), atom2.GetIdx(), order=bond_type)
        return mol

    def can_apply(self, atom1, atom2):
        """
        Check if the rule can be applied to merge atom1 and atom2.

        Arguments:
            atom1 (rdkit.Chem.Atom): Atom in first compound.
            atom2 (rdkit.Chem.Atom): Atom in second compound.

        Returns:
            bool: True if the rule can be applied, false otherwise.
        """
        if self.sym:
            return self.__can_apply(atom1, atom2) or self.__can_apply(atom2, atom1)
        else:
            return self.__can_apply(atom1, atom2)

    def apply(self, mol, atom1, atom2):
        """
        Apply the merge rule to the given molecule.

        Arguments:
            mol (rdkit.Chem.Mol): The molecule containing both compounds.
            atom1 (rdkit.Chem.Atom): The boundary atom in compound 1.
            atom2 (rdkit.Chem.Atom): The boundary atom in compound 2.

        Returns:
            rdkit.Chem.Mol: The merged molecule.
        """
        try:
            if self.sym:
                if self.__can_apply(atom1, atom2):
                    return self.__apply(mol, atom1, atom2)
                else:
                    return self.__apply(mol, atom2, atom1)
            else:
                return self.__apply(mol, atom1, atom2)
        except Exception as e:
            raise MergeError(self.name, str(e)) from e


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


class CompoundRule:
    """
    Class for defining a compound expansion rule. The compound is added if the
    boundary atom meets the condition. A compound rule can be configured by
    providing a suitable dictionary. Examples on how to configure compound
    rules can be found in SynRBL/SynMCS/compound_rules.json file.

    Attributes:
        name (str, optional): A descriptive name for the rule. This attribute
            is just for readability and does not serve a functional purpose.
        condition (SynRBL.SynMCS.mol_merge.AtomCondition, optional): Condition
            for the boundary atom.
        compound (dict): The compound to add as dictionary in the following
            form: {'smiles': <SMILES>, 'index': <boundary atom index>}.
    """

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "unnamed")
        self.condition = AtomCondition(**kwargs.get("condition", {}))
        self.compound = kwargs.get("compound", None)

    def can_apply(self, atom, neighbor):
        """
        Checks if the compound rule can be applied to the atom.

        Arguments:
            atom (rdkit.Chem.Atom): The boundary atom which is checked for the
                compound expansion.
            neighbor (str): The neighboring atom to the boundary. This
                additional information is required to find the correct
                compound.

        Returns:
            bool: True if the compound rule can be applied, false otherwise.
        """
        return self.condition.check(atom, neighbor=neighbor)

    def apply(self, atom, neighbor):
        """
        Apply the compound rule.

        Arguments:
            atom (rdkit.Chem.Atom): The boundary atom.
            neighbor (str): The neighboring atom to the boundary atom.

        Returns:
            rdkit.Chem.Mol: Returns the compound for expansion.
        """
        if not self.can_apply(atom, neighbor):
            raise ValueError("Can not apply compound rule.")
        result = None
        if self.compound is not None and all(
            k in self.compound.keys() for k in ("smiles", "index")
        ):
            result = {
                "mol": Chem.MolFromSmiles(self.compound["smiles"]),
                "index": self.compound["index"],
            }
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
                atom.SetProp("atomLabel", str(atom.GetIdx()))
        mol_img = Draw.MolToImage(mol)
        if titles is not None and i < len(titles):
            a.set_title(titles[i])
        a.axis("off")
        a.imshow(mol_img)


def get_merge_rules() -> list[MergeRule]:
    """
    Get a list of merge rules. The rules are configured in
    SynRBL/SynMCS/merge_rules.json.

    Returns:
        list[SynRBL.SynMCS.mol_merge.MergeRule]: Returns a list of merge rules.
    """
    global _merge_rules
    if _merge_rules is None:
        json_data = files(SynRBL.SynMCS).joinpath("merge_rules.json").read_text()
        _merge_rules = [MergeRule(**c) for c in json.loads(json_data)]
    return _merge_rules


def get_compound_rules() -> list[CompoundRule]:
    """
    Get a list of compound rules. The rules are configured in
    SynRBL/SynMCS/compound_rules.json.

    Returns:
        list[SynRBL.SynMCS.mol_merge.CompoundRule]: Returns a list of compound
        rules.
    """
    global _compound_rules
    if _compound_rules is None:
        json_data = files(SynRBL.SynMCS).joinpath("compound_rules.json").read_text()
        _compound_rules = [CompoundRule(**c) for c in json.loads(json_data)]
    return _compound_rules


def get_compound(atom, neighbor):
    """
    Get the second compound for the specified boundary atom.

    Arguments:
        atom (rdkit.Chem.Atom): The boundary atom for which to find an
            extension compound.
        neighbor (str): The neighboring atom to the boundary atom.

    Returns:
        rdkit.Chem.Mol: Returns the extension molecule.
    """
    for rule in get_compound_rules():
        if rule.can_apply(atom, neighbor):
            return rule.apply(atom, neighbor), rule
    raise NoCompoundRuleError(atom.GetSymbol(), neighbor)


def merge_mols(mol1, mol2, idx1, idx2, mol1_track=None, mol2_track=None):
    """
    Merge two molecules. How and if the molecules are merge is defined by
    merge rules. For more details on merge rules see the
    SynRBL.SynMCS.mol_merge.MergeRule class documentation and the rule
    configuration in SynRBL/SynMCS/merge_rules.json.

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
            atom index mappings at 'aam1' and 'aam2' as well as the applied
            merge rule at 'rule'.
    """
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
    if not merge_rule:
        raise NoMergeRuleError(atom1, atom2, mol1, mol2)
    Chem.SanitizeMol(mol)
    return {
        "mol": mol,
        "merge_rules": [merge_rule],
        "aam1": mol1_tracker.to_dict(),
        "aam2": mol2_tracker.to_dict(),
    }


def merge_expand(mol, bound_indices, neighbors=None):
    """
    Expand and merge a single molecule with the product of a suitable compound
    rule. For more informatino on compound rules see the
    SynRBL.SynMCS.mol_merge.CompoundRule class documentation and the compound
    rule config in SynRBL/SynMCS/compound_rules.json.

    Arguments:
        mol (rdkit.Chem.Mol): The molecule to expand.
        bound_indices (list[int]): A list of boundary atom indices.
        neighbors (list[str]): The neighboring atom for each boundary atom.

    Returns:
        dict: A dictionary containing the expanded molecule at key 'mol', the
        used compound rules at 'compound_rules' and the used merge rules at
        'merge_rules'.
    """
    if not isinstance(bound_indices, list):
        raise ValueError("bound_indices must be of type list")
    bound_len = len(bound_indices)
    if neighbors is None:
        neighbors = [None for _ in range(bound_len)]
    if len(neighbors) != bound_len:
        raise ValueError(
            "neighbors list must be of same length as bound_indices. "
            + "(bound_indices={}, neighbors={})".format(bound_indices, neighbors)
        )

    merged_mol = mol
    used_compound_rules = []
    used_merge_rules = []
    for i in range(bound_len):
        atom = merged_mol.GetAtoms()[bound_indices[i]]
        comp, rule = get_compound(atom, neighbors[i])
        used_compound_rules.append(rule)
        if comp is not None:
            merge_result = merge_mols(
                merged_mol,
                comp["mol"],
                bound_indices[i],
                comp["index"],
                mol1_track=bound_indices,
            )
            bound_indices = [merge_result["aam1"][str(idx)] for idx in bound_indices]
            merged_mol = merge_result["mol"]
            used_merge_rules.extend(merge_result["merge_rules"])
    return {
        "mol": merged_mol,
        "compound_rules": used_compound_rules,
        "merge_rules": used_merge_rules,
    }


def _check_atoms(mol, atom_dict):
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
            _check_atoms(mol, e)
    elif isinstance(atom_dict, dict):
        sym, idx = next(iter(atom_dict.items()))
        actual_sym = mol.GetAtomWithIdx(idx).GetSymbol()
        if actual_sym != sym:
            raise InvalidAtomDict(sym, actual_sym, idx, Chem.MolToSmiles(mol))
    else:
        raise ValueError("agom_dict must be either a list or a dict.")


def _ad2t(atom_dict):
    """
    Convert atom dict to symbol and index tuple.

    Arguments:
        atom_dict (dict): Atom dictionary in the for of {<symbol>: <index>}

    Returns:
        (str, int): Atom dictionary as tuple (<symbol>, <index>}
    """
    if not isinstance(atom_dict, dict) or len(atom_dict) != 1:
        raise ValueError("atom_dict must be of type {<symbol>: <index>}")
    return next(iter(atom_dict.items()))


def _adl2t(atom_dict_list):
    """
    Split atom dict into symbol and indices lists.

    Arguments:
        atom_dict_list (list[dict]): The atom dictionary list.

    Returns:
        (list[str], list[ind]): Returns the symbol and indices lists.
    """
    sym_list = []
    idx_list = []
    for a in atom_dict_list:
        sym, idx = _ad2t(a)
        sym_list.append(sym)
        idx_list.append(idx)
    return sym_list, idx_list


def _split_mol(mol, bounds, neighbors):
    """
    Split not connected compounds in the molecule object into individual
    fragments and correct the bounds and neighbors lists accordingly.

    Arguments:
        mol (rdkit.Chem.Mol): The molecule to check for splits.
        bounds (list[dict]): Atom dict list of boundary atoms.
        neighbors (list[dict]): Atom dict list of boundary neighboring atoms.

    Returns:
        (list[rdkit.Chem.Mol], list[list[dict]], list[list[dict]]): Returns
            a list of compouns and the adjusted bounds and neighbors atom
            dict lists.
    """
    if not (
        isinstance(bounds, list) and len(bounds) > 0 and isinstance(bounds[0], dict)
    ):
        raise ValueError("bounds must be a list of atom dicts.")
    if not (
        isinstance(neighbors, list)
        and len(neighbors) == len(bounds)
        and isinstance(neighbors[0], dict)
    ):
        raise ValueError(
            "neighbors must be a list of atom dicts with the "
            + "same length as bounds."
        )

    frags = list(rdmolops.GetMolFrags(mol, asMols=True))
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
    """
    Merge molecules. This works for either a single molecule which is extended
    by a compound rule or for two molecules where a suitable merge rule exists.
    For additional information on compound and merge rules see the MergeRule
    and CompoundRule class documentation in module SynRBL.SynMCS.mol_merge and
    the rule configuration in merge_rules.json and compound_rules.json in
    SynRBL/SynMCS/.

    Arguments:
        mols (list[rdkit.Chem.Mol]): A list of molecules. Merging is only
            supported for individual expansions and molecule pairs.
        bounds (list[list[dict]]): A list of boundary atom dictionaries for
            each molecule.
        neighbors (list[list[dict]]): A list of neighboring atom dictionaries
            for each molecule.

    Returns:
        dict: Returns a dictionary with the merge molecule at key 'mol', the
            list of used merge rules at 'merge_rules', and the list of used
            compound rules at 'compound_rules'.
    """
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
        merged_mols.append(
            {
                "mol": merged_mol["mol"],
                "merge_rules": merged_mol["merge_rules"],
                "compound_rules": [],
            }
        )
    elif len(mols) > 2:
        raise NotImplementedError(
            "Merging of {} molecules is not supported.".format(len(mols))
        )
    return merged_mols
