from __future__ import annotations
import json
import numpy as np
import SynRBL.SynMCSImputer
import importlib.resources
import rdkit.Chem as Chem
import rdkit.Chem.rdchem as rdchem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdmolops as rdmolops

from .structure import Boundary, Compound


def parse_bond_type(bond):
    """
    Convert bond name into the correct rdkit type.

    Arguments:
        bond (str): Bond type name. E.g.: single, double

    Returns:
        rdkit.Chem.rdchem.BondType: The rdkit bond type.
    """
    if bond is None:
        return None, 0
    elif bond == "single":
        return Chem.rdchem.BondType.SINGLE, 1
    elif bond == "double":
        return Chem.rdchem.BondType.DOUBLE, 2
    else:
        raise NotImplementedError("Bond type '{}' is not implemented.".format(bond))


class Property:
    """
    Generic property for dynamic rule configuration.

    Attributes:
        neg_values (list): List of forbidden values. Forbidden value check is
            skipped if this list is empty.
        pos_values (list): List of valid values. If this list is empty, all
            values are excepted.
        allow_none (bool): If a value of None is allowed and checked as valid.
    """

    def __init__(
        self,
        config: str | list[str] | None = None,
        dtype: type[int | str] = str,
        allow_none=False,
    ):
        """
        Generic property for dynamic rule configuration.

        Arguments:
            config (str | list[str]): Configuration for this property. This is
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
        if config is not None:
            if not isinstance(config, list):
                config = [config]
            for item in config:
                if not isinstance(item, str):
                    raise ValueError(
                        "Property configuration must be of type str or list[str]."
                    )
                if len(item) > 0 and item[0] == "!":
                    self.neg_values.append(dtype(item[1:]))
                else:
                    self.pos_values.append(dtype(item))

    def check(self, value) -> bool:
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
            raise ValueError("value must be of type '{}'.".format(self.__dtype))
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


def reduce(mol: rdchem.Mol, index, depth) -> rdchem.Mol:
    def _dfs(
        mol: rdchem.RWMol,
        atom: rdchem.Atom,
        dist: int,
        dist_array: list[int],
    ):
        dist_array[atom.GetIdx()] = dist
        for neighbor in atom.GetNeighbors():
            idx = neighbor.GetIdx()
            if dist + 1 < dist_array[idx]:
                _dfs(mol, neighbor, dist + 1, dist_array)

    atom_cnt = len(mol.GetAtoms())
    dist_array = [atom_cnt + 1 for _ in range(atom_cnt)]
    rwmol = rdchem.RWMol(mol)
    atom = rwmol.GetAtomWithIdx(index)
    _dfs(mol, atom, 0, dist_array)
    for i, d in reversed(list(enumerate(dist_array))):
        if d > depth:
            rwmol.RemoveAtom(i)
    return rwmol, atom.GetIdx()


class FGConfig:
    def __init__(self, pattern, group_atoms=None, anti_pattern=[], depth=None):
        if pattern != Chem.CanonSmiles(pattern):
            # We don't fix the pattern smiles because group_atoms might rely on the pattern
            raise ValueError(
                ("Pattern must be canonical smiles. (value: {}, expected: {})").format(
                    pattern, Chem.CanonSmiles(pattern)
                )
            )
        self.pattern_mol = rdmolfiles.MolFromSmiles(pattern)
        if group_atoms is None:
            self.group_mol = rdmolfiles.MolFromSmiles(pattern)
        else:
            self.group_mol = rdchem.RWMol(self.pattern_mol)
            rm_indices = [
                a.GetIdx()
                for a in self.group_mol.GetAtoms()
                if a.GetIdx() not in group_atoms
            ]
            for i in sorted(rm_indices, reverse=True):
                self.group_mol.RemoveAtom(i)
        anti_pattern = (
            anti_pattern if isinstance(anti_pattern, list) else [anti_pattern]
        )
        self.anti_pattern = sorted(
            [rdmolfiles.MolFromSmiles(s) for s in anti_pattern],
            key=lambda x: len(x.GetAtoms()),
            reverse=True,
        )
        self.max_depth = (
            depth
            if depth is not None
            else np.max(
                [len(c.GetAtoms()) for c in [self.pattern_mol] + self.anti_pattern]
            )
        )


functional_group_config = {
    "phenol": FGConfig("Oc1ccccc1"),
    "alcohol": FGConfig(
        "CO", anti_pattern=["C=O", "C(=O)O", "C=CO", "COC", "OCO", "Oc1ccccc1"]
    ),
    "ether": FGConfig(
        "COC", group_atoms=[1], anti_pattern=["C=O", "C=CO", "OCOC", "OC=O"]
    ),
    "enol": FGConfig("C=CO"),
    "amid": FGConfig("NC=O", anti_pattern=["O=C(N)O"]),
    "acyl": FGConfig(
        "C=O", anti_pattern=["SC=O", "C(N)=O", "O=C(O)O", "O=C(C)OC", "O=C(C)O"]
    ),
    "diol": FGConfig("OCO", anti_pattern=["OCOC", "O=C(O)O"]),
    "hemiacetal": FGConfig("COCO", anti_pattern=["COCOC", "O=C(O)O"]),
    "acetal": FGConfig("COCOC"),
    "urea": FGConfig("NC(=O)O"),
    "carbonat": FGConfig("O=C(O)O"),
    "anhydrid": FGConfig("CC(=O)OC=O"),
    "ester": FGConfig("COC(C)=O", group_atoms=[1, 2, 4], anti_pattern=["O=C(C)OC=O"]),
    "acid": FGConfig(
        "CC(=O)O",
        group_atoms=[1, 2, 3],
        anti_pattern=["O=CS", "O=C(C)OC=O", "O=C(C)OC"],
    ),
    "anilin": FGConfig("Nc1ccccc1"),
    "amin": FGConfig("CN", anti_pattern=["C=O", "Nc1ccccc1", "NC=O"]),
    "nitril": FGConfig("C#N"),
    "hydroxylamin": FGConfig("NO", anti_pattern=["O=NO"]),
    "nitrose": FGConfig("N=O", anti_pattern=["O=NO"]),
    "nitro": FGConfig("O=NO"),
    "thioether": FGConfig("CSC", group_atoms=[1], anti_pattern=["O=CS"]),
    "thioester": FGConfig("O=CS"),
}


def is_functional_group(mol: rdchem.Mol, group_name: str, index: int) -> bool:
    if group_name not in functional_group_config.keys():
        raise NotImplementedError(
            "Functional group '{}' is not implemented.".format(group_name)
        )
    config = functional_group_config[group_name]

    rmol, index = reduce(mol, index, config.max_depth - 1)
    pattern_match = list(rmol.GetSubstructMatch(config.pattern_mol))
    is_func_group = False
    if index in pattern_match:
        group_match = list(rmol.GetSubstructMatch(config.group_mol))
        is_func_group = index in group_match

    last_len = config.max_depth
    for ap_mol, ap_mol_size in sorted(
        [(m, len(m.GetAtoms())) for m in config.anti_pattern],
        key=lambda x: x[1],
        reverse=True,
    ):
        if not is_func_group:
            break
        ap_mol_size = len(ap_mol.GetAtoms())
        if last_len > ap_mol_size:
            rmol, index = reduce(rmol, index, ap_mol_size - 1)
            last_len = ap_mol_size
        group_match = list(rmol.GetSubstructMatch(ap_mol))
        is_func_group = is_func_group and index not in group_match
    return is_func_group


class FunctionalGroupProperty(Property):
    def __init__(self, functional_groups=None):
        super().__init__(functional_groups, allow_none=False)

    def check(self, value: Boundary) -> bool:
        if not isinstance(value, Boundary):
            raise TypeError("Value must be of type boundary.")
        if len(self.pos_values) + len(self.neg_values) == 0:
            return True
        src_mol = value.promise_src()
        neighbor_index = value.promise_neighbor_index()
        if len(self.pos_values) > 0:
            found = False
            for v in self.pos_values:
                if is_functional_group(src_mol, v, neighbor_index):
                    found = True
                    break
            if not found:
                return False
        if len(self.neg_values) > 0:
            for v in self.neg_values:
                if is_functional_group(src_mol, v, neighbor_index):
                    return False
        return True


class BoundaryCondition:
    """
    Atom condition class to check if a rule is applicable to a specific
    molecule. Property configs can be prefixed with '!' to negate the check.
    See SynRBL.SynMCSImputer.rule_formation.Property for more information.

    Example:
        Check if atom is Carbon and has Oxygen or Nitrogen as neighbor.
        >>> cond = AtomCondition(atom=['C'], neighbors=['O', 'N'])
        >>> mol = rdkit.Chem.rdmolfiles.MolFromSmiles('CO')
        >>> cond.check(mol.GetAtomFromIdx(0), neighbor='O')
        True

    Attributes:
        atom (SynRBL.SynMCSImputer.rule_formation.Property): Atom property
        neighbors (SynRBL.SynMCSImputer.rule_formation.Property): Neighbors property
    """

    def __init__(self, atom=None, neighbors=None, functional_groups=None, **kwargs):
        """
        Atom condition class to check if a rule is applicable to a specific
        molecule. Property configs can be prefixed with '!' to negate the
        check. See SynRBL.SynMCSImputer.rule_formation.Property for more information.

        Arguments:
            atom: Atom property configuration.
            neighbors: Neighbors property configuration.
            functional_groups: Functional group property configuration.
        """
        atom = kwargs.get("atom", atom)
        neighbors = kwargs.get("neighbors", neighbors)
        functional_groups = kwargs.get("functional_groups", functional_groups)

        self.atom = Property(atom)
        self.neighbors = Property(neighbors, allow_none=True)
        self.functional_groups = FunctionalGroupProperty(functional_groups)

    def check(self, boundary: Boundary):
        """
        Check if the boundary meets the condition.

        Arguments:
            boundary (SynRBS.SynMCS.structure.Boundary): Boundary the
                condition should be checked for.

        Returns:
            bool: True if the boundary fulfills the condition, false otherwise.
        """
        return all(
            [
                self.atom.check(boundary.symbol),
                self.neighbors.check(boundary.neighbor_symbol),
                self.functional_groups.check(boundary),
            ]
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
                rdmolfiles.MolToSmiles(rdmolops.RemoveHs(mol1)),
                rdmolfiles.MolToSmiles(rdmolops.RemoveHs(mol2)),
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


class MergeRule:
    """
    Class for defining a merge rule between two compounds. If boundary atom1
    meets condition1 and boundary atom2 meets condition2 a bond is formed
    between atom1 and atom2. A merge rule can be configured by providing a
    suitable dictionary. Examples on how to configure merge rules can be found
    in SynRBL/SynMCS/merge_rules.json file.

    Example:
        The following example shows a complete default configuration for a merge rule.
        config = {
            "name": "unnamed",
            "condition1": {
                "atom": None,
                "neighbors": None,
            },
            "condition2": {
                "atom": None,
                "neighbors": None,
            },
            "bond": None,
            "sym": True
        }
        rule = MergeRule(**config)

    Attributes:
        name (str, optional): A descriptive name for the rule. This attribute
            is just for readability and does not serve a functional purpose.
        condition1 (AtomCondition, optional): Condition
            for the first boundary atom.
        condition2 (AtomCondition, optional): Condition
            for the second boundary atom.
        bond (str, optional): The bond type to form between the two compounds.
        sym (bool): If the rule is symmetric. If set to True order of condition
            and passed compounds does not matter. Default: True
    """

    _merge_rules: list[MergeRule] | None = None

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "unnamed")
        self.condition1 = BoundaryCondition(**kwargs.get("condition1", {}))
        self.condition2 = BoundaryCondition(**kwargs.get("condition2", {}))
        self.bond = kwargs.get("bond", None)
        self.sym = kwargs.get("sym", True)

    @classmethod
    def get_all(cls) -> list[MergeRule]:
        """
        Get a list of merge rules. The rules are configured in
        SynRBL/SynMCS/merge_rules.json.

        Returns:
            list[MergeRule]: Returns a list of merge rules.
        """
        if cls._merge_rules is None:
            json_data = (
                importlib.resources.files(SynRBL.SynMCSImputer)
                .joinpath("merge_rules.json")
                .read_text()
            )
            cls._merge_rules = [MergeRule(**c) for c in json.loads(json_data)]
        return cls._merge_rules

    def __can_apply(self, boundary1, boundary2):
        return self.condition1.check(boundary1) and self.condition2.check(boundary2)

    def can_apply(self, boundary1: Boundary, boundary2: Boundary):
        """
        Check if the rule can be applied to merge atom1 and atom2.

        Arguments:
            boundary1 (SynRBL.SynMCSImputer.structure.Boundary): First boundary.
            boundary2 (SynRBL.SynMCSImputer.structure.Boundary): Second boundary.

        Returns:
            bool: True if the rule can be applied, false otherwise.
        """
        if self.sym:
            return self.__can_apply(boundary1, boundary2) or self.__can_apply(
                boundary2, boundary1
            )
        else:
            return self.__can_apply(boundary1, boundary2)

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

        def _fix_Hs(atom, bond_nr):
            if atom.GetNumExplicitHs() > 0:
                atom.SetNumExplicitHs(
                    int(np.max([0, atom.GetNumExplicitHs() - bond_nr]))
                )

        bond_type, bond_nr = parse_bond_type(self.bond)
        if bond_type is not None:
            _fix_Hs(atom1, bond_nr)
            _fix_Hs(atom2, bond_nr)
            mol.AddBond(atom1.GetIdx(), atom2.GetIdx(), order=bond_type)
        return mol


class CompoundRule:
    """
    Class for defining a compound expansion rule. The compound is added if the
    boundary atom meets the condition. A compound rule can be configured by
    providing a suitable dictionary. Examples on how to configure compound
    rules can be found in SynRBL/SynMCS/compound_rules.json file.

    Attributes:
        name (str, optional): A descriptive name for the rule. This attribute
            is just for readability and does not serve a functional purpose.
        condition (AtomCondition, optional): Condition
            for the boundary atom.
        compound (dict): The compound to add as dictionary in the following
            form: {'smiles': <SMILES>, 'index': <boundary atom index>}.
    """

    _compound_rules: list[CompoundRule] | None = None

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "unnamed")
        self.condition = BoundaryCondition(**kwargs.get("condition", {}))
        self.compound = kwargs.get("compound", None)

    @classmethod
    def get_all(cls) -> list[CompoundRule]:
        """
        Get a list of compound rules. The rules are configured in
        SynRBL/SynMCS/compound_rules.json.

        Returns:
            list[CompoundRule]: Returns a list of compound rules.
        """
        if cls._compound_rules is None:
            json_data = (
                importlib.resources.files(SynRBL.SynMCSImputer)
                .joinpath("compound_rules.json")
                .read_text()
            )
            cls._compound_rules = [CompoundRule(**c) for c in json.loads(json_data)]
        return cls._compound_rules

    def can_apply(self, boundary: Boundary):
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
        return self.condition.check(boundary)

    def apply(self) -> Compound:
        """
        Apply the compound rule.

        Returns:
            SynRBL.SynMCSImputer.structure.Compound: The compound generated by this
                rule.
        """
        compound = Compound(self.compound["smiles"])
        compound.add_boundary(self.compound["index"])
        return compound


def get_merge_rules() -> list[MergeRule]:
    """
    Get a list of merge rules. The rules are configured in
    SynRBL/SynMCS/merge_rules.json.

    Returns:
        list[MergeRule]: Returns a list of merge rules.
    """
    return MergeRule.get_all()


def get_compound_rules() -> list[CompoundRule]:
    """
    Get a list of compound rules. The rules are configured in
    SynRBL/SynMCS/compound_rules.json.

    Returns:
        list[CompoundRule]: Returns a list of compound rules.
    """
    return CompoundRule.get_all()