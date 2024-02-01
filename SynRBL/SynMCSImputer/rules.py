from __future__ import annotations
import json
import numpy as np
import importlib.resources
import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdmolops as rdmolops

import SynRBL.SynMCSImputer
import SynRBL.SynUtils.functional_group_utils as fgutils

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


def run_check(prop: Property, value, pos_callback, neg_callback, *args):
    if prop.allow_none and value is None:
        return True
    if len(prop.pos_values) > 0:
        found = False
        for i, pos_value in enumerate(prop.pos_values):
            if pos_callback(value, i, pos_value, *args):
                found = True
                break
            if not found:
                return False
    if len(prop.neg_values) > 0:
        for i, neg_value in enumerate(prop.neg_values):
            if neg_callback(value, i, neg_value, *args):
                return False
    return True


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
                if fgutils.is_functional_group(src_mol, v, neighbor_index):
                    found = True
                    break
            if not found:
                return False
        if len(self.neg_values) > 0:
            for v in self.neg_values:
                if fgutils.is_functional_group(src_mol, v, neighbor_index):
                    return False
        return True


class PatternCondition:
    def __init__(self, pattern=None, anchor=None, **kwargs):
        pattern = kwargs.get("pattern", pattern)
        anchor = kwargs.get("anchor", anchor)

        self.pattern = Property(pattern)
        pattern_cnt = len(self.pattern.pos_values) + len(self.pattern.neg_values)
        if anchor is None:
            anchor = [None for _ in range(pattern_cnt)]
        anchor = anchor if isinstance(anchor, list) else [anchor]
        if pattern_cnt != len(anchor):
            raise ValueError(
                "Config error! Pattern '{}' and anchor '{}' must be of same length.".format(
                    pattern, anchor
                )
            )

        self.anchor = anchor

    def check(self, boundary: Boundary) -> bool:
        def _check(boundary, pattern, pattern_anchor):
            pattern_mol = rdmolfiles.MolFromSmiles(pattern)
            match, mapping = fgutils.pattern_match(
                boundary.promise_src(),
                boundary.promise_neighbor_index(),
                pattern_mol,
                pattern_anchor,
            )
            print(
                boundary.compound.src_smiles,
                boundary.neighbor_index,
                rdmolfiles.MolToSmiles(pattern_mol),
                pattern_anchor,
                match,
                mapping,
            )
            return match

        def _pos_check(boundary, index, pattern, offset):
            pattern_anchor = self.anchor[index]
            return _check(boundary, pattern, pattern_anchor)

        def _neg_check(boundary, index, pattern, offset):
            pattern_anchor = self.anchor[offset + index]
            return _check(boundary, pattern, pattern_anchor)

        return run_check(
            self.pattern, boundary, _pos_check, _neg_check, len(self.pattern.pos_values)
        )


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

    def __init__(
        self,
        atom=None,
        neighbors=None,
        functional_groups=None,
        pattern_match=None,
        **kwargs,
    ):
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
        pattern_match = kwargs.get(
            "pattern_match", {} if pattern_match is None else pattern_match
        )

        self.atom = Property(atom)
        self.neighbors = Property(neighbors, allow_none=True)
        self.functional_groups = FunctionalGroupProperty(functional_groups)
        self.pattern_match = PatternCondition(**pattern_match)

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
                self.pattern_match.check(boundary),
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
