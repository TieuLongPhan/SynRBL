from __future__ import annotations
import json
import numpy as np
import importlib.resources
import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.rdchem as rdchem

import SynRBL.SynMCSImputer
import SynRBL.SynUtils.functional_group_utils as fgutils
import SynRBL.SynMCSImputer.utils as utils

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
        allow_none=False,
    ):
        """
        Generic property for dynamic rule configuration.

        Arguments:
            config (str | list[str]): Configuration for this property. This is
                a list of acceptable and forbiden values.
                e.g.: ['A', 'B'] -> check is true if value is A or B
                      ['!C', '!D'] -> check is true if value is not C and not D
            allow_none (bool): Flag if a check with value None is valid or not.
        """
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
                    self.neg_values.append(item[1:])
                else:
                    self.pos_values.append(item)

    def check(self, value, check_value) -> bool:
        return value == check_value

    def __call__(self, value) -> bool:
        """
        Check if the property is true for the given value.

        Arguments:
            value: The value to check. It must be of the same datatype as
                specified in the constructor.

        Returns:
            bool: True if the value fulfills the property, false otherwise.
        """
        if value is None:
            if self.allow_none:
                return True
            else:
                raise ValueError(
                    "Passed value None to property that does not allow None."
                )
        if len(self.pos_values) > 0:
            found = False
            for pv in self.pos_values:
                r = self.check(value, pv)
                # print("{} pos. check return {}.".format(self.__class__.__name__, r))
                if r:
                    found = True
                    break
            if not found:
                return False
        if len(self.neg_values) > 0:
            for nv in self.neg_values:
                r = self.check(value, nv)
                # print("{} neg. check return {}.".format(self.__class__.__name__, r))
                if r:
                    return False
        return True


class Action:
    __actions = {}

    @classmethod
    def build(cls, name, **kwargs) -> Action:
        if name not in cls.__actions.keys():
            raise NotImplementedError("No action named '{}' found.".format(name))
        inst = cls.__actions[name](**kwargs)
        return inst

    @classmethod
    def register(cls, name: str, action):
        if name in cls.__actions.keys():
            raise ValueError("Action with name '{}' already exists.".format(name))
        cls.__actions[name] = action

    def apply(self, boundary: Boundary):
        pass

    def __call__(self, boundary: Boundary):
        self.apply(boundary)


class ChangeBondAction(Action):
    def __init__(self, pattern=None, bond=None, **kwargs):
        pattern = kwargs.get("pattern", pattern)
        bond = kwargs.get("bond", bond)

        if pattern is None:
            raise ValueError("Missing required parameter 'pattern'.")

        if bond is None:
            raise ValueError("Missing required parameter 'bond'.")

        self.pattern_mol = rdmolfiles.MolFromSmiles(pattern)
        self.bond_type, _ = parse_bond_type(bond)

        if (
            len(self.pattern_mol.GetAtoms()) != 2
            or len(self.pattern_mol.GetBonds()) != 1
        ):
            raise ValueError(
                "Pattern for change_bond action must be a smiles with 2 "
                + "atoms and 1 bond. "
                + "Value '{}' is invalid.".format(pattern)
            )

    def apply(self, boundary: Boundary):
        match, mapping = fgutils.pattern_match(
            boundary.compound.mol, boundary.index, self.pattern_mol
        )
        assert match, (
            "No match found for pattern '{}' in '{}' @ {}. "
            + "Fix the condition in the rule configuration."
        ).format(
            rdmolfiles.MolToSmiles(self.pattern_mol),
            boundary.compound.smiles,
            boundary.index,
        )
        emol = rdchem.EditableMol(boundary.compound.mol)
        emol.RemoveBond(mapping[0][0], mapping[1][0])
        emol.AddBond(mapping[0][0], mapping[1][0], self.bond_type)
        boundary.compound.mol = emol.GetMol()


Action.register("change_bond", ChangeBondAction)


class FunctionalGroupProperty(Property):
    def __init__(self, config=None):
        super().__init__(config, allow_none=False)

    def check(self, value: Boundary, check_value) -> bool:
        if value.compound.src_mol is not None and value.neighbor_index is not None:
            src_mol = value.promise_src()
            neighbor_index = value.promise_neighbor_index()
            return fgutils.is_functional_group(src_mol, check_value, neighbor_index)
        else:
            return False


class PatternProperty(Property):
    def __init__(self, config=None, use_src_mol: bool = False):
        super().__init__(config, allow_none=False)
        self.use_src_mol = use_src_mol

    def check(self, boundary: Boundary, value) -> bool:
        pattern_mol = rdmolfiles.MolFromSmiles(value)
        mol = boundary.compound.mol
        index = boundary.index
        if self.use_src_mol:
            mol = boundary.promise_src()
            index = boundary.promise_neighbor_index()
        match, _ = fgutils.pattern_match(
            mol,
            index,
            pattern_mol,
        )
        return match


class BoundarySymbolProperty(Property):
    def __init__(self, config=None):
        super().__init__(config, allow_none=False)

    def check(self, value: Boundary, check_value) -> bool:
        return value.symbol == check_value


class NeighborSymbolProperty(Property):
    def __init__(self, config=None):
        super().__init__(config, allow_none=False)

    def check(self, value: Boundary, check_value) -> bool:
        return value.neighbor_symbol == check_value


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
        neighbor_atom (SynRBL.SynMCSImputer.rule_formation.Property): Neighbors property
    """

    def __init__(
        self,
        atom=None,
        neighbor_atom=None,
        functional_group=None,
        pattern=None,
        src_pattern=None,
        **kwargs,
    ):
        """
        Atom condition class to check if a rule is applicable to a specific
        molecule. Property configs can be prefixed with '!' to negate the
        check. See SynRBL.SynMCSImputer.rule_formation.Property for more information.

        Arguments:
            atom: Atom property configuration.
            neighbor: Neighbor property configuration.
            functional_groups: Functional group property configuration.
        """
        atom = kwargs.get("atom", atom)
        neighbor_atom = kwargs.get("neighbor_atom", neighbor_atom)
        functional_group = kwargs.get("functional_group", functional_group)
        pattern = kwargs.get("pattern", pattern)
        src_pattern = kwargs.get("src_pattern", src_pattern)

        self.properties = [
            BoundarySymbolProperty(atom),
            NeighborSymbolProperty(neighbor_atom),
            FunctionalGroupProperty(functional_group),
            PatternProperty(pattern),
            PatternProperty(src_pattern, use_src_mol=True),
        ]

    def __call__(self, boundary: Boundary):
        """
        Check if the boundary meets the condition.

        Arguments:
            boundary (SynRBS.SynMCS.structure.Boundary): Boundary the
                condition should be checked for.

        Returns:
            bool: True if the boundary fulfills the condition, false otherwise.
        """
        for prop in self.properties:
            if not prop(boundary):
                return False
        return True


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
    """

    _merge_rules: list[MergeRule] | None = None

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "unnamed")
        self.condition1 = BoundaryCondition(**kwargs.get("condition1", {}))
        self.condition2 = BoundaryCondition(**kwargs.get("condition2", {}))

        actions1 = kwargs.get("action1", [])
        actions2 = kwargs.get("action2", [])
        if not isinstance(actions1, list):
            actions1 = [actions1]
        if not isinstance(actions2, list):
            actions2 = [actions2]

        self.action1 = [Action.build(a["type"], **a) for a in actions1]
        self.action2 = [Action.build(a["type"], **a) for a in actions2]
        self.bond = kwargs.get("bond", None)

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

    def can_apply(self, boundary1: Boundary, boundary2: Boundary):
        """
        Check if the rule can be applied to merge atom1 and atom2.

        Arguments:
            boundary1 (SynRBL.SynMCSImputer.structure.Boundary): First boundary.
            boundary2 (SynRBL.SynMCSImputer.structure.Boundary): Second boundary.

        Returns:
            bool: True if the rule can be applied, false otherwise.
        """
        return (self.condition1(boundary1) and self.condition2(boundary2)) or (
            self.condition1(boundary2) and self.condition2(boundary1)
        )

    def apply(self, boundary1: Boundary, boundary2: Boundary) -> Compound | None:
        def _fix_Hs(atom, bond_nr):
            if atom.GetNumExplicitHs() > 0:
                atom.SetNumExplicitHs(
                    int(np.max([0, atom.GetNumExplicitHs() - bond_nr]))
                )

        if not (self.condition1(boundary1) and self.condition2(boundary2)):
            boundary1, boundary2 = boundary2, boundary1
        assert self.condition1(boundary1) and self.condition2(
            boundary2
        ), "Rule can not be applied."

        for a in self.action1:
            a(boundary1)
        for a in self.action2:
            a(boundary2)

        bond_type, bond_nr = parse_bond_type(self.bond)
        if bond_type is not None:
            _fix_Hs(boundary1.get_atom(), bond_nr)
            _fix_Hs(boundary2.get_atom(), bond_nr)

        merge_result = utils.merge_two_mols(
            boundary1.compound.mol,
            boundary2.compound.mol,
            boundary1.index,
            boundary2.index,
            bond_type,
        )

        mol = merge_result["mol"]
        rdmolops.SanitizeMol(mol)

        boundary1.compound.update(mol, boundary1)
        boundary1.compound.rules.extend(boundary2.compound.rules)
        boundary1.compound.rules.append(self)
        return boundary1.compound


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
        return self.condition(boundary)

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
