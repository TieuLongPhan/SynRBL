from __future__ import annotations

import json
import inspect
import logging
import numpy as np
import importlib.resources
import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.rdchem as rdchem

import synrbl.SynMCSImputer
import synrbl.SynUtils.functional_group_utils as fgutils
import synrbl.SynMCSImputer.utils as utils

from synrbl.SynUtils.chem_utils import remove_atom_mapping
from .structure import Boundary, Compound

logger = logging.getLogger("synrbl")


def _check_config(init, ignore=[], **kwargs):
    exp_args = inspect.getfullargspec(init)[0]
    for k in kwargs.keys():
        if k in ignore:
            continue
        if k not in exp_args:
            raise KeyError(
                "Parameter '{}' is not valid for '{}'.".format(k, init.__name__)
            )


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
            # type bool or int
            if isinstance(config, bool) or isinstance(config, int):
                self.pos_values.append(config)
            # type str
            else:
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


class CompoundProperty(Property):
    def __init__(self, config=None, allow_none=False):
        super().__init__(config, allow_none)

    def __call__(self, value) -> bool:
        if not isinstance(value, Compound):
            raise TypeError(
                "Compound property can only be called with a compound as argument."
            )
        return super().__call__(value)


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
        _check_config(ChangeBondAction, ignore=["type"], **kwargs)
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


class ChangeChargeAction(Action):
    def __init__(self, charge=None, relative=False, **kwargs):
        _check_config(ChangeChargeAction, ignore=["type"], **kwargs)
        self.charge = kwargs.get("charge", charge)
        self.relative = kwargs.get("relative", charge)

        if self.charge is None:
            raise ValueError("Missing required parameter 'charge'.")

        self.charge = int(self.charge)

    def apply(self, boundary: Boundary):
        atom = boundary.compound.mol.GetAtomWithIdx(boundary.index)
        atom.SetFormalCharge(self.charge)


class ReplaceAction(Action):
    def __init__(self, pattern=None, value=None, **kwargs):
        _check_config(ReplaceAction, ignore=["type"], **kwargs)
        self.pattern = kwargs.get("pattern", pattern)
        self.value = kwargs.get("value", value)

    def apply(self, boundary: Boundary):
        smiles = boundary.compound.smiles
        smiles.replace(self.pattern, self.value)
        mol = rdmolfiles.MolFromSmiles(smiles)
        new_symbol = mol.GetAtomWithIdx(boundary.index).GetSymbol()
        if boundary.symbol != new_symbol:
            raise ValueError(
                (
                    "Replace action changed boundary atom type from {} to {}. "
                    + "This is not allowed."
                ).format(boundary.symbol, new_symbol)
            )
        boundary.compound.mol = mol


Action.register("change_bond", ChangeBondAction)
Action.register("change_charge", ChangeChargeAction)
Action.register("replace", ReplaceAction)


class CompoundAction:
    __actions = {}

    @classmethod
    def build(cls, name, **kwargs) -> CompoundAction:
        if name not in cls.__actions.keys():
            raise NotImplementedError(
                "No compound action named '{}' found.".format(name)
            )
        inst = cls.__actions[name](**kwargs)
        return inst

    @classmethod
    def register(cls, name: str, action):
        if name in cls.__actions.keys():
            raise ValueError(
                "Compound action with name '{}' already exists.".format(name)
            )
        cls.__actions[name] = action

    def apply(self, compound: Compound):
        pass

    def __call__(self, compound: Compound) -> Compound | None:
        self.apply(compound)


class AddBoundaryAction(CompoundAction):
    def __init__(self, functional_group=None, pattern=None, index=None, **kwargs):
        _check_config(AddBoundaryAction, ignore=["type"], **kwargs)
        functional_group = kwargs.get("functional_group", functional_group)
        pattern = kwargs.get("pattern", pattern)
        index = kwargs.get("index", index)

        if pattern is None:
            raise ValueError("Missing required parameter 'pattern'.")

        if index is None:
            raise ValueError("Missing required parameter 'index'.")

        self.functional_group = functional_group
        self.pattern_mol = rdmolfiles.MolFromSmiles(pattern)
        self.index = int(index)

        if len(self.pattern_mol.GetAtoms()) <= self.index:
            raise ValueError(
                "Index for pattern in add_boundary action is out of range."
            )

    def apply(self, compound: Compound):
        for atom in compound.mol.GetAtoms():
            if atom.GetSymbol() in ["C", "H"]:
                continue
            atom_index = atom.GetIdx()
            if self.functional_group is not None:
                is_fg = fgutils.is_functional_group(
                    compound.mol, self.functional_group, atom_index
                )
                if not is_fg:
                    continue
            match, mapping = fgutils.pattern_match(
                compound.mol, atom_index, self.pattern_mol, self.index
            )
            if match:
                for m in mapping:
                    if m[1] == self.index:
                        compound.add_boundary(m[0])
                        return
        raise RuntimeError("Action add_boundary could not be applied.")


class SetActiveAction(CompoundAction):
    def __init__(self, active=None, **kwargs):
        _check_config(SetActiveAction, ignore=["type"], **kwargs)
        self.active = bool(kwargs.get("active", active))

    def apply(self, compound: Compound):
        compound.active = self.active


CompoundAction.register("add_boundary", AddBoundaryAction)
CompoundAction.register("set_active", SetActiveAction)


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


class CountBoundariesCompoundProperty(CompoundProperty):
    def __init__(self, config=None, use_set=False):
        super().__init__(config, allow_none=False)
        self.use_set = use_set

    def check(self, value: Compound, check_value) -> bool:
        value_l = len(value.boundaries)
        if self.use_set:
            value_l = len(value.compound_set.boundaries)
        return value_l == int(check_value)


class CountCompoundsCompoundProperty(CompoundProperty):
    def __init__(self, config=None, compound_type=None):
        super().__init__(config, allow_none=False)
        self.compound_type = compound_type

    def check(self, value: Compound, check_value) -> bool:
        compounds = value.compound_set.compounds
        if self.compound_type == "open":
            compounds = [c for c in compounds if not c.is_catalyst]
        elif self.compound_type == "catalyst":
            compounds = [c for c in compounds if c.is_catalyst]
        return len(compounds) == int(check_value)


class IsCatalystCompoundProperty(CompoundProperty):
    def __init__(self, config=None):
        super().__init__(config, allow_none=False)

    def check(self, value: Compound, check_value) -> bool:
        return value.is_catalyst == bool(check_value)


class FunctionalGroupCompoundProperty(CompoundProperty):
    def __init__(self, config=None):
        super().__init__(config, allow_none=False)

    def check(self, value: Compound, check_value) -> bool:
        for atom in value.mol.GetAtoms():
            idx = atom.GetIdx()
            if atom.GetSymbol() not in ["H", "C"]:
                is_fg = fgutils.is_functional_group(value.mol, check_value, idx)
                if is_fg:
                    return True
        return False


class SmilesCompoundProperty(CompoundProperty):
    def __init__(self, config=None, use_src_mol: bool = False):
        super().__init__(config, allow_none=False)
        self.use_src_mol = use_src_mol

    def check(self, value: Compound, check_value) -> bool:
        if self.use_src_mol:
            if value.src_smiles is not None:
                return remove_atom_mapping(value.src_smiles) == check_value
            return value.src_smiles == check_value
        else:
            return remove_atom_mapping(value.smiles) == check_value


class BoundaryCondition:
    """
    Atom condition class to check if a rule is applicable to a specific
    molecule. Property configs can be prefixed with '!' to negate the check.
    See synrbl.SynMCSImputer.rule_formation.Property for more information.

    Example:
        Check if atom is Carbon and has Oxygen or Nitrogen as neighbor.
        >>> cond = AtomCondition(atom=['C'], neighbors=['O', 'N'])
        >>> mol = rdkit.Chem.rdmolfiles.MolFromSmiles('CO')
        >>> cond.check(mol.GetAtomFromIdx(0), neighbor='O')
        True

    Attributes:
        atom (synrbl.SynMCSImputer.rule_formation.Property): Atom property
        neighbor_atom (synrbl.SynMCSImputer.rule_formation.Property): Neighbors property
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
        check. See synrbl.SynMCSImputer.rule_formation.Property for more information.

        Arguments:
            atom: Atom property configuration.
            neighbor: Neighbor property configuration.
            functional_groups: Functional group property configuration.
        """
        _check_config(BoundaryCondition, **kwargs)
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


class CompoundCondition:
    def __init__(
        self,
        nr_boundaries=None,
        is_catalyst=None,
        smiles=None,
        functional_group=None,
        **kwargs,
    ):
        _check_config(CompoundCondition, **kwargs)
        nr_boundaries = kwargs.get("nr_boundaries", nr_boundaries)
        is_catalyst = kwargs.get("is_catalyst", is_catalyst)
        smiles = kwargs.get("smiles", smiles)
        functional_group = kwargs.get("functional_group", functional_group)

        self.properties = [
            CountBoundariesCompoundProperty(nr_boundaries),
            IsCatalystCompoundProperty(is_catalyst),
            SmilesCompoundProperty(smiles),
            FunctionalGroupCompoundProperty(functional_group),
        ]

    def __call__(self, compound: Compound):
        for prop in self.properties:
            if not prop(compound):
                return False
        return True


class SetCondition:
    def __init__(self, nr_boundaries=None, nr_compounds=None, **kwargs):
        _check_config(SetCondition, **kwargs)
        nr_boundaries = kwargs.get("nr_boundaries", nr_boundaries)
        nr_compounds = kwargs.get("nr_compounds", nr_compounds)

        self.properties = [
            CountBoundariesCompoundProperty(nr_boundaries, use_set=True),
            CountCompoundsCompoundProperty(nr_compounds),
        ]

    def __call__(self, compound: Compound):
        for prop in self.properties:
            if not prop(compound):
                return False
        return True


class CompoundRuleCondition:
    def __init__(self, compound={}, set={}, **kwargs):
        _check_config(CompoundRuleCondition, **kwargs)
        compound_condition = kwargs.get("compound", compound)
        set_condition = kwargs.get("set", set)

        self.compound_condition = CompoundCondition(**compound_condition)
        self.set_condition = SetCondition(**set_condition)

    def __call__(self, compound: Compound) -> bool:
        return self.compound_condition(compound) and self.set_condition(compound)


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
    in synrbl/SynMCS/merge_rules.json file.

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

    def __init__(
        self,
        name="unnamed",
        condition1={},
        condition2={},
        action1=[],
        action2=[],
        bond=None,
        **kwargs,
    ):
        _check_config(MergeRule, **kwargs)
        self.name = kwargs.get("name", name)
        self.condition1 = BoundaryCondition(**kwargs.get("condition1", condition1))
        self.condition2 = BoundaryCondition(**kwargs.get("condition2", condition2))

        actions1 = kwargs.get("action1", action1)
        actions2 = kwargs.get("action2", action2)
        if not isinstance(actions1, list):
            actions1 = [actions1]
        if not isinstance(actions2, list):
            actions2 = [actions2]

        self.action1 = [Action.build(a["type"], **a) for a in actions1]
        self.action2 = [Action.build(a["type"], **a) for a in actions2]
        self.bond = kwargs.get("bond", bond)

    @classmethod
    def get_all(cls) -> list[MergeRule]:
        """
        Get a list of merge rules. The rules are configured in
        synrbl/SynMCS/merge_rules.json.

        Returns:
            list[MergeRule]: Returns a list of merge rules.
        """
        if cls._merge_rules is None:
            json_data = (
                importlib.resources.files(synrbl.SynMCSImputer)
                .joinpath("merge_rules.json")
                .read_text()
            )
            cls._merge_rules = [MergeRule(**c) for c in json.loads(json_data)]
        return cls._merge_rules

    def can_apply(self, boundary1: Boundary, boundary2: Boundary):
        """
        Check if the rule can be applied to merge atom1 and atom2.

        Arguments:
            boundary1 (synrbl.SynMCSImputer.structure.Boundary): First boundary.
            boundary2 (synrbl.SynMCSImputer.structure.Boundary): Second boundary.

        Returns:
            bool: True if the rule can be applied, false otherwise.
        """
        try:
            return (self.condition1(boundary1) and self.condition2(boundary2)) or (
                self.condition1(boundary2) and self.condition2(boundary1)
            )
        except Exception as e:
            logger.warning(
                (
                    "Applicability check for rule '{}' failed with "
                    + "the following exception: {}"
                ).format(self.name, str(e))
            )
            return False

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


class ExpandRule:
    """
    Class for defining a compound expansion rule. The compound is added if the
    boundary atom meets the condition. A compound rule can be configured by
    providing a suitable dictionary. Examples on how to configure compound
    rules can be found in synrbl/SynMCS/expand_rules.json file.

    Attributes:
        name (str, optional): A descriptive name for the rule. This attribute
            is just for readability and does not serve a functional purpose.
        condition (AtomCondition, optional): Condition
            for the boundary atom.
        compound (dict): The compound to add as dictionary in the following
            form: {'smiles': <SMILES>, 'index': <boundary atom index>}.
    """

    _expand_rules: list[ExpandRule] | None = None

    def __init__(self, name="unnamed", condition={}, compound=None, **kwargs):
        _check_config(ExpandRule, **kwargs)
        self.name = kwargs.get("name", name)
        self.condition = BoundaryCondition(**kwargs.get("condition", condition))
        self.compound = kwargs.get("compound", compound)

    @classmethod
    def get_all(cls) -> list[ExpandRule]:
        """
        Get a list of compound expansion rules. The rules are configured in
        synrbl/SynMCS/expand.json.

        Returns:
            list[ExpandRule]: Returns a list of compound rules.
        """
        if cls._expand_rules is None:
            json_data = (
                importlib.resources.files(synrbl.SynMCSImputer)
                .joinpath("expand_rules.json")
                .read_text()
            )
            cls._expand_rules = [ExpandRule(**c) for c in json.loads(json_data)]
        return cls._expand_rules

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
            synrbl.SynMCSImputer.structure.Compound: The compound generated by this
                rule.
        """
        compound = Compound(self.compound["smiles"])
        compound.add_boundary(self.compound["index"])
        compound.rules.append(self)
        return compound


class CompoundRule:
    _compound_rules: list[CompoundRule] | None = None

    def __init__(self, name="unnamed", condition={}, action=[], **kwargs):
        _check_config(CompoundRule, **kwargs)

        action = kwargs.get("action", action)
        if not isinstance(action, list):
            action = [action]
        condition = kwargs.get("condition", condition)

        self.name = kwargs.get("name", name)
        self.condition = CompoundRuleCondition(**condition)
        self.action = [CompoundAction.build(a["type"], **a) for a in action]

    @classmethod
    def get_all(cls) -> list[CompoundRule]:
        if cls._compound_rules is None:
            json_data = (
                importlib.resources.files(synrbl.SynMCSImputer)
                .joinpath("compound_rules.json")
                .read_text()
            )
            cls._compound_rules = [CompoundRule(**c) for c in json.loads(json_data)]
        return cls._compound_rules

    def can_apply(self, compound: Compound):
        return self.condition(compound)

    def apply(self, compound: Compound) -> Compound:
        for action in self.action:
            action(compound)
        compound.rules.append(self)
        return compound


def get_merge_rules() -> list[MergeRule]:
    """
    Get a list of merge rules. The rules are configured in
    synrbl/SynMCS/merge_rules.json.

    Returns:
        list[MergeRule]: Returns a list of merge rules.
    """
    return MergeRule.get_all()


def get_expand_rules() -> list[ExpandRule]:
    """
    Get a list of compound expandsion rules. The rules are configured in
    synrbl/SynMCS/expand_rules.json.

    Returns:
        list[ExpandRule]: Returns a list of compound rules.
    """
    return ExpandRule.get_all()


def get_compound_rules() -> list[CompoundRule]:
    return CompoundRule.get_all()
