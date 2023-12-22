from __future__ import annotations
import json
import SynRBL.SynMCS
import importlib.resources
import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdmolops as rdmolops

from .structure import Boundary
from .rule_formation import BoundaryCondition


def parse_bond_type(bond):
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
                importlib.resources.files(SynRBL.SynMCS)
                .joinpath("merge_rules.json")
                .read_text()
            )
            cls._merge_rules = [MergeRule(**c) for c in json.loads(json_data)]
        return cls._merge_rules

    def __can_apply(self, boundary1, boundary2):
        return self.condition1.check(boundary1) and self.condition2.check(boundary2)

    def __apply(self, mol, atom1, atom2):
        if not self.__can_apply(atom1, atom2):
            raise ValueError("Can not apply merge rule.")
        bond_type = parse_bond_type(self.bond)
        if bond_type is not None:
            mol.AddBond(atom1.GetIdx(), atom2.GetIdx(), order=bond_type)
        return mol

    def can_apply(self, boundary1: Boundary, boundary2: Boundary):
        """
        Check if the rule can be applied to merge atom1 and atom2.

        Arguments:
            boundary1 (SynRBL.SynMCS.structure.Boundary): First boundary.
            boundary2 (SynRBL.SynMCS.structure.Boundary): Second boundary.

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
                importlib.resources.files(SynRBL.SynMCS)
                .joinpath("merge_rules.json")
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

    def apply(self):
        """
        Apply the compound rule.

        Returns:
            SynRBL.SynMCS.structure.Compound: The compound generated by this
                rule.
        """
        # TODO continue
        result = None
        if self.compound is not None and all(
            k in self.compound.keys() for k in ("smiles", "index")
        ):
            result = {
                "mol": rdmolfiles.MolFromSmiles(self.compound["smiles"]),
                "index": self.compound["index"],
            }
        return result


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
