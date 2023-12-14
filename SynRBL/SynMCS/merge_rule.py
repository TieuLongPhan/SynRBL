from __future__ import annotations
import json
import SynRBL.SynMCS
from importlib.resources import files
from rdkit.Chem import rdmolops
from SynRBL.SynMCS.rule_formation import AtomCondition, ActionSet
from rdkit import Chem

_merge_rules = None


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
        rule = MergeRule(**config)

    Attributes:
        name (str, optional): A descriptive name for the rule. This attribute
            is just for readability and does not serve a functional purpose.
        condition1 (SynRBL.SynMCS.mol_merge.AtomCondition, optional): Condition
            for the first boundary atom.
        condition2 (SynRBL.SynMCS.mol_merge.AtomCondition, optional): Condition
            for the second boundary atom.
        action1 (SynRBL.SynMCS.rule_formation.ActionSet, optional): Actions to
            performe on the first boundary atom.
        action2 (SynRBL.SynMCS.rule_formation.ActionSet, optional): Actions to
            performe on the second boundary atom.
        bond (str, optional): The bond type to form between the two compounds.
        sym (bool): If the rule is symmetric. If set to True order of condition
            and passed compounds does not matter. Default: True
    """

    def __init__(self, **kwargs):
        self.name = kwargs.get("name", "unnamed")
        self.condition1 = AtomCondition(**kwargs.get("condition1", {}))
        self.condition2 = AtomCondition(**kwargs.get("condition2", {}))
        self.actions1 = ActionSet(kwargs.get("action1", []))
        self.actions2 = ActionSet(kwargs.get("action2", []))
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

    @staticmethod
    def get_all() -> list[MergeRule]:
        return get_merge_rules()

    def __can_apply(self, atom1, atom2):
        return self.condition1.check(atom1) and self.condition2.check(atom2)

    def __apply(self, mol, atom1, atom2):
        if not self.__can_apply(atom1, atom2):
            raise ValueError("Can not apply merge rule.")
        self.actions1(mol, atom1)
        self.actions2(mol, atom2)
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
