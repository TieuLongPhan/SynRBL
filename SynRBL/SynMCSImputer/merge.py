import rdkit.Chem.rdchem as rdchem
import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.rdmolfiles as rdmolfiles

from .rules import MergeRule, CompoundRule
from .structure import Boundary, Compound


class NoCompoundRule(Exception):
    def __str__(self):
        return "No compound rule found."


def expand_boundary(boundary: Boundary) -> Compound:
    compound = None
    for rule in CompoundRule.get_all():
        if rule.can_apply(boundary):
            compound = rule.apply()
            compound.rules.append(rule)
            break
    if compound is None:
        raise NoCompoundRule()
    return compound


def merge_boundaries(boundary1: Boundary, boundary2: Boundary) -> Compound | None:
    for rule in MergeRule.get_all():
        if not rule.can_apply(boundary1, boundary2):
            continue
        return rule.apply(boundary1, boundary2)
    return None


def concat_compounds(compound1: Compound, compound2: Compound) -> Compound:
    if len(compound1.boundaries) > 0 or len(compound2.boundaries) > 0:
        raise ValueError(
            "Can not concat compounds with open boundaries. Try merging them."
        )
    src_mol = None
    try:
        src_mol = "{}.{}".format(compound1.src_smiles, compound2.src_smiles)
    except:
        pass
    concat_compound = Compound(
        "{}.{}".format(compound1.smiles, compound2.smiles),
        src_mol=src_mol,
    )
    concat_compound.rules = compound1.rules + compound2.rules
    return concat_compound


def _merge_one_compound(compound: Compound) -> Compound:
    merged_compound = compound
    while len(merged_compound.boundaries) > 0:
        boundary1 = merged_compound.boundaries[0]
        try:
            compound2 = expand_boundary(boundary1)
            if len(compound2.boundaries) != 1:
                raise NotImplementedError(
                    "Compound expansion and merge is only supported for "
                    + "compounds with a single boundary atom."
                )
            merged_compound = merge_boundaries(boundary1, compound2.boundaries[0])
        except NoCompoundRule:
            # If no compound rule is found, leave the compound as is
            merged_compound.update(merged_compound.mol, boundary1)
            pass
        if merged_compound is None:
            raise ValueError("No merge rule found.")
    return merged_compound


def _merge_two_compounds(compound1: Compound, compound2: Compound) -> Compound:
    boundaries1 = compound1.boundaries
    boundaries2 = compound2.boundaries
    merged_compound = None
    if len(boundaries1) != 1:
        raise NotImplementedError(
            ("Can only merge compounds with single boundary atom. ({})").format(
                len(boundaries1)
            )
        )
    if len(boundaries1) != len(boundaries2):
        if compound1.num_compounds == 1 and compound2.num_compounds == 1:
            # If boundaries don't match, try to expand-merge them.
            # This is only safe if the smiles contains only one compound,
            # otherwise MCS was probably wrong.
            compound1 = _merge_one_compound(compound1)
            compound2 = _merge_one_compound(compound2)
            merged_compound = concat_compounds(compound1, compound2)
        else:
            raise ValueError(
                (
                    "Can not merge compounds with unequal "
                    + "number of boundaries. ({} != {})."
                ).format(len(boundaries1), len(boundaries2))
            )
    else:
        merged_compound = merge_boundaries(boundaries1[0], boundaries2[0])
    if merged_compound is None:
        raise ValueError("No merge rule found.")
    return merged_compound


def merge(compounds: Compound | list[Compound], cs_passthrough=False) -> Compound:
    merged_compound = None

    if isinstance(compounds, Compound):
        compounds = list([compounds])

    catalysts_solvents = []
    if cs_passthrough:
        b_compounds = []  # compounds with open boundaries
        for compound in compounds:
            if len(compound.boundaries) == 0:
                catalysts_solvents.append(compound)
            else:
                b_compounds.append(compound)
        compounds = b_compounds

    if len(compounds) == 1:
        merged_compound = _merge_one_compound(compounds[0])
    elif len(compounds) == 2:
        merged_compound = _merge_two_compounds(compounds[0], compounds[1])

    if merged_compound is not None:
        for c in catalysts_solvents:
            merged_compound = concat_compounds(merged_compound, c)
        return merged_compound

    raise NotImplementedError(
        "Merging {} compounds is not supported.".format(len(compounds))
    )
