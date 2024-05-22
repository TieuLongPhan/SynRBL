from .rules import MergeRule, ExpandRule, CompoundRule
from .structure import Boundary, Compound, CompoundSet


class NoExpandRule(Exception):
    def __str__(self):
        return "No expand rule found."


def expand_boundary(boundary: Boundary) -> Compound:
    compound = None
    for rule in ExpandRule.get_all():
        if rule.can_apply(boundary):
            compound = rule.apply()
            break
    if compound is None:
        raise NoExpandRule()
    return compound


def merge_boundaries(boundary1: Boundary, boundary2: Boundary) -> Compound | None:
    for rule in MergeRule.get_all():
        if not rule.can_apply(boundary1, boundary2):
            continue
        return rule.apply(boundary1, boundary2)
    return None


def update_compound(compound: Compound):
    for rule in CompoundRule.get_all():
        if rule.can_apply(compound):
            rule.apply(compound)
            break


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
        except NoExpandRule:
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
            compound1.concat(compound2)
            merged_compound = compound1
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


def merge(compound_set: CompoundSet) -> Compound:
    merged_compound = None

    comps_with_boundaries, comps_without_boundaries = [], []
    removed_rules = []
    for c in compound_set.compounds:
        update_compound(c)
        if not c.active:
            removed_rules.extend(c.rules)
            continue
        if len(c.boundaries) == 0:
            comps_without_boundaries.append(c)
        else:
            comps_with_boundaries.append(c)

    if len(comps_with_boundaries) == 0 and len(comps_without_boundaries) > 0:
        merged_compound = comps_without_boundaries.pop()
    elif len(comps_with_boundaries) == 1:
        merged_compound = _merge_one_compound(comps_with_boundaries[0])
    elif len(comps_with_boundaries) == 2:
        merged_compound = _merge_two_compounds(
            comps_with_boundaries[0], comps_with_boundaries[1]
        )

    if merged_compound is None:
        raise NotImplementedError(
            "Merging {} compounds is not supported.".format(len(comps_with_boundaries))
        )

    merged_compound.rules = removed_rules + merged_compound.rules
    for c in comps_without_boundaries:
        merged_compound.concat(c)
    return merged_compound
