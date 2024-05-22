from synrbl.SynMCSImputer.structure import CompoundSet
from synrbl.SynMCSImputer.utils import is_carbon_balanced
from synrbl.SynMCSImputer.merge import merge
from rdkit.rdBase import BlockLogs


def build_compounds(data_dict) -> CompoundSet:
    src_smiles = data_dict["sorted_reactants"]
    smiles = data_dict["smiles"]
    boundaries = data_dict["boundary_atoms_products"]
    neighbors = data_dict["nearest_neighbor_products"]
    mcs_results = data_dict["mcs_results"]
    n = len(smiles)
    if n != len(src_smiles):
        raise ValueError(
            "Smiles and sorted reactants are not of the same length. ({} != {})".format(
                len(smiles), len(src_smiles)
            )
        )
    if n != len(boundaries) or n != len(neighbors):
        raise ValueError(
            "Boundaries and nearest neighbors must be of same length as compounds."
        )
    if n != len(mcs_results):
        raise ValueError("MCS results must be of same length as compounds.")
    cset = CompoundSet()
    for s, ss, b, n, mcs in zip(smiles, src_smiles, boundaries, neighbors, mcs_results):
        if s is None:
            if mcs == "":
                # TODO use compound rule for that
                if ss == "O":
                    # water is not catalyst -> binds to other compound
                    c = cset.add_compound(ss, src_mol=ss)
                    c.add_boundary(0, symbol="O")
                else:
                    # catalysis compound
                    c = cset.add_compound(ss, src_mol=ss)
            else:
                # empty compound
                pass
        else:
            c = cset.add_compound(s, src_mol=ss)
            if len(b) != len(n):
                raise ValueError(
                    (
                        "Boundary and neighbor missmatch. "
                        + "(boundary={}, neighbor={})"
                    ).format(b, n)
                )
            for bi, ni in zip(b, n):
                bi_s, bi_i = list(bi.items())[0]
                ni_s, ni_i = list(ni.items())[0]
                c.add_boundary(
                    bi_i, symbol=bi_s, neighbor_index=ni_i, neighbor_symbol=ni_s
                )
    return cset


def impute_reaction(
    reaction_dict,
    reaction_col,
    issue_col,
    carbon_balance_col,
    mcs_data_col,
    smiles_standardizer=[],
):
    issue = reaction_dict[issue_col] if issue_col in reaction_dict.keys() else ""
    if issue != "":
        raise ValueError("Skip reaction because of previous issue.\n" + issue)
    compound_set = build_compounds(reaction_dict[mcs_data_col])
    if len(compound_set) == 0:
        raise ValueError("Empty compound set.")
    merge_result = merge(compound_set)
    carbon_balance = reaction_dict[carbon_balance_col]
    if carbon_balance == "reactants":
        # Imputing reactant side carbon imbalance is not (yet) supported
        raise ValueError("Skipped because of reactants imbalance.")
    elif carbon_balance in ["products", "balanced"]:
        merged_smiles = merge_result.smiles
        for standardizer in smiles_standardizer:
            merged_smiles = standardizer(merged_smiles)
        imputed_reaction = "{}.{}".format(reaction_dict[reaction_col], merged_smiles)
    else:
        raise ValueError(
            "Invalid value '{}' for carbon balance.".format(carbon_balance)
        )
    rules = [r.name for r in merge_result.rules]
    is_balanced = is_carbon_balanced(imputed_reaction)
    if not is_balanced:
        raise RuntimeError(
            (
                "Failed to impute the correct structure. "
                + "Carbon atom count in reactants and products does not match."
            )
        )
    return imputed_reaction, rules


class MCSBasedMethod:
    def __init__(
        self,
        reaction_col,
        output_col,
        mcs_data_col="mcs",
        issue_col="issue",
        rules_col="rules",
        carbon_balance_col="carbon_balance_check",
        smiles_standardizer=[],
    ):
        self.reaction_col = reaction_col
        self.output_col = output_col if isinstance(output_col, list) else [output_col]
        self.mcs_data_col = mcs_data_col
        self.issue_col = issue_col
        self.rules_col = rules_col
        self.carbon_balance_col = carbon_balance_col
        self.smiles_standardizer = smiles_standardizer

    def run(self, reactions: list[dict], stats=None):
        mcs_applied = 0
        mcs_solved = 0
        block_logs = BlockLogs()
        for reaction in reactions:
            if self.mcs_data_col not in reaction.keys():
                continue
            mcs_applied += 1
            if reaction[self.mcs_data_col] is None:
                continue
            try:
                result, rules = impute_reaction(
                    reaction,
                    mcs_data_col=self.mcs_data_col,
                    reaction_col=self.reaction_col,
                    issue_col=self.issue_col,
                    carbon_balance_col=self.carbon_balance_col,
                    smiles_standardizer=self.smiles_standardizer,
                )
                for col in self.output_col:
                    reaction[col] = result
                reaction[self.rules_col] = rules
                mcs_solved += 1
            except Exception as e:
                reaction[self.issue_col] = str(e)

        del block_logs
        if stats is not None:
            stats["mcs_applied"] = mcs_applied
            stats["mcs_solved"] = mcs_solved
        return reactions
