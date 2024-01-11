from SynRBL.SynMCSImputer.structure import Compound
from SynRBL.SynMCSImputer.utils import is_carbon_balanced
from SynRBL.SynMCSImputer.merge import merge


def build_compounds(data_dict) -> list[Compound]:
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
    compounds = []
    for s, ss, b, n, mcs in zip(smiles, src_smiles, boundaries, neighbors, mcs_results):
        c = None
        if s is None:
            if mcs == "":
                if ss == "O":
                    # water is not catalyst -> binds to other compound
                    c = Compound(ss, src_mol=ss)
                    c.add_boundary(0, symbol="O")
                else:
                    # catalysis compound
                    c = Compound(ss, src_mol=ss)
            else:
                # empty compound
                pass
        else:
            c = Compound(s, src_mol=ss)
            if len(b) != len(n):
                raise ValueError(
                    "Boundary and neighbor missmatch. (boundary={}, neighbor={})".format(
                        b, n
                    )
                )
            for bi, ni in zip(b, n):
                bi_s, bi_i = list(bi.items())[0]
                ni_s, ni_i = list(ni.items())[0]
                c.add_boundary(
                    bi_i, symbol=bi_s, neighbor_index=ni_i, neighbor_symbol=ni_s
                )
        if c is not None:
            compounds.append(c)
    return compounds


def impute_reaction(reaction_dict):
    reaction_dict["rules"] = []
    new_reaction = reaction_dict["old_reaction"]
    reaction_dict["mcs_carbon_balanced"] = reaction_dict["carbon_balance_check"] == 'balanced'
    try:
        if reaction_dict["issue"] != "":
            raise ValueError(
                "Skip reaction because of previous issue.\n" + reaction_dict["issue"]
            )
        compounds = build_compounds(reaction_dict)
        if len(compounds) == 0:
            return
        result = merge(compounds)
        carbon_balance = reaction_dict["carbon_balance_check"]
        if carbon_balance == "reactants":
            imputed_reaction = "{}.{}".format(
               result.smiles, reaction_dict["old_reaction"]
            )
            # imputed_reaction = reaction_dict["old_reaction"]
        elif carbon_balance in ["products", "balanced"]:
            imputed_reaction = "{}.{}".format(
                reaction_dict["old_reaction"], result.smiles
            )
        else:
            raise ValueError(
                "Invalid value '{}' for carbon balance.".format(carbon_balance)
            )
        rules = [r.name for r in result.rules]
        is_balanced = is_carbon_balanced(imputed_reaction)
        reaction_dict["mcs_carbon_balanced"] = is_balanced
        if not is_balanced:
            raise RuntimeError(
                (
                    "Failed to impute the correct structure. "
                    + "Carbon atom count in reactants and products does not match. "
                )
            )
        new_reaction = imputed_reaction
        reaction_dict["rules"] = rules
    except Exception as e:
        reaction_dict["issue"] = str(e)
    finally:
        reaction_dict["new_reaction"] = new_reaction


class MCSImputer:
    def __init__(self):
        pass

    @staticmethod
    def impute_reaction(reaction_dict):
        impute_reaction(reaction_dict)
