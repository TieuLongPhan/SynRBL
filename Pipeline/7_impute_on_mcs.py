import argparse
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCS.structure import Compound
from SynRBL.SynMCS.rules import MergeRule, CompoundRule
from SynRBL.SynMCS.merge import merge


def build_compounds(item):
    src_smiles = item["sorted_reactants"]
    smiles = item["smiles"]
    boundaries = item["boundary_atoms_products"]
    neighbors = item["nearest_neighbor_products"]
    l = len(smiles)
    if len(boundaries) != len(neighbors) or l != len(src_smiles):
        print(smiles, src_smiles, boundaries, neighbors)
        raise ValueError("Unequal leghts.")
    compounds = []
    s_i = 0
    for s, ss in zip(smiles, src_smiles):
        if s is None:
            continue
        b = boundaries[s_i]
        n = neighbors[s_i]
        c = Compound(s, src_mol=ss)
        if len(b) != len(n):
            raise ValueError("Boundary and neighbor missmatch.")
        for bi, ni in zip(b, n):
            bi_s, bi_i = list(bi.items())[0]
            ni_s, ni_i = list(ni.items())[0]
            c.add_boundary(bi_i, symbol=bi_s, neighbor_index=ni_i, neighbor_symbol=ni_s)
        compounds.append(c)
        s_i += 1
    if len(boundaries) != s_i:
        raise ValueError("Compounds do not match boundaries and neighbors.")
    return compounds


def impute_new_reaction(data):
    rule_map = {r.name: set() for r in CompoundRule.get_all() + MergeRule.get_all()}
    rule_map["no rule"] = set()
    for i, item in enumerate(data):
        if data[i]["issue"] != "":
            print("[ERROR] [{}]".format(i), "Skip because of previous issue.")
            continue
        data[i]["rules"] = []
        data[i]["new_reaction"] = data[i]["old_reaction"]
        try:
            compounds = build_compounds(item)
            result = merge(compounds)
            new_reaction = "{}.{}".format(item["old_reaction"], result.smiles)
            data[i]["new_reaction"] = new_reaction
            rules = [r.name for r in result.rules]
            data[i]["rules"] = rules
            if len(rules) == 0:
                rule_map["no rule"].add(i)
            else:
                for r in data[i]["rules"]:
                    rule_map[r].add(i)
        except Exception as e:
            # traceback.print_exc()
            data[i]["issue"] = str(e)
            print("[ERROR] [{}]".format(i), e)
    return rule_map


def print_rule_summary(rule_map):
    for rule, ids in rule_map.items():
        print("{:<30} {}".format(rule, len(ids)))


def get_database_path(dataset, name):
    return "./Data/Validation_set/{}/MCS/{}.json.gz".format(dataset, name)


if __name__ == "__main__":
    dataset = "USPTO_test"
    parser = argparse.ArgumentParser(
        prog="MCSImpute", description="Imputes new compounds based on the MCS results."
    )
    data = load_database(get_database_path(dataset, "Final_Graph"))
    rule_map = impute_new_reaction(data)
    save_database(data, get_database_path(dataset, "MCS_Impute"))
    print_rule_summary(rule_map)
