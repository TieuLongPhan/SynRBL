import argparse
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCS.rules import MergeRule, CompoundRule
from SynRBL.SynMCS.merge import merge
import SynRBL.SynMCS.structure as structure


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
            compounds = structure.build_compounds(item)
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
    print("{}".format("=" * 40))
    print("  {:<30} {}".format("Rule", "#"))
    print("{}".format("-" * 40))
    for rule, ids in rule_map.items():
        print("  {:<30} {}".format(rule, len(ids)))


def get_database_path(dataset, name):
    return "./Data/Validation_set/{}/MCS/{}.json.gz".format(dataset, name)


if __name__ == "__main__":
    dataset = "USPTO_test"
    parser = argparse.ArgumentParser(
        prog="MCSImpute", description="Imputes new compounds based on the MCS results."
    )
    parser.add_argument(
        "--dataset",
        default="USPTO_test",
        help="The name of the dataset directory in ./Data/Validation_set/",
    )
    parser.add_argument("--summary", action='store_true')
    args = parser.parse_args()
    data = load_database(get_database_path(args.dataset, "Final_Graph"))
    rule_map = impute_new_reaction(data)
    save_database(data, get_database_path(dataset, "MCS_Impute"))
    if args.summary:
        print_rule_summary(rule_map)
