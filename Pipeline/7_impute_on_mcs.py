import argparse

from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCSImputer.rules import MergeRule, CompoundRule
from SynRBL.SynMCSImputer.model import MCSImputer


def print_rule_summary(rule_map):
    line_fmt = "  {:<30} {:4} {:>5}"
    header = line_fmt.format("Rule", "Type", "#")
    print("{}".format("=" * len(header)))
    print(header)
    print("{}".format("-" * len(header)))
    merge_rules = [r.name for r in MergeRule.get_all()]
    compound_rules = [r.name for r in CompoundRule.get_all()]
    for rule, ids in rule_map.items():
        rule_type = ""
        if rule in merge_rules:
            rule_type = "MR"
        elif rule in compound_rules:
            rule_type = "CR"
        print(line_fmt.format(rule, rule_type, len(ids)))


def print_success_rate(dataset):
    success_cnt = 0
    for item in dataset:
        if item["issue"] == "":
            success_cnt += 1
    print(
        "Reached carbon balance on {:.2%} ({}) of the reactions.".format(
            success_cnt / len(dataset),
            success_cnt,
        )
    )


def impute_new_reactions(data):
    imputer = MCSImputer()
    rule_map = {r.name: set() for r in CompoundRule.get_all() + MergeRule.get_all()}
    rule_map["no rule"] = set()
    for i, item in enumerate(data):
        imputer.impute_reaction(item)
        issue = item["issue"]
        if issue != "":
            print("[ERROR] [{}] {}".format(i, issue))
        rules = item["rules"]
        if len(rules) > 0:
            for r in rules:
                rule_map[r].add(i)
        else:
            rule_map["no rule"].add(i)
    return rule_map


def get_database_path(dataset, name):
    return "./Data/Validation_set/{}/MCS/{}.json.gz".format(dataset, name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MCSImpute", description="Imputes new compounds based on the MCS results."
    )
    parser.add_argument(
        "--dataset",
        default="USPTO_test",
        help="The name of the dataset directory in ./Data/Validation_set/",
    )
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    data = load_database(get_database_path(args.dataset, "Final_Graph"))
    rule_map = impute_new_reactions(data)
    save_database(data, get_database_path(args.dataset, "MCS_Impute"))
    print_rule_summary(rule_map)
    print_success_rate(data)
