import os
import argparse
import collections

from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCSImputer.rules import MergeRule, CompoundRule
from SynRBL.SynMCSImputer.model import MCSImputer

DATABASE_PATH = "./Data/Validation_set/"


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
        "Reached carbon balance on {:.2%} ({}/{}) of the reactions.".format(
            success_cnt / len(dataset), success_cnt, len(dataset)
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
    return os.path.join(DATABASE_PATH, dataset, "MCS", "{}.json.gz".format(name))


def get_databases():
    dbs = set()
    dir_names = [f.name for f in os.scandir(DATABASE_PATH) if f.is_dir()]
    for d in dir_names:
        if os.path.exists(os.path.join(DATABASE_PATH, d, "MCS/Final_Graph.json.gz")):
            dbs.add(d)
        else:
            print("[ERROR] Directory '{}' is not a valid database.".format(d))
    return list(dbs)


def run_impute(args):
    data = load_database(get_database_path(args.dataset, "Final_Graph"))
    rule_map = impute_new_reactions(data)
    save_database(data, get_database_path(args.dataset, "MCS_Impute"))
    print_rule_summary(rule_map)
    print_success_rate(data)


def run_report(args):
    dbs = get_databases()
    line_fmt = "{:<25} {:>12} {:>12} {:>9}"
    header = line_fmt.format("Dataset", "Reactions", "C balanced", "Rate")
    print(header)
    print("-" * len(header))
    for db in dbs:
        path = get_database_path(db, "MCS_Impute")
        if not os.path.exists(path):
            print(
                (
                    "[ERROR] No MCS_Impute data found for dataset '{}'. "
                    + "Run 'impute --dataset {}' first."
                ).format(db, db)
            )
        data = load_database(path)
        success_cnt = 0
        for item in data:
            if item["issue"] == "":
                success_cnt += 1
        rate_str = "{:.2%}".format(success_cnt / len(data))
        print(line_fmt.format(db, len(data), success_cnt, rate_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="MCSImpute", description="Imputes new compounds based on the MCS results."
    )
    subparsers = parser.add_subparsers(dest="command")

    impute_parser = subparsers.add_parser(
        "impute", description="Impute compounds from MCS results."
    )
    impute_parser.add_argument(
        "--dataset",
        default="USPTO_test",
        help="The name of the dataset directory in ./Data/Validation_set/",
    )
    impute_parser.set_defaults(func=run_impute)

    report_parser = subparsers.add_parser(
        "report", description="Print summary on MCS imputation performance."
    )
    report_parser.set_defaults(func=run_report)

    args = parser.parse_args()
    args.func(args)
