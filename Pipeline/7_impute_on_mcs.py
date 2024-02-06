import os
import argparse
import collections

from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCSImputer.rules import MergeRule, ExpandRule
from SynRBL.SynMCSImputer.model import MCSImputer
from SynRBL.SynUtils.chem_utils import normalize_smiles

DATABASE_PATH = "./Data/Validation_set/"


def print_rule_summary(rule_map):
    line_fmt = "  {:<30} {:4} {:>5}"
    header = line_fmt.format("Rule", "Type", "#")
    print("{}".format("=" * len(header)))
    print(header)
    print("{}".format("-" * len(header)))
    merge_rules = [r.name for r in MergeRule.get_all()]
    compound_rules = [r.name for r in ExpandRule.get_all()]
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


def impute_new_reactions(data, verbose=True):
    imputer = MCSImputer(cs_passthrough=True)
    rule_map = {r.name: set() for r in ExpandRule.get_all() + MergeRule.get_all()}
    rule_map["no rule"] = set()
    for i, item in enumerate(data):
        imputer.impute_reaction(item)
        issue = item["issue"]
        if issue != "" and verbose:
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


def get_validation_set_path(dataset):
    return os.path.join(DATABASE_PATH, dataset, "corrected.json.gz")


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
    params = {}
    if args.dataset is None:
        for db in get_databases():
            print("Impute {}".format(db))
            data = load_database(get_database_path(db, "Final_Graph"))
            rule_map = impute_new_reactions(data, verbose=False, **params)
            save_database(data, get_database_path(db, "MCS_Impute"))
            print_success_rate(data)
    else:
        data = load_database(get_database_path(args.dataset, "Final_Graph"))
        rule_map = impute_new_reactions(data, **params)
        save_database(data, get_database_path(args.dataset, "MCS_Impute"))
        print_rule_summary(rule_map)
        print_success_rate(data)


def run_report(args):
    dbs = get_databases()
    line_fmt = "{:<25} {:>12} {:>12} {:>9}"
    cols = ["Dataset", "Reactions", "C balanced", "Rate"]
    if args.validate:
        line_fmt += " {:>12}"
        cols.append("Accuracy")
    header = line_fmt.format(*cols)
    rows = []
    for db in dbs:
        path = get_database_path(db, "MCS_Impute")
        if not os.path.exists(path):
            raise ValueError(
                (
                    "[ERROR] No MCS_Impute data found for dataset '{}'. "
                    + "Run 'impute --dataset {}' first."
                ).format(db, db)
            )
        data = load_database(path)
        val_data = None
        if args.validate:
            val_set_path = get_validation_set_path(db)
            if os.path.exists(val_set_path):
                val_data = load_database(val_set_path)
                if len(data) != len(val_data):
                    print(
                        "Data and val_data are not of same length. (Dataset: {}, {} != {})".format(
                            db, len(data), len(val_data)
                        )
                    )
                    val_data = None
        success_cnt = 0
        correct_cnt = None
        for i, item in enumerate(data):
            if item["issue"] == "":
                success_cnt += 1
                if val_data is not None:
                    val_item = val_data[i]
                    assert (
                        item["R-id"] == val_item["R-id"]
                    ), "R-id at index {} does not match.".format(i)
                    if (
                        normalize_smiles(item["new_reaction"])
                        == val_item["correct_reaction"]
                    ):
                        correct_cnt = 1 if correct_cnt is None else correct_cnt + 1
        rate_str = "{:.2%}".format(success_cnt / len(data))
        values = [db, len(data), success_cnt, rate_str]
        if args.validate:
            if correct_cnt is None:
                values.append("-")
            else:
                values.append("{:.2%}".format(correct_cnt / success_cnt))
        rows.append(line_fmt.format(*values))
    print(header)
    print("-" * len(header))
    for r in rows:
        print(r)


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
        default=None,
        help="The name of the dataset directory in ./Data/Validation_set/",
    )
    #impute_parser.add_argument(
    #    "--cs-passthrough",
    #    action="store_true",
    #    help="Flag if catalysts and solvents should be passed through.",
    #)
    impute_parser.set_defaults(func=run_impute)

    report_parser = subparsers.add_parser(
        "report", description="Print summary on MCS imputation performance."
    )
    report_parser.add_argument(
        "--validate",
        action="store_true",
        help="Flag to use the corrected.json.gz datasets for validation. (Reports accuracy)",
    )
    report_parser.set_defaults(func=run_report)

    args = parser.parse_args()
    args.func(args)
