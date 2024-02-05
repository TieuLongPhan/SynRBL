import os
import json
import argparse
import pandas as pd
from SynRBL.SynUtils.chem_utils import normalize_smiles
from SynRBL.rsmi_utils import load_database

_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
_FINAL_VALIDATION_PATH = os.path.join(
    _PATH, "Pipeline/Validation/Analysis/final_validation.csv"
)
_DATASET_PATH_FMT = os.path.join(_PATH, "Data/Validation_set/{}/MCS/MCS_Impute.json.gz")
_SNAPSHOT_PATH = os.path.join(_PATH, "Data/Validation_set/snapshot.json")
_DATASETS = [
    "Jaworski",
    "golden_dataset",
    "USPTO_unbalance_class",
    "USPTO_random_class",
    "USPTO_diff",
]


def load_data(dataset):
    data = load_database(os.path.abspath(_DATASET_PATH_FMT.format(dataset)))
    df = pd.read_csv(_FINAL_VALIDATION_PATH)
    with open(_SNAPSHOT_PATH, "r") as f:
        snapshot = json.load(f)
    return data, df, snapshot


def verify_dataset(dataset):
    def _fmt(id, initial_r, result_r, correct_r=None, checked_r=None):
        return {
            "initial_reaction": initial_r,
            "result_reaction": result_r,
            "correct_reaction": correct_r,
            "checked_reaction": checked_r,
            "R-id": id,
        }

    wrong_rxn, unknown_rxn = [], []
    rxn_cnt = 0
    success_cnt = 0
    correct_cnt = 0
    data, df, snapshot = load_data(dataset)

    for item in data:
        id = item["R-id"]
        df_index = df.index[df["R-id"] == id].to_list()
        if len(df_index) == 0:
            continue
        assert len(df_index) == 1
        sn_item = snapshot[id]
        assert id in snapshot.keys(), "Id not in snapshot."
        df_index = df_index[0]
        df_row = df.iloc[df_index]
        is_correct = df_row["Result"]
        initial_reaction = df_row["reactions"]
        result_reaction = item["new_reaction"]
        if item["issue"] == "":
            success_cnt += 1
        result_reaction_n = normalize_smiles(result_reaction)
        if is_correct:
            correct_reaction = df_row["correct_reaction"]
            if result_reaction_n != normalize_smiles(correct_reaction):
                wrong_rxn.append(
                    _fmt(
                        id,
                        initial_reaction,
                        result_reaction,
                        correct_r=correct_reaction,
                    )
                )
            else:
                correct_cnt += 1
        else:
            wrong_reactions = sn_item["wrong_reactions"]
            wrong_reactions_n = [normalize_smiles(r) for r in wrong_reactions]
            if result_reaction_n not in wrong_reactions_n:
                unknown_rxn.append(
                    _fmt(
                        id,
                        initial_reaction,
                        result_reaction,
                        checked_r=wrong_reactions[0],
                    )
                )
        rxn_cnt += 1
    return {
        "wrong_reactions": wrong_rxn,
        "unknown_reactions": unknown_rxn,
        "reaction_cnt": rxn_cnt,
        "success_cnt": success_cnt,
        "correct_cnt": correct_cnt
    }


def verify_datasets(dataset_name=None):
    results = {} 
    for dataset in _DATASETS:
        if dataset_name is not None and dataset_name.lower() != dataset.lower():
            continue
        results[dataset] = verify_dataset(dataset)
    return results

def print_result_table(results):
    line_fmt = "{:<25} {:>12} {:>12} {:>12} {:>12}"
    cols = ["Dataset", "Reactions", "C balanced", "Succ. Rate", "Accuracy"]
    head_line = line_fmt.format(*cols)
    print("=" * len(head_line))
    print(head_line)
    print("-" * len(head_line))
    for db, result in results.items():
        rxn_cnt = result["reaction_cnt"]
        success_cnt = result["success_cnt"]
        correct_cnt = result["correct_cnt"]
        success_rate_str = "{:.2%}".format(success_cnt / rxn_cnt)
        values = [db, rxn_cnt, success_cnt, success_rate_str]
        values.append("{:.2%}".format(correct_cnt / success_cnt))
        print(line_fmt.format(*values))
    print("-" * len(head_line))


def print_verification_result(results):
    good = True
    wrong_rxns, unknown_rxns = [], []
    reaction_cnt = 0
    for db, r in results.items():
        wrong_rxns.extend(r["wrong_reactions"])
        unknown_rxns.extend(r["unknown_reactions"])
        reaction_cnt += r["reaction_cnt"]
    print("[INFO] Checked {} reactions.".format(reaction_cnt))
    if len(wrong_rxns) > 0:
        print(
            "[WARNING] {} reactions that were correct are now wrong!".format(
                len(wrong_rxns)
            )
        )
        good = False
    if len(unknown_rxns) > 0:
        print(
            (
                "[INFO] {} reactions that were wrong changed. "
                + "Please check if they are correct now."
            ).format(len(unknown_rxns))
        )
        good = False
    if good:
        print("[INFO] All good!")


def run_test(args):
    results = verify_datasets(args.dataset)
    print_result_table(results)
    print_verification_result(results)


def configure_argparser(argparser: argparse._SubParsersAction):
    test_parser = argparser.add_parser(
        "test", description="Test success rate and accuracy of SynRBL."
    )

    test_parser.add_argument(
        "--dataset", default=None, help="Use a specific dataset for testing."
    )

    test_parser.set_defaults(func=run_test)
