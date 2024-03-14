import os
import json
import copy
import pickle
import argparse
import hashlib
import logging
import pandas as pd
import numpy as np

from SynRBL.SynAnalysis.analysis_utils import (
    calculate_chemical_properties,
    count_boundary_atoms_products_and_calculate_changes,
)

from SynRBL import SynRBL

_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
_TMP_DIR = os.path.join(_PATH, "tmp")
_CONFIDENCE_MODEL_PATH = os.path.join(_PATH, "Data/scoring_function.pkl")
_SRC_FILE = None
_HASH_KEY = None
_CACHE_ENA = True
_ID_COL = "__synrbl_id"

_CACHE_KEYS = [
    "raw",
    "reactions",
    "rule_based",
    "rule_based_input",
    "rule_based_unsolved",
    "mcs",
    "mcs_based",
    "confidence",
    "output",
]

logger = logging.getLogger(__name__)


def get_hash(file):
    sha1 = hashlib.sha1()
    with open(file, "rb") as f:
        while True:
            data = f.read(65535)
            if not data:
                break
            sha1.update(data)
    return sha1.hexdigest()


def get_cache_file():
    file_name = "{}.cache".format(_HASH_KEY)
    if not os.path.exists(_TMP_DIR):
        os.makedirs(_TMP_DIR)
    return os.path.join(_TMP_DIR, file_name)


def write_cache(data):
    file = get_cache_file()
    _data = {}
    if os.path.exists(file):
        _data = read_cache()
    for k, v in data.items():
        assert k in _CACHE_KEYS, "'{}' is not a valid cache key.".format(k)
        _data[k] = v
    with open(file, "w+") as f:
        json.dump(data, f)


def is_cached(hash_key):
    if not _CACHE_ENA or not os.path.exists(get_cache_file()):
        return False
    _data = read_cache()
    return hash_key in _data.keys() and _data[hash_key] is not None


def read_cache(key=None):
    with open(get_cache_file(), "r") as f:
        _data = json.load(f)
    if key is not None and key not in _data.keys():
        raise RuntimeError(
            (
                "[ERROR] Tried to load '{}' from cache but this data was not found. "
                + "If you run individual steps make sure that all previous steps "
                + "are present in cache. "
                + "(Order of steps: --rule-based, --mcs, --mcs-based)"
            ).format(key)
        )
    if key is not None:
        print("[INFO] Load '{}' from cache.".format(key))
    return _data


def print_dataset_stats(data):
    reactions = data["reactions"]
    balance_cnt = len(
        filter_data(
            reactions,
            unbalance_values=["Balance"],
            formula_key="Diff_formula",
            element_key=None,
            min_count=0,
            max_count=0,
        )
    )
    print("[INFO] " + "=" * 50)
    print(
        "[INFO] Dataset contains {} reactions.".format(
            len(reactions),
        )
    )
    print("[INFO] {} reactions are already balanced.".format(balance_cnt))
    output = data["output"]
    r_suc_c, mcs_suc_c = 0, 0
    for r in output:
        solved_by = r["solved_by"]
        if solved_by == "rule-based-method":
            r_suc_c += 1
        elif solved_by == "mcs-based-method":
            mcs_suc_c += 1
    r_suc = 0
    if "rule_based_input" in data.keys() and len(data["rule_based_input"]) > 0:
        r_suc = r_suc_c / len(data["rule_based_input"])
    print(
        "[INFO] Rule-based method solved {} reactions. (Success rate: {:.2%})".format(
            r_suc_c, r_suc
        )
    )
    mcs_suc = 0
    if "mcs" in data.keys() and len(data["mcs"]) > 0:
        mcs_suc = mcs_suc_c / len(data["mcs"])
    print(
        "[INFO] MCS-based method solved {} reactions. (Success rate: {:.2%})".format(
            mcs_suc_c, mcs_suc
        )
    )
    blnc_c = r_suc_c + mcs_suc_c + balance_cnt
    print(
        "[INFO] Overall balanced reactions: {} ({:.2%})".format(
            (blnc_c), (blnc_c) / len(reactions)
        )
    )


def compoute_confidence(data, col, scoring_function_path: str):
    # Load and process merge data
    merge_data = copy.deepcopy(data["mcs_based"])
    merge_data = count_boundary_atoms_products_and_calculate_changes(merge_data)

    # Load and process MCS data
    mcs_data = copy.deepcopy(data["rule_based_unsolved"])
    ids = [value[_ID_COL] for value in merge_data]
    mcs_data = [value for value in mcs_data if value[_ID_COL] in ids]
    mcs_data = calculate_chemical_properties(mcs_data)

    # Combine data and filter if necessary
    combined_data = pd.concat(
        [
            pd.DataFrame(mcs_data)[
                [
                    _ID_COL,
                    col,
                    "carbon_difference",
                    "fragment_count",
                    "total_carbons",
                    "total_bonds",
                    "total_rings",
                ]
            ],
            pd.DataFrame(merge_data)[
                [
                    "mcs_carbon_balanced",
                    "num_boundary",
                    "ring_change_merge",
                    "bond_change_merge",
                    "new_reaction",
                ]
            ],
        ],
        axis=1,
    )

    combined_data = combined_data.reset_index(drop=True)
    unnamed_columns = [col for col in combined_data.columns if "Unnamed" in col]
    combined_data = combined_data.drop(unnamed_columns, axis=1)

    # Prepare data for prediction
    X_pred = combined_data[
        [
            "carbon_difference",
            "fragment_count",
            "total_carbons",
            "total_bonds",
            "total_rings",
            "num_boundary",
            "ring_change_merge",
            "bond_change_merge",
        ]
    ]

    # Load model and predict confidence
    with open(scoring_function_path, "rb") as file:
        loaded_model = pickle.load(file)

    confidence = np.round(loaded_model.predict_proba(X_pred)[:, 1], 3)
    combined_data["confidence"] = confidence
    data["confidence"] = combined_data.to_dict("records")
    return data


def generate_output(reactions, reaction_col, cols=[], min_confidence: float = 0):
    def _row(
        reaction_col,
        initial_reaction,
        cols: list,
        values: list,
        new_reaction=None,
        solved_by=None,
        confidence=None,
        applied_rules=[],
        issue=None,
        solved=False,
    ):
        if new_reaction is None:
            new_reaction = initial_reaction
        row = {}
        assert len(cols) == len(values)
        for c, v in zip(cols, values):
            row[c] = v
        row[reaction_col] = initial_reaction
        row["new_reaction"] = new_reaction
        row["solved"] = solved
        row["solved_by"] = solved_by
        row["confidence"] = confidence
        row["modifiers"] = applied_rules
        row["issue"] = issue
        return row

    rule_based_result = (
        reactions["rule_based"] if "rule_based" in reactions.keys() else []
    )
    mcs_based_result = reactions["mcs_based"] if "mcs_based" in reactions.keys() else []
    confidence_result = (
        reactions["confidence"] if "confidence" in reactions.keys() else []
    )

    balanced_reactions = filter_data(
        reactions["reactions"],
        unbalance_values=["Balance"],
        formula_key="Diff_formula",
        element_key=None,
        min_count=0,
        max_count=0,
    )
    result_map = {}
    for r in balanced_reactions:
        idx_id = r[_ID_COL]
        assert idx_id not in result_map.keys()
        result_map[idx_id] = {"solved_by": "was-balanced", "result": r}
    for r in rule_based_result:
        idx_id = r[_ID_COL]
        assert idx_id not in result_map.keys()
        result_map[idx_id] = {"solved_by": "rule-based-method", "result": r}

    feature_lookup = {}
    for r in confidence_result:
        idx_id = r[_ID_COL]
        assert idx_id not in feature_lookup.keys()
        feature_lookup[idx_id] = r

    for r in mcs_based_result:
        idx_id = r[_ID_COL]
        assert idx_id not in result_map.keys()
        confidence_item = feature_lookup.get(idx_id, None)
        confidence = ""
        is_solved = True
        if confidence_item is not None and r["solved"]:
            v = confidence_item["confidence"]
            confidence = "{:.2%}".format(v)
            is_solved = v >= min_confidence
        if is_solved:
            result_map[idx_id] = {
                "solved_by": "mcs-based-method",
                "result": r,
                "confidence": confidence,
            }

    output = []
    assert len(reactions["raw"]) == len(reactions["reactions"])
    for src_item, item in zip(reactions["raw"], reactions["reactions"]):
        values = []
        for c in cols:
            if c not in src_item.keys():
                raise ValueError(
                    "No column named '{}' found in the input file.".format(c)
                )
            values.append(src_item[c])
        synrbl_id = item[_ID_COL]
        initial_reaction = src_item[reaction_col]
        assert initial_reaction == item[reaction_col]
        if synrbl_id in result_map.keys():
            map_item = result_map[synrbl_id]
            solved_by = map_item["solved_by"]
            result = map_item["result"]
            if solved_by == "was-balanced":
                output.append(
                    _row(
                        reaction_col,
                        initial_reaction,
                        cols,
                        values,
                        new_reaction=initial_reaction,
                        solved_by=solved_by,
                        solved=True,
                    )
                )
            elif solved_by == "rule-based-method":
                output.append(
                    _row(
                        reaction_col,
                        initial_reaction,
                        cols,
                        values,
                        new_reaction=result["new_reaction"],
                        solved_by=solved_by,
                        solved=True,
                    )
                )
            elif solved_by == "mcs-based-method":
                issue = result["issue"]
                solved = result["solved"]
                if solved:
                    mcs_carbon_balanced = result["mcs_carbon_balanced"]
                    assert mcs_carbon_balanced and issue == ""
                output.append(
                    _row(
                        reaction_col,
                        initial_reaction,
                        cols,
                        values,
                        new_reaction=result["new_reaction"],
                        solved_by=solved_by if solved else None,
                        applied_rules=result["rules"],
                        issue=issue,
                        solved=solved,
                        confidence=map_item["confidence"],
                    )
                )
            else:
                raise NotImplementedError()
        else:
            output.append(_row(reaction_col, initial_reaction, cols, values))
    reactions["output"] = output
    return reactions


def impute(
    src_file,
    output_file,
    reaction_col,
    tmp_dir,
    passthrough_cols,
    min_confidence,
):
    if tmp_dir is not None:
        tmp_dir = os.path.abspath(tmp_dir)

    input_reactions = pd.read_csv(src_file).to_dict('records')
    
    synrbl = SynRBL(reaction_col=reaction_col)
    rbl_reactions = synrbl.rebalance(input_reactions, output_dict=True)
    
    for in_r, out_r in zip(input_reactions, rbl_reactions):
        for c in passthrough_cols:
            out_r[c] = in_r[c]

    df = pd.DataFrame(rbl_reactions)
    df.to_csv(output_file)


def run(args):
    columns = args.columns if isinstance(args.columns, list) else [args.columns]

    impute(
        args.filename,
        args.o,
        reaction_col=args.col,
        passthrough_cols=columns,
        tmp_dir=args.tmp_dir,
        min_confidence=args.min_confidence,
    )


class Range(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def __eq__(self, other):
        return self.start <= other <= self.end

    def __str__(self):
        return "[{}, {}]".format(self.start, self.end)


def configure_argparser(argparser: argparse._SubParsersAction):
    test_parser = argparser.add_parser(
        "run", description="Try to rebalance chemical reactions."
    )

    test_parser.add_argument(
        "filename", help="Path to file containing reaction SMILES."
    )

    test_parser.add_argument(
        "-o", default="SynRBL_results.csv", help="Path to results file."
    )

    test_parser.add_argument(
        "--tmp-dir",
        default="./tmp",
        help="Path where SynRBL stores intermediate results.",
    )
    test_parser.add_argument(
        "--col",
        default="reaction",
        help="The reactions column name for in the input .csv file. (Default: 'reaction')",
    )
    test_parser.add_argument(
        "--columns",
        default=[],
        help="A list of columns from the input that should be added to the output.",
    )
    test_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0,
        choices=[Range(0.0, 1.0)],
        help=(
            "Set a confidence threshold for the results "
            + "from the MCS-based method. (Default: 0.5)"
        ),
    )

    test_parser.set_defaults(func=run)
