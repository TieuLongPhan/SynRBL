import os
import json
import copy
import argparse
import hashlib
from numpy import who
import pandas as pd
import rdkit

from SynRBL.SynRuleImputer import SyntheticRuleImputer
from SynRBL.SynRuleImputer.synthetic_rule_constraint import RuleConstraint
from SynRBL.SynProcessor import (
    RSMIProcessing,
    RSMIDecomposer,
    RSMIComparator,
    BothSideReact,
    CheckCarbonBalance,
)
from SynRBL.rsmi_utils import (
    save_database,
    load_database,
    filter_data,
    extract_results_by_key,
)
from SynRBL.SynMCSImputer.SubStructure.mcs_process import ensemble_mcs
from SynRBL.SynUtils.data_utils import load_database, save_database
from SynRBL.SynMCSImputer.SubStructure.extract_common_mcs import ExtractMCS
from SynRBL.SynMCSImputer.MissingGraph.find_graph_dict import find_graph_dict
from SynRBL.SynMCSImputer.model import MCSImputer

_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
_TMP_DIR = os.path.join(_PATH, "tmp")
_SRC_FILE = None
_HASH_KEY = None
_CACHE_ENA = True

_CACHE_KEYS = [
    "reactions",
    "rule_based",
    "rule_based_input",
    "rule_based_unsolved",
    "mcs",
    "mcs_based",
    "output"
]


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
    return os.path.join(_TMP_DIR, file_name)


def write_cache(data):
    if not _CACHE_ENA:
        return
    file = get_cache_file()
    _data = {}
    if os.path.exists(file):
        _data = read_cache()
    for k, v in data.items():
        assert k in _CACHE_KEYS, "'{}' is not a valid cache key.".format(k)
        _data[k] = v
    with open(file, "w") as f:
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
    if "rule_based_input" in data.keys():
        r_suc = r_suc_c / len(data["rule_based_input"])
    print(
        "[INFO] Rule-based method solved {} reactions. (Success rate: {:.2%})".format(
            r_suc_c, r_suc 
        )
    )
    mcs_suc = 0
    if "mcs" in data.keys():
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


def load_reactions(file, reaction_col, n_jobs):
    print("[INFO] Load reactions from file.")
    df = pd.read_csv(file)
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)

    # 1. process data
    process = RSMIProcessing(
        data=df,
        rsmi_col=reaction_col,
        parallel=True,
        n_jobs=n_jobs,
        data_name="index",
        save_json=False,
        save_path_name=None,
    )
    reactions = process.data_splitter().to_dict("records")

    # 2. check carbon balance
    check = CheckCarbonBalance(
        reactions, rsmi_col=reaction_col, symbol=">>", atom_type="C", n_jobs=n_jobs
    )
    reactions = check.check_carbon_balance()

    # 3. decompose into dict of symbols
    decompose = RSMIDecomposer(
        smiles=None,
        data=reactions,
        reactant_col="reactants",
        product_col="products",
        parallel=True,
        n_jobs=n_jobs,
        verbose=1,
    )
    react_dict, product_dict = decompose.data_decomposer()

    # 4. compare dict and check balance
    comp = RSMIComparator(reactants=react_dict, products=product_dict, n_jobs=n_jobs)
    unbalance, diff_formula = comp.run_parallel(
        reactants=react_dict, products=product_dict
    )

    # 5. solve the both side reaction
    both_side = BothSideReact(react_dict, product_dict, unbalance, diff_formula)
    diff_formula, unbalance = both_side.fit(n_jobs=n_jobs)

    reactions_clean = pd.concat(
        [
            pd.DataFrame(reactions),
            pd.DataFrame([unbalance]).T.rename(columns={0: "Unbalance"}),
            pd.DataFrame([diff_formula]).T.rename(columns={0: "Diff_formula"}),
        ],
        axis=1,
    ).to_dict(orient="records")
    # ME save_database(reactions_clean, save_dir / "reactions_clean.json.gz")
    return {"reactions": reactions_clean}


def rule_based_method(data, n_jobs):
    reactions = data["reactions"]
    print("[INFO] Run rule-based method on {} reactions.".format(len(reactions)))
    unbalance_reactions = filter_data(
        reactions,
        unbalance_values=["Reactants", "Products"],
        formula_key="Diff_formula",
        element_key=None,
        min_count=0,
        max_count=0,
    )

    both_side_reactions = filter_data(
        reactions,
        unbalance_values=["Both"],
        formula_key="Diff_formula",
        element_key=None,
        min_count=0,
        max_count=0,
    )

    # Initialize SyntheticRuleImputer and perform parallel imputation
    rules = load_database(os.path.join(_PATH, "Data/Rules/rules_manager.json.gz"))
    imp = SyntheticRuleImputer(rule_dict=rules, select="all", ranking="ion_priority")
    expected_result = imp.parallel_impute(unbalance_reactions, n_jobs=n_jobs)

    solve, unsolve = extract_results_by_key(expected_result)

    unsolve = both_side_reactions + unsolve

    # 8. Handle uncertainty in imputation
    constrain = RuleConstraint(
        solve,
        ban_atoms=["[O].[O]", "F-F", "Cl-Cl", "Br-Br", "I-I", "Cl-Br", "Cl-I", "Br-I"],
    )
    certain_reactions, uncertain_reactions = constrain.fit()

    id_uncertain = [entry["R-id"] for entry in uncertain_reactions]
    new_uncertain_reactions = [
        entry for entry in reactions if entry["R-id"] in id_uncertain
    ]

    unsolve = unsolve + new_uncertain_reactions

    print(
        "[INFO] Rule-based method solved {} of {} reactions.".format(
            len(certain_reactions), len(unbalance_reactions)
        )
    )

    # save_database(mcs_based, save_dir / "mcs_based_reactions.json.gz")
    data["rule_based"] = certain_reactions
    data["rule_based_input"] = unbalance_reactions
    data["rule_based_unsolved"] = unsolve
    return data

    # ME save_database(certain_reactions, save_dir / "rule_based_reactions.json.gz")
    # ME save_database(mcs_based, save_dir / "mcs_based_reactions.json.gz")


def find_mcs(data, col):
    mcs_reactions = data["rule_based_unsolved"]
    print(
        "[INFO] Find maximum-common-substructure for {} reactions.".format(
            len(mcs_reactions)
        )
    )
    conditions = [
        {
            "RingMatchesRingOnly": True,
            "CompleteRingsOnly": True,
            "method": "MCIS",
            "sort": "MCIS",
            "ignore_bond_order": True,
        },
        {
            "RingMatchesRingOnly": True,
            "CompleteRingsOnly": True,
            "method": "MCIS",
            "sort": "MCIS",
            "ignore_bond_order": False,
        },
        {
            "RingMatchesRingOnly": False,
            "CompleteRingsOnly": False,
            "method": "MCIS",
            "sort": "MCIS",
            "ignore_bond_order": True,
        },
        {
            "RingMatchesRingOnly": False,
            "CompleteRingsOnly": False,
            "method": "MCIS",
            "sort": "MCIS",
            "ignore_bond_order": False,
        },
        {"method": "MCES", "sort": "MCES"},
    ]

    # Run and save conditions
    ensemble_mcs(
        mcs_reactions, _PATH, _TMP_DIR, conditions, batch_size=5000, Timeout=90
    )

    condition_1 = load_database(f"{_TMP_DIR}/Condition_1.json.gz")
    condition_2 = load_database(f"{_TMP_DIR}/Condition_2.json.gz")
    condition_3 = load_database(f"{_TMP_DIR}/Condition_3.json.gz")
    condition_4 = load_database(f"{_TMP_DIR}/Condition_4.json.gz")
    condition_5 = load_database(f"{_TMP_DIR}/Condition_5.json.gz")

    analysis = ExtractMCS()
    mcs_dict, threshold_index = analysis.extract_matching_conditions(
        0,
        100,
        condition_1,
        condition_2,
        condition_3,
        condition_4,
        condition_5,
        extraction_method="largest_mcs",
        using_threshold=True,
    )
    save_database(mcs_dict, f"{_TMP_DIR}/MCS_Largest.json.gz")

    # mcs_dir = root_dir / f"Data/Validation_set/{data_name}/MCS"
    # mcs = load_database(mcs_dir / "MCS_Largest.json.gz")
    missing_results_largest = find_graph_dict(
        msc_dict_path=os.path.join(_TMP_DIR, "MCS_Largest.json.gz"),
        save_path=os.path.join(_TMP_DIR, "Final_Graph.json.gz"),
    )
    miss_id = [value["R-id"] for value in mcs_dict]
    data_2 = [
        mcs_reactions[key]
        for key, value in enumerate(mcs_reactions)
        if value["R-id"] in miss_id
    ]
    for key, _ in enumerate(missing_results_largest):
        missing_results_largest[key]["R-id"] = mcs_dict[key]["R-id"]
        missing_results_largest[key]["sorted_reactants"] = mcs_dict[key][
            "sorted_reactants"
        ]
        missing_results_largest[key]["carbon_balance_check"] = mcs_dict[key][
            "carbon_balance_check"
        ]
        missing_results_largest[key]["mcs_results"] = mcs_dict[key]["mcs_results"]
        missing_results_largest[key]["mcs_results"] = mcs_dict[key]["mcs_results"]
        missing_results_largest[key]["old_reaction"] = data_2[key][col]

    save_database(
        missing_results_largest, os.path.join(_TMP_DIR, "Final_Graph.json.gz")
    )
    data["mcs"] = missing_results_largest
    return data


def mcs_based_method(data, cs_passthrough):
    mcs_reactions = copy.deepcopy(data["mcs"])
    print("[INFO] Run MCS-based method on {} reactions.".format(len(mcs_reactions)))
    imputer = MCSImputer(cs_passthrough=cs_passthrough)
    for item in mcs_reactions:
        imputer.impute_reaction(item)
    data["mcs_based"] = mcs_reactions
    return data


def generate_output(reactions, col):
    def _row(
        initial_reaction,
        new_reaction=None,
        solved_by=None,
        confidence=None,
        applied_rules=[],
        issue=None,
    ):
        if new_reaction is None:
            new_reaction = initial_reaction
        return {
            col: initial_reaction,
            "new_reaction": new_reaction,
            "solved_by": solved_by,
            "confidence": confidence,
            "modifiers": applied_rules,
            "issue": issue,
        }

    rule_based_result = (
        reactions["rule_based"] if "rule_based" in reactions.keys() else []
    )
    mcs_based_result = reactions["mcs_based"] if "mcs_based" in reactions.keys() else []

    result_map = {}
    for r in rule_based_result:
        id = r["R-id"]
        assert id not in result_map.keys()
        result_map[id] = {"solved_by": "rule-based-method", "result": r}
    for r in mcs_based_result:
        id = r["R-id"]
        assert id not in result_map.keys()
        result_map[id] = {"solved_by": "mcs-based-method", "result": r}

    output = []
    for r in reactions["reactions"]:
        id = r["R-id"]
        initial_reaction = r["reaction"]
        if id in result_map.keys():
            map_item = result_map[id]
            solved_by = map_item["solved_by"]
            result = map_item["result"]
            if solved_by == "rule-based-method":
                output.append(_row(initial_reaction, result["new_reaction"], solved_by))
            elif solved_by == "mcs-based-method":
                issue = result["issue"]
                mcs_carbon_balanced = result["mcs_carbon_balanced"]
                valid = mcs_carbon_balanced and issue == ""
                if valid:
                    output.append(
                        _row(
                            initial_reaction,
                            result["new_reaction"],
                            solved_by,
                            applied_rules=result["rules"],
                            issue=issue,
                        )
                    )
                else:
                    output.append(_row(initial_reaction))
            else:
                raise NotImplementedError()
        else:
            output.append(_row(initial_reaction))
    reactions["output"] = output
    return reactions


def export(reactions, file):
    output = reactions["output"]
    df = pd.DataFrame(output)
    df.to_csv(file)


def run(args):
    global _TMP_DIR, _SRC_FILE, _HASH_KEY, _CACHE_ENA
    _SRC_FILE = args.filename
    _HASH_KEY = get_hash(_SRC_FILE)
    if args.no_cache:
        _CACHE_ENA = False
        print("[INFO] Caching is disabled.")
    if args.tmp_dir is not None:
        _TMP_DIR = os.path.abspath(args.tmp_dir)
    lg = rdkit.RDLogger.logger()
    lg.setLevel(rdkit.RDLogger.ERROR)
    rdkit.RDLogger.DisableLog("rdApp.info")
    rdkit.RDLogger.DisableLog("rdApp.*")

    if is_cached("reactions"):
        print("[INFO] Load preprocessed reactions from cache.")
        reactions = read_cache()
    else:
        print("[INFO] Preprocess reactions.")
        reactions = load_reactions(args.filename, args.col, args.p)
        write_cache(reactions)

    if any([args.rule_based, args.mcs, args.mcs_based]):
        if args.rule_based:
            reactions = rule_based_method(reactions, args.p)
            write_cache(reactions)
        elif is_cached("rule_based"):
            reactions = read_cache("rule_based")

        if args.mcs:
            reactions = find_mcs(reactions, args.col)
            write_cache(reactions)
        elif is_cached("mcs"):
            reactions = read_cache("mcs")

        if args.mcs_based:
            reactions = mcs_based_method(reactions, args.cs_passthrough)
            write_cache(reactions)
        elif is_cached("mcs_based"):
            reactions = read_cache("mcs_based")
    else:
        reactions = rule_based_method(reactions, args.p)
        write_cache(reactions)
        reactions = find_mcs(reactions, args.col)
        write_cache(reactions)
        mcs_based_result = mcs_based_method(reactions, args.cs_passthrough)
        write_cache(mcs_based_result)
        reactions = generate_output(reactions, args.col)
        write_cache(reactions)

    reactions = generate_output(reactions, args.col)
    write_cache(reactions)
    print_dataset_stats(reactions)
    export(reactions, args.o)


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
        "-p", default=-1, help="Number of processes used for imputation."
    )

    test_parser.add_argument(
        "--tmp-dir",
        default=None,
        help="Path where SynRBL stores intermediate results.",
    )
    test_parser.add_argument(
        "--col",
        default="reaction",
        help="The column name for reactions in the input .csv file.",
    )
    test_parser.add_argument(
        "--rule-based",
        action="store_true",
        help="Run rule-based method.",
    )
    test_parser.add_argument(
        "--mcs",
        action="store_true",
        help="Find maximum-common-substractures.",
    )
    test_parser.add_argument(
        "--mcs-based",
        action="store_true",
        help="Run MCS-based method.",
    )
    test_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of intermediate results.",
    )
    cs_passthrough_default = True
    test_parser.add_argument(
        "--cs-passthrough",
        default=cs_passthrough_default,
        help="Flag to controle if catalyst and solvent compounds are "
        + "passed through to the product side. (Default={})".format(
            cs_passthrough_default
        ),
    )

    test_parser.set_defaults(func=run)