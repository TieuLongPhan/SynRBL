import os
import io
import json
import argparse
import collections
import pandas as pd
from SynRBL.SynUtils.chem_utils import normalize_smiles
from SynRBL.rsmi_utils import load_database
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import rdkit.Chem.rdChemReactions as rdChemReactions
import PIL.Image as Image

from .cmd_run import impute

_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../..")
_FINAL_VALIDATION_PATH = os.path.join(
    _PATH, "Pipeline/Validation/Analysis/final_validation.csv"
)
_VALSET_PATH = os.path.join(_PATH, "Data/Validation_set/validation_set.csv")
_RESULT_PATH = os.path.join(
    _PATH, "Data/Validation_set/validation_set_result.csv"
)
_SNAPSHOT_PATH = os.path.join(_PATH, "Data/Validation_set/snapshot.json")
_REACTION_COL = "reaction"


def get_result_path():
    return os.path.join(_PATH, _RESULT_PATH)


def get_validation_set_path():
    return os.path.join(_PATH, _VALSET_PATH)


def get_reaction_img(smiles):
    rxn = rdChemReactions.ReactionFromSmarts(smiles, useSmiles=True)
    d = rdMolDraw2D.MolDraw2DCairo(2000, 500)
    d.DrawReaction(rxn)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def plot_reactions(smiles, titles=None, suptitle=None, filename=None, dpi=300):
    if not isinstance(smiles, list):
        smiles = [smiles]
    l = len(smiles)
    fig, axs = plt.subplots(l, 1, figsize=(10, l * 3), dpi=dpi)
    if suptitle is not None:
        fig.suptitle(suptitle, color="gray")
    if l == 1:
        axs = [axs]
    if titles is None:
        titles = ["" for _ in range(l)]
    for s, ax, title in zip(smiles, axs, titles):
        img = get_reaction_img(s)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()
    fig.clf()
    plt.close()


def plot_reaction(item, path=None, dpi=300):
    smiles = [normalize_smiles(item["initial_reaction"])]
    titles = ["Initial Reaction"]
    correct_r = item["correct_reaction"]
    checked_r = item["checked_reaction"]
    if correct_r is not None:
        smiles.append(normalize_smiles(correct_r))
        titles.append("Correct Reaction")
    elif checked_r is not None:
        smiles.append(normalize_smiles(checked_r))
        titles.append("Checked but WRONG")
    smiles.append(normalize_smiles(item["result_reaction"]))
    titles.append("New Imputation")
    filename = None
    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path, "{}.jpg".format(item["R-id"]))
    plot_reactions(smiles, titles, item["R-id"], filename=filename, dpi=300)


def get_val_index(val_set, rid):
    val_index = val_set.index[val_set["R-id"] == rid].to_list()
    if len(val_index) == 0:
        return None
    if len(val_index) != 1:
        raise RuntimeError(
            "Found {} validation set indices for reaction R-id '{}'.".format(
                len(val_index), rid
            )
        )
    return val_index[0]


_results = None
_val_set = None
_snapshot = None


def load_data(use_cache=True):
    global _results, _val_set, _snapshot
    if _results is None or _val_set is None or _snapshot is None or not use_cache:
        _results = pd.read_csv(get_result_path())
        _val_set = pd.read_csv(_FINAL_VALIDATION_PATH, index_col=0)
        with open(_SNAPSHOT_PATH, "r") as f:
            _snapshot = json.load(f)
    return _results, _val_set, _snapshot


_rid_dict = None


def get_reaction(rid):
    global _rid_dict
    results, val_set, snapshot = load_data()
    if _rid_dict is None:
        _rid_dict = {}
        for _idx, _item in results.iterrows():
            _rid = _item["R-id"]
            assert rid in snapshot.keys(), "Id not in snapshot."
            _val_idx = get_val_index(val_set, _rid)
            assert _rid not in _rid_dict.keys(), "Duplicate in _rid_dict."
            _rid_dict[_rid] = (_idx, _val_idx)
    if rid not in _rid_dict.keys():
        raise KeyError("Reaction with R-id '{}' not found.".format(id))
    ids = _rid_dict[rid]
    item = results.iloc[ids[0]]
    val_item = val_set.iloc[ids[1]]
    s_item = snapshot[rid]
    return item, val_item, s_item, ids


def set_reaction_correct(rid, save=False, override=None):
    _, val_set, snapshot = load_data()
    item, val_item, _, ids = get_reaction(rid)
    _, val_idx = ids
    correct_reaction = item["new_reaction"]
    if val_item["Result"] == True:
        msg = "Reaction '{}' is already marked correct.".format(rid)
        if override == True:
            print("[WARN] {} Override correct reaction.".format(msg))
        else:
            raise RuntimeError(msg)
    val_set.at[val_idx, "correct_reaction"] = correct_reaction
    val_set.at[val_idx, "Result"] = True
    snapshot[rid]["checked_reaction"] = correct_reaction
    if save:
        val_set.to_csv(_FINAL_VALIDATION_PATH)
        with open(_SNAPSHOT_PATH, "w") as f:
            json.dump(snapshot, f, indent=4)


def set_reaction_wrong(rid, save=False):
    _, _, snapshot = load_data()
    item, val_item, _, _ = get_reaction(rid)
    wrong_reaction = item["new_reaction"]
    if val_item["Result"] == True:
        raise RuntimeError("Reaction '{}' has already a correct result.".format(rid))
    snapshot[rid]["wrong_reactions"].insert(0, wrong_reaction)
    if save:
        with open(_SNAPSHOT_PATH, "w") as f:
            json.dump(snapshot, f, indent=4)


def verify_results(ignore_rib=False):
    def _fmt(rid, initial_r, result_r, correct_r=None, checked_r=None):
        return {
            "initial_reaction": initial_r,
            "result_reaction": result_r,
            "correct_reaction": correct_r,
            "checked_reaction": checked_r,
            "R-id": rid,
        }

    output = collections.defaultdict(
        lambda: {
            "wrong_rxns": [],
            "unknown_rxns": [],
            "new_rxns": [],
            "rxn_cnt": 0,
            "rb_cnt": 0,
            "mcs_suc_cnt": 0,
            "mcs_correct_cnt": 0,
        }
    )
    results, val_set, snapshot = load_data()

    for _, item in results.iterrows():
        rid = item["R-id"]
        dataset = item["dataset"]
        _o = output[dataset]
        _o["rxn_cnt"] += 1  # type: ignore
        solved = item["solved"]
        if not solved:  # type: ignore
            continue
        solved_by = item["solved_by"]
        initial_reaction = item[_REACTION_COL]
        if solved_by == "was-balanced":
            pass
        elif solved_by == "rule-based-method":
            _o["rb_cnt"] += 1  # type: ignore
        elif solved_by == "mcs-based-method":
            _o["mcs_suc_cnt"] += 1  # type: ignore
            result_reaction = item["new_reaction"]
            val_index = get_val_index(val_set, rid)
            if val_index is None:
                print(
                    "[WARNING] R-id '{}' was not found in final_validation.csv".format(
                        rid
                    )
                )
                _o["new_rxns"].append(  # type: ignore
                    _fmt(rid, initial_reaction, result_reaction)
                )
                continue
            val_item = val_set.iloc[val_index]
            sn_item = snapshot[rid]
            assert rid in snapshot.keys(), "Id '{}' not in snapshot.".format(rid)
            is_correct = val_item["Result"]
            assert initial_reaction == val_item["reactions"]
            result_reaction_n = normalize_smiles(result_reaction)  # type: ignore
            if is_correct:
                correct_reaction = val_item["correct_reaction"]
                if result_reaction_n != normalize_smiles(correct_reaction):
                    _o["wrong_rxns"].append(  # type: ignore
                        _fmt(
                            rid,
                            initial_reaction,
                            result_reaction,
                            correct_r=correct_reaction,
                        )
                    )
                else:
                    _o["mcs_correct_cnt"] += 1  # type: ignore
            else:
                wrong_reactions = sn_item["wrong_reactions"]
                wrong_reactions_n = [normalize_smiles(r) for r in wrong_reactions]
                if result_reaction_n not in wrong_reactions_n:
                    wrong_reaction = None
                    if len(wrong_reactions_n) > 0:
                        wrong_reaction = wrong_reactions_n[0]
                    _o["unknown_rxns"].append(  # type: ignore
                        _fmt(
                            rid,
                            initial_reaction,
                            result_reaction,
                            checked_r=wrong_reaction,
                        )
                    )
        else:
            raise NotImplementedError()
    return output


def print_result_table(results):
    line_fmt = "{:<25} {:>12} {:>12} {:>12} {:>12}"
    cols = [
        "Dataset",
        "Reactions",
        "Rule-based Suc.",
        "MCS-based Suc.",
        "MCS-based Acc.",
    ]
    head_line = line_fmt.format(*cols)
    print("=" * len(head_line))
    print(head_line)
    print("-" * len(head_line))
    for db, result in results.items():
        rxn_cnt = result["rxn_cnt"]
        rb_cnt = result["rb_cnt"]
        mcs_cnt = rxn_cnt - rb_cnt
        mcs_suc_cnt = result["mcs_suc_cnt"]
        mcs_correct_cnt = result["mcs_correct_cnt"]
        rb_suc_rate_str = "inf" if rb_cnt == 0 else "{:.2%}".format(rb_cnt / rxn_cnt)
        mcs_suc_rate_str = (
            "-" if mcs_cnt == 0 else "{:.2%}".format(mcs_suc_cnt / mcs_cnt)
        )
        mcs_acc_rate_str = (
            "-" if mcs_suc_cnt == 0 else "{:.2%}".format(mcs_correct_cnt / mcs_suc_cnt)
        )
        values = [db, rxn_cnt, rb_suc_rate_str, mcs_suc_rate_str, mcs_acc_rate_str]
        print(line_fmt.format(*values))
    print("-" * len(head_line))


def print_verification_result(results):
    good = True
    wrong_rxns, unknown_rxns, new_rxns = [], [], []
    reaction_cnt = 0
    for db, r in results.items():
        wrong_rxns.extend(r["wrong_rxns"])
        unknown_rxns.extend(r["unknown_rxns"])
        new_rxns.extend(r["new_rxns"])
        reaction_cnt += r["rxn_cnt"]
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
    if len(new_rxns) > 0:
        print(
            (
                "[INFO] {} reactions were not checked before. "
                + "Please check if they are correct now."
            ).format(len(new_rxns))
        )
        good = False
    if good:
        print("[INFO] All good!")


def export(results, path, exp_wrong=False, exp_unknown=False, exp_new=False, n=None):
    wrong_rxns, unknown_rxns, new_rxns = [], [], []
    for _, v in results.items():
        wrong_rxns.extend(v["wrong_rxns"])
        unknown_rxns.extend(v["unknown_rxns"])
        new_rxns.extend(v["new_rxns"])
    if n is not None:
        n = int(n)
        unknown_rxns = unknown_rxns[: n if exp_unknown else 0]
        wrong_rxns = wrong_rxns[: n if exp_wrong else 0]
        new_rxns = new_rxns[: n if exp_new else 0]
    print("[INFO] Export {} unknown reactions to {}.".format(len(unknown_rxns), path))
    for item in unknown_rxns:
        plot_reaction(item, path=path)
    print("[INFO] Export {} wrong reactions to {}.".format(len(wrong_rxns), path))
    for item in wrong_rxns:
        plot_reaction(item, path=path)
    print("[INFO] Export {} new reactions to {}.".format(len(new_rxns), path))
    for item in new_rxns:
        plot_reaction(item, path=path)


def run_impute(no_cache=False):
    print("[INFO] Impute validation set.")
    src_file = get_validation_set_path()
    result_file = get_result_path()
    impute(
        src_file,
        result_file,
        reaction_col=_REACTION_COL,
        cols=["R-id", "dataset"],
        no_cache=no_cache,
    )


def run_test(args):
    run_fix_mode = False
    if args.set_correct is not None:
        run_fix_mode = True
        for id in args.set_correct:
            print("[INFO] Save reaction '{}' as correct.".format(id))
            set_reaction_correct(id, save=True, override=args.override)
    if args.set_wrong is not None:
        run_fix_mode = True
        for id in args.set_wrong:
            print("[INFO] Save reaction '{}' as wrong.".format(id))
            set_reaction_wrong(id, save=True)
    if run_fix_mode:
        return

    if args.impute:
        run_impute(no_cache=args.no_cache)

    results = verify_results(ignore_rib=args.ignore_rib)
    print_result_table(results)
    print_verification_result(results)
    f_exp = args.export
    f_exp_n = args.export_new or f_exp
    f_exp_w = args.export_wrong or f_exp
    f_exp_u = args.export_unknown or f_exp
    if any([f_exp_n, f_exp_w, f_exp_u]):
        export(results, args.o, exp_new=f_exp_n, exp_wrong=f_exp_w, exp_unknown=f_exp_u)


def configure_argparser(argparser: argparse._SubParsersAction):
    test_parser = argparser.add_parser(
        "test", description="Test success rate and accuracy of SynRBL."
    )

    test_parser.add_argument("-o", default="./out", help="Path where output is saved.")
    test_parser.add_argument(
        "--dataset", default=None, help="Use a specific dataset for testing."
    )
    test_parser.add_argument(
        "--export",
        action="store_true",
        help="Export unknown, wrong and new reactions as image. "
        + "Use -o to specify the output directory.",
    )
    test_parser.add_argument(
        "--export-new",
        action="store_true",
        help="Export new reactions as image. "
        + "Use -o to specify the output directory.",
    )
    test_parser.add_argument(
        "--export-wrong",
        action="store_true",
        help="Export wrong reactions as image. "
        + "Use -o to specify the output directory.",
    )
    test_parser.add_argument(
        "--export-unknown",
        action="store_true",
        help="Export unknown reactions as image. "
        + "Use -o to specify the output directory.",
    )
    test_parser.add_argument(
        "--export-count",
        default=None,
        help="Set the number of reactions to export.",
    )

    test_parser.add_argument(
        "--set-correct",
        nargs="*",
        metavar="id",
        help="The reaction ids that are now correct.",
    )
    test_parser.add_argument(
        "--set-wrong",
        nargs="*",
        metavar="id",
        help="The reaction ids that are now wrong.",
    )
    test_parser.add_argument(
        "--override", action="store_true", help="Flag to override correct reactions."
    )
    test_parser.add_argument(
        "--ignore-rib",
        action="store_true",
        help="Flag to ignore reactant side imbalances.",
    )
    test_parser.add_argument(
        "--impute",
        action="store_true",
        help="Flag to (re)-impute test cases.",
    )
    test_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching of intermediate results.",
    )

    test_parser.set_defaults(func=run_test)
