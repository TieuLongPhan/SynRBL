import os
import io
import json
import argparse
import pandas as pd
from SynRBL.SynUtils.chem_utils import normalize_smiles
from SynRBL.rsmi_utils import load_database
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import rdkit.Chem.rdChemReactions as rdChemReactions
import PIL.Image as Image

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


def load_data(dataset):
    data = load_database(os.path.abspath(_DATASET_PATH_FMT.format(dataset)))
    df = pd.read_csv(_FINAL_VALIDATION_PATH)
    with open(_SNAPSHOT_PATH, "r") as f:
        snapshot = json.load(f)
    return data, df, snapshot


def load_reaction_data(id):
    for dataset in _DATASETS:
        data, df, snapshot = load_data(dataset)
        for item in data:
            _id = item["R-id"]
            if id == _id:
                assert id in snapshot.keys(), "Id not in snapshot."
                df_index = df.index[df["R-id"] == id].to_list()
                assert len(df_index) == 1
                df_index = df_index[0]
                return item, df, df_index, snapshot
    raise KeyError("Reaction '{}' not found.".format(id))


def set_reaction_correct(id, save=False, override=None):
    item, df, df_index, snapshot = load_reaction_data(id)
    row = df.iloc[df_index]
    correct_reaction = item["new_reaction"]
    if row["Result"] == True:
        msg = "Reaction '{}' is already marked correct.".format(id)
        if override == True:
            print("[WARN] {} Override correct reaction.".format(msg))
        else:
            raise RuntimeError(msg)
    with open(_SNAPSHOT_PATH, "r") as f:
        snapshot = json.load(f)
    if id not in snapshot.keys():
        raise KeyError("Reaction '{}' not found in snapshot.".format(id))
    df = pd.read_csv(_FINAL_VALIDATION_PATH)
    df.at[df_index, "correct_reaction"] = correct_reaction
    df.at[df_index, "Result"] = True
    snapshot[id]["checked_reaction"] = correct_reaction
    if save:
        df.to_csv(_FINAL_VALIDATION_PATH)
        with open(_SNAPSHOT_PATH, "w") as f:
            json.dump(snapshot, f, indent=4)


def set_reaction_wrong(id, save=False):
    item, df, df_index, snapshot = load_reaction_data(id)
    row = df.iloc[df_index]
    wrong_reaction = item["new_reaction"]
    if row["Result"] == True:
        raise RuntimeError("Reaction '{}' has already a correct result.".format(id))
    with open(_SNAPSHOT_PATH, "r") as f:
        snapshot = json.load(f)
    if id not in snapshot.keys():
        raise KeyError("Reaction '{}' not found in snapshot.".format(id))
    df = pd.read_csv(_FINAL_VALIDATION_PATH)
    snapshot[id]["wrong_reactions"].insert(0, wrong_reaction)
    if save:
        with open(_SNAPSHOT_PATH, "w") as f:
            json.dump(snapshot, f, indent=4)


def verify_dataset(dataset, ignore_rib=False):
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
            print("[WARNING] Reaction '{}' is not part of final_validation.".format(id))
            continue
        rxn_cnt += 1
        if ignore_rib and item["carbon_balance_check"] == "reactants":
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
                wrong_reaction = None
                if len(wrong_reactions_n) > 0:
                    wrong_reaction = wrong_reactions_n[0]
                unknown_rxn.append(
                    _fmt(
                        id, initial_reaction, result_reaction, checked_r=wrong_reaction
                    )
                )
    return {
        "wrong_reactions": wrong_rxn,
        "unknown_reactions": unknown_rxn,
        "reaction_cnt": rxn_cnt,
        "success_cnt": success_cnt,
        "correct_cnt": correct_cnt,
    }


def verify_datasets(dataset_name=None, ignore_rib=False):
    results = {}
    for dataset in _DATASETS:
        if dataset_name is not None and dataset_name.lower() != dataset.lower():
            continue
        results[dataset] = verify_dataset(dataset, ignore_rib)
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


def export(results, path, n=None):
    wrong_reactions, unknown_reactions = [], []
    for _, v in results.items():
        wrong_reactions.extend(v["wrong_reactions"])
        unknown_reactions.extend(v["unknown_reactions"])
    if n is not None:
        n = int(n)
        unknown_reactions = unknown_reactions[:n]
        wrong_reactions = wrong_reactions[:n]
    print(
        "[INFO] Export {} unknown reactions to {}.".format(len(unknown_reactions), path)
    )
    for item in unknown_reactions:
        plot_reaction(item, path=path)
    print("[INFO] Export {} wrong reactions to {}.".format(len(wrong_reactions), path))
    for item in wrong_reactions:
        plot_reaction(item, path=path)


def run_test(args):
    run_fix = False
    if args.set_correct is not None:
        run_fix = True
        for id in args.set_correct:
            print("[INFO] Save reaction '{}' as correct.".format(id))
            set_reaction_correct(id, save=True, override=args.override)
    if args.set_wrong is not None:
        run_fix = True
        for id in args.set_wrong:
            print("[INFO] Save reaction '{}' as wrong.".format(id))
            set_reaction_wrong(id, save=True)
    if run_fix:
        return

    results = verify_datasets(args.dataset, args.ignore_rib)
    print_result_table(results)
    print_verification_result(results)
    if args.export:
        export(results, args.o, args.export_count)


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
        help="Export unknown and wrong reactions as image. "
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

    test_parser.set_defaults(func=run_test)
