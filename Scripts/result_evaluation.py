import os
import io
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit.Chem.rdChemReactions as rdChemReactions
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
from PIL import Image
from synrbl.SynUtils.chem_utils import normalize_smiles, remove_atom_mapping

import scripts.validation_set_interface as db


def get_reaction_img(smiles):
    rxn = rdChemReactions.ReactionFromSmarts(smiles, useSmiles=True)
    d = rdMolDraw2D.MolDraw2DCairo(2000, 500)
    d.DrawReaction(rxn)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def plot_reactions(smiles, titles=None, suptitle=None):
    if not isinstance(smiles, list):
        smiles = [smiles]
    s_len = len(smiles)
    fig, axs = plt.subplots(s_len, 1, figsize=(10, s_len * 3))
    if suptitle is not None:
        fig.suptitle(suptitle, color="gray")
    if s_len == 1:
        axs = [axs]
    if titles is None:
        titles = ["" for _ in range(s_len)]
    for s, ax, title in zip(smiles, axs, titles):
        img = get_reaction_img(remove_atom_mapping(s))
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def export_reaction(in_r, act, path, exp=None):
    i = 0
    rows = 2
    if exp is not None:
        rows += 1

    _, axs = plt.subplots(rows, 1, dpi=400, figsize=(10, 7))

    in_img = get_reaction_img(in_r)
    axs[i].imshow(in_img)
    axs[i].set_title("Input")
    i += 1

    if exp is not None:
        exp_img = get_reaction_img(exp)
        axs[i].imshow(exp_img)
        axs[i].set_title("Expected")
        i += 1

    act_img = get_reaction_img(act)
    axs[i].imshow(act_img)
    axs[i].set_title("Actual")
    i += 1

    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def update_correct_reactions_in_output():
    df = pd.read_csv("dataset_out.csv")
    cnt = 0
    for idx, row in df.iterrows():
        db_entry = db.get(row["input_reaction"])
        correct_rxn = db_entry["correct_reaction"]
        if correct_rxn is not None and row["correct_reaction"] != correct_rxn:
            cnt += 1
            df.loc[idx, "correct_reaction"] = db_entry["correct_reaction"]
    df.to_csv("dataset_out.csv")
    print("Updated {} entires.".format(cnt))


# update_correct_reactions_in_output()

if not os.path.exists("imgs"):
    os.mkdir("imgs")

df = pd.read_csv("dataset_extended_out.csv")
correct_cnt = 0
known_correct_wrong_cnt = 0
known_wrong_cnt = 0
unknown_cnt = 0
solved = 0

export_n = 20
export_cnt = 0

for idx, row in df.iterrows():
    if row["solved_by"] not in ["input-balanced", "rule-based", "mcs-based"]:
        continue
    solved += 1
    exp_rxn = None
    # db_entry = db.get(row["input_reaction"])
    if row["correct_reaction"] is not np.nan:
        # exp_rxn = normalize_smiles(db_entry["correct_reaction"])
        exp_rxn = normalize_smiles(row["correct_reaction"])
    # wrong_rxns = list(db_entry["wrong_reactions"])
    wrong_rxns = row["wrong_reactions"]
    in_rxn = normalize_smiles(row["input_reaction"])
    act_rxn = normalize_smiles(row["reaction"])
    if exp_rxn != act_rxn:
        if exp_rxn is not None:
            known_correct_wrong_cnt += 1
            # print(
            #    "----- Wrong Reaction ({},{}) -----\n{}\n{}".format(
            #        idx, row["solved_by"], exp_rxn, act_rxn
            #    )
            # )
            if export_cnt < export_n:
                export_reaction(
                    in_rxn, act_rxn, "imgs/{}-{}.png".format(idx, "wrong"), exp=exp_rxn
                )
                export_cnt += 1
        else:
            if act_rxn in wrong_rxns:
                known_wrong_cnt += 1
            else:
                diffs = [len(act_rxn) - len(wr) for wr in wrong_rxns]
                if 2 in diffs:
                    unknown_cnt += 1
                    # print(
                    #    "----- Uncertain Reaction ({},{}) -----\n{}\n{}".format(
                    #        idx, row["solved_by"], exp_rxn, act_rxn
                    #    )
                    # )
                    if export_cnt < export_n:
                        export_reaction(
                            in_rxn,
                            act_rxn,
                            "imgs/{}-{}.png".format(idx, "unknown"),
                            exp=exp_rxn,
                        )
                        export_cnt += 1
                else:
                    known_wrong_cnt += 1
    else:
        correct_cnt += 1

assert solved == (correct_cnt + known_correct_wrong_cnt + known_wrong_cnt + unknown_cnt)

print("Reactions: {}".format(len(df)))
print("Correct: {}".format(correct_cnt))
print("Known correct and now wrong: {}".format(known_correct_wrong_cnt))
print("Uncertain: {}".format(unknown_cnt))
