import os
import io
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rdkit.Chem.rdChemReactions as rdChemReactions
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
from PIL import Image
from SynRBL.SynUtils.chem_utils import normalize_smiles, remove_atom_mapping

import db_interface as db

def get_reaction_img(smiles):
    rxn = rdChemReactions.ReactionFromSmarts(smiles, useSmiles=True)
    d = rdMolDraw2D.MolDraw2DCairo(2000, 500)
    d.DrawReaction(rxn)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def plot_reactions(smiles, titles=None, suptitle=None):
    if not isinstance(smiles, list):
        smiles = [smiles]
    l = len(smiles)
    fig, axs = plt.subplots(l, 1, figsize=(10, l * 3))
    if suptitle is not None:
        fig.suptitle(suptitle, color="gray")
    if l == 1:
        axs = [axs]
    if titles is None:
        titles = ["" for _ in range(l)]
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

    fig, axs = plt.subplots(rows, 1, dpi=400, figsize=(10,7))

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
        if db_entry["correct_reaction"] != row["correct_reaction"]:
            cnt += 1
            row["correct_reaction"] = db_entry["correct_reaction"]
    df.to_csv("dataset_out.csv")
    print("Updated {} entires.".format(cnt))
        
#update_correct_reactions_in_output()

if not os.path.exists("imgs"):
    os.mkdir("imgs") 

df = pd.read_csv("dataset_out.csv")
wrong_cnt = 0
for idx, row in df.iterrows():
    if wrong_cnt == 10:
        break
    exp_rxn = None
    if row["correct_reaction"] is not np.nan:
        exp_rxn = normalize_smiles(row["correct_reaction"])
    in_rxn = normalize_smiles(row["input_reaction"])
    act_rxn = normalize_smiles(row["reaction"])
    if exp_rxn == None:
        wrong_cnt += 1
        print(
            "----- Unequal Reaction ({},{}) -----\n{}\n{}".format(
                idx, row["solved_by"], exp_rxn, act_rxn
            )
        )
        export_reaction(in_rxn, act_rxn, "imgs/{}.png".format(idx), exp=exp_rxn)

print("Wrong: {}".format(wrong_cnt))
#db.flush()

