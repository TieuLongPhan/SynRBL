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

def export_reaction(in_r, exp, act, path):
    fig, axs = plt.subplots(3, 1, figsize=(10,7))
    exp_img = get_reaction_img(exp)
    act_img = get_reaction_img(act)
    in_img = get_reaction_img(in_r)
    axs[0].imshow(in_img)
    axs[0].set_title("Input")
    axs[1].imshow(exp_img)
    axs[1].set_title("Expected")
    axs[2].imshow(act_img)
    axs[2].set_title("Actual")
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    

if not os.path.exists("imgs"):
    os.mkdir("imgs") 

df = pd.read_csv("dataset_out.csv")
for idx, row in df.iterrows():
    if row["solved_by"] != "mcs-based":
        continue
    exp_rxn = None
    if row["correct_reaction"] is not np.nan:
        exp_rxn = normalize_smiles(row["correct_reaction"])
    in_rxn = normalize_smiles(row["input_reaction"])
    act_rxn = normalize_smiles(row["reaction"])
    if exp_rxn != act_rxn and exp_rxn != None:
        d = len(exp_rxn) - len(act_rxn)
        if "remove_water_catalyst" in row["rules"] and d == 2:
            print(
                "----- Unequal Reaction ({},{}) -----\n{}\n{}".format(
                    idx, row["solved_by"], exp_rxn, act_rxn
                )
            )
            db.update(in_rxn, correct_reaction=row["reaction"])
            #export_reaction(in_rxn, exp_rxn, act_rxn, "imgs/{}-{}.png".format(idx, d))
db.flush()
