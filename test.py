from SynRBL import Balancer
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

def get_reaction_img(smiles):
    rxn = rdChemReactions.ReactionFromSmarts(smiles, useSmiles=True)
    d = rdMolDraw2D.MolDraw2DCairo(2000, 500)
    d.DrawReaction(rxn)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))

rxn = "[C:5]([O:4][CH2:3][CH3:2])=[O:6].[OH2:1]>>[C:5]([OH:1])=[O:6]"

plt.imshow(get_reaction_img(rxn))
plt.show()

balancer = Balancer()
result = balancer.rebalance([rxn], output_dict=True)

plt.imshow(get_reaction_img(result[0]["reaction"]))
plt.show()
