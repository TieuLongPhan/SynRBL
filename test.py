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

rxn = "O=CC1=CC=CC=C1.C=CCBr>>O[C@@H](CC=C)C1=CC=CC=C1.[H]Br"

plt.imshow(get_reaction_img(rxn))
plt.show()

balancer = Balancer()
result = balancer.rebalance([rxn], output_dict=True)
print(result)

plt.imshow(get_reaction_img(result[0]["reaction"]))
plt.show()
