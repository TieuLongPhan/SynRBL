import re
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.Draw as Draw
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import matplotlib.pyplot as plt
from SynRBL.SynVis.reaction_visualizer import ReactionVisualizer
from SynRBL.SynRuleImputer.synthetic_rule_constraint import RuleConstraint
import rdkit.Chem.MolStandardize.rdMolStandardize as rdMolStandardize
from SynRBL.rsmi_utils import load_database, save_database

path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("golden_dataset", "Final_Graph")
data = load_database(path)

def get_reaction_by_id(data, id):
    for i, item in enumerate(data):
        if item["R-id"] == id:
            return i, item
    return None

def remove_atom_numbers(data):
    def _str_op(smiles):
        if isinstance(smiles, str):
            mapping_pattern = re.compile(r':\d+')
            return mapping_pattern.sub('', smiles)
        else:
            return smiles
    for i, item in enumerate(data):
        for k in ["new_reaction", "old_reaction", "smiles", "sorted_reactants"]:
            if k in item.keys():
                if isinstance(item[k], list):
                    data[i][k] = [_str_op(e) for e in data[i][k]]
                else: 
                    data[i][k] = _str_op(data[i][k])

remove_atom_numbers(data)
print(data[0])

