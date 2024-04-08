import copy
import json
import pandas as pd
from SynRBL.SynUtils.chem_utils import normalize_smiles, remove_atom_mapping

_FILE_NAME = "dataset"

_reaction_dict = None 

def _open():
    global _reaction_dict
    if _reaction_dict is None:
        with open("{}.json".format(_FILE_NAME), "r") as f:
            reaction_list = json.load(f)
        _reaction_dict = {}
        for r in reaction_list:
            _reaction_dict[normalize_smiles(r["reaction"])] = r

def _flush():
    global _reaction_dict
    assert _reaction_dict is not None
    reaction_list = list(_reaction_dict.values())

    with open("{}.json".format(_FILE_NAME), "w") as f:
        json.dump(reaction_list, f, indent=4)

    df = pd.DataFrame(reaction_list)
    df.to_csv("{}.csv".format(_FILE_NAME))

def update(reaction, correct_reaction=None, wrong_reaction=None):
    global _reaction_dict
    _open()
    assert _reaction_dict is not None
    if correct_reaction is not None:
        _reaction_dict[reaction]["correct_reaction"] = correct_reaction
    if wrong_reaction is not None:
        _reaction_dict[reaction]["wrong_reactions"].append(wrong_reaction)
    _flush()

