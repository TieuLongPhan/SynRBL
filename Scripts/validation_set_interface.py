import copy
import json
import pandas as pd

from synrbl.SynUtils.chem_utils import normalize_smiles

_FILE_NAME = "dataset"

_reaction_dict = None


def _load():
    global _reaction_dict
    if _reaction_dict is None:
        with open("{}.json".format(_FILE_NAME), "r") as f:
            reaction_list = json.load(f)
        _reaction_dict = {}
        for rdata in reaction_list:
            rdata["wrong_reactions"] = set(
                [normalize_smiles(r) for r in rdata["wrong_reactions"]]
            )
            _reaction_dict[normalize_smiles(rdata["reaction"])] = rdata


def flush():
    global _reaction_dict
    if _reaction_dict is None:
        return
    reaction_list = []
    for r in _reaction_dict.values():
        _r = copy.deepcopy(r)
        _r["wrong_reactions"] = list(r["wrong_reactions"])
        reaction_list.append(_r)

    with open("{}.json".format(_FILE_NAME), "w") as f:
        json.dump(reaction_list, f, indent=4)

    df = pd.DataFrame(reaction_list)
    df.to_csv("{}.csv".format(_FILE_NAME))


def update(reaction, correct_reaction=None, wrong_reaction=None):
    global _reaction_dict
    _load()
    r_key = normalize_smiles(reaction)
    assert _reaction_dict is not None
    if correct_reaction is not None:
        _reaction_dict[r_key]["correct_reaction"] = correct_reaction
    if wrong_reaction is not None:
        _reaction_dict[r_key]["wrong_reactions"].update(
            [normalize_smiles(wrong_reaction)]
        )


def get(reaction):
    global _reaction_dict
    _load()
    assert _reaction_dict is not None
    r = normalize_smiles(reaction)
    return _reaction_dict[r]
