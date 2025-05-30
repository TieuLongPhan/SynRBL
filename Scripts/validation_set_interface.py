import copy
import json
import pandas as pd
from typing import Optional, Dict, Any
from synrbl.SynUtils.chem_utils import normalize_smiles

_FILE_NAME = "dataset"
_reaction_dict: Optional[Dict[str, Dict[str, Any]]] = None


def _load() -> None:
    """Lazy‐load the JSON into the module‐level _reaction_dict."""
    global _reaction_dict
    if _reaction_dict is not None:
        return

    with open(f"{_FILE_NAME}.json", "r") as f:
        reaction_list = json.load(f)

    _reaction_dict = {}
    for rdata in reaction_list:
        # normalize and dedupe wrong_reactions
        raw_wrongs = rdata.get("wrong_reactions", [])
        rdata["wrong_reactions"] = {normalize_smiles(r) for r in raw_wrongs}

        # use the normalized “reaction” as the key
        key = normalize_smiles(rdata["reaction"])
        _reaction_dict[key] = rdata


def flush() -> None:
    """
    Write the in-memory _reaction_dict back out to JSON and CSV.
    If nothing’s loaded, do nothing.
    """
    if _reaction_dict is None:
        return

    reaction_list = []
    for rdata in _reaction_dict.values():
        entry = copy.deepcopy(rdata)
        entry["wrong_reactions"] = list(entry["wrong_reactions"])
        reaction_list.append(entry)

    with open(f"{_FILE_NAME}.json", "w") as f:
        json.dump(reaction_list, f, indent=4)

    df = pd.DataFrame(reaction_list)
    df.to_csv(f"{_FILE_NAME}.csv", index=False)


def update(
    reaction: str,
    correct_reaction: Optional[str] = None,
    wrong_reaction: Optional[str] = None,
) -> None:
    """
    Update a single entry in _reaction_dict:
      - set correct_reaction if given
      - add one more wrong_reaction if given
    Raises KeyError if the base reaction isn’t yet in the dataset.
    """
    _load()
    key = normalize_smiles(reaction)

    if _reaction_dict is None or key not in _reaction_dict:
        raise KeyError(f"Reaction not found: {reaction!r}")

    entry = _reaction_dict[key]
    if correct_reaction is not None:
        entry["correct_reaction"] = correct_reaction

    if wrong_reaction is not None:
        entry["wrong_reactions"].add(normalize_smiles(wrong_reaction))


def get(reaction: str) -> Dict[str, Any]:
    """
    Return the stored record for one reaction (after normalizing).
    Raises KeyError if missing.
    """
    _load()
    key = normalize_smiles(reaction)

    if _reaction_dict is None or key not in _reaction_dict:
        raise KeyError(f"Reaction not found: {reaction!r}")

    return _reaction_dict[key]
