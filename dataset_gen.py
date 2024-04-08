import json
import copy
import numpy as np
import pandas as pd

snapshot_path = "Data/Validation_set/snapshot.json"
dataset_path_fmt = "Data/Validation_set/{}.csv"
final_validation = "Data/Validation_set/final_validation.csv"
datasets = [
    "USPTO_random_class",
    "USPTO_diff",
    "USPTO_unbalance_class",
    "golden_dataset",
    "Jaworski",
]

with open(snapshot_path, "r") as f:
    snapshot = json.load(f)
df_final_val = pd.read_csv(final_validation)

reaction_dict = {}
rid_rxn_map = {}
for idx, row in df_final_val.iterrows():
    rxn = row["reactions"]
    rid = row["R-id"]
    rid_rxn_map[rid] = rxn
    dataset = "_".join(rid.split("_")[:-1])
    if rxn in reaction_dict.keys():
        reaction_dict[rxn]["R-ids"].append(rid)
        reaction_dict[rxn]["datasets"].update([dataset])
    else:
        correct_reaction = row["correct_reaction"]
        if correct_reaction is np.nan:
            correct_reaction = None
        reaction_dict[rxn] = {
            "R-ids": [rid],
            "datasets": set([dataset]),
            "correct_reaction": correct_reaction,
            "wrong_reactions": [],
        }

for dataset in datasets:
    df_dataset = pd.read_csv(dataset_path_fmt.format(dataset))
    for idx, row in df_dataset.iterrows():
        rxn = row["reactions"]
        if rxn in reaction_dict.keys():
            reaction_dict[rxn]["datasets"].update([dataset])
        else:
            reaction_dict[rxn] = {
                "R-ids": [],
                "correct_reaction": None,
                "wrong_reactions": [],
                "datasets": set([dataset]),
            }

for rid, data in snapshot.items():
    assert rid in rid_rxn_map
    rxn = rid_rxn_map[rid]
    reaction_dict[rxn]["wrong_reactions"].extend(data["wrong_reactions"])

reaction_list = []
for k, v in reaction_dict.items():
    _v = copy.deepcopy(v)
    _v["reaction"] = k
    _v["datasets"] = list(v["datasets"])
    reaction_list.append(_v)

with open("dataset.json", "w") as f:
    json.dump(reaction_list, f, indent=4)

df = pd.DataFrame(reaction_list)
df.to_csv("dataset.csv")

