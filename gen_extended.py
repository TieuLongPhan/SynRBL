import json
import copy
import collections
import pandas as pd

db_file = "dataset"
with open("{}.json".format(db_file), "r") as f:
    dataset = json.load(f)

extended_db = []

for e in dataset:
    correct_rxn = e["correct_reaction"]
    if correct_rxn is None:
        continue
    correct_rxn_split = correct_rxn.split(">>")
    correct_rxn_psplit = correct_rxn_split[1].split(".")
    if len(correct_rxn_psplit) > 1:
        for i in range(len(correct_rxn_psplit)):
            _e = copy.deepcopy(e)
            used_psplit = [p for j, p in enumerate(correct_rxn_psplit) if j != i]
            assert len(correct_rxn_psplit) - len(used_psplit) == 1
            rxn_p = ".".join(used_psplit)
            rxn = ">>".join([correct_rxn_split[0], rxn_p])
            _e["reaction"] = rxn
            extended_db.append(_e) 
    else:
        extended_db.append(e)

df = pd.DataFrame(extended_db)
print(df)

df.to_csv("{}_extended.csv".format(db_file))

