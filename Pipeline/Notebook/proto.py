import collections
import numpy as np
import pandas as pd
import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.Draw as Draw
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import matplotlib.pyplot as plt
from SynRBL.SynVis.reaction_visualizer import ReactionVisualizer
import rdkit.Chem.MolStandardize.rdMolStandardize as rdMolStandardize
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynUtils.chem_utils import remove_atom_mapping
from SynRBL.SynMCSImputer.rules import MergeRule, CompoundRule, ExpandRule

path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("golden_dataset", "Final_Graph")
org_data = load_database(path)


def get_reaction_by_id(data, id):
    for i, item in enumerate(data):
        if item["R-id"] == id:
            return i, item
    return None


def clear_atom_nums(dataset):
    for k in ["new_reaction", "old_reaction"]:
        for i in range(len(dataset)):
            if k in dataset[i].keys():
                dataset[i][k] = remove_atom_mapping(dataset[i][k])


def print_error_summary(data):
    error_map = collections.defaultdict(lambda: [])
    fail_cnt = 0
    for i, r in enumerate(data):
        err = r["issue"]
        if len(err) > 0:
            error_map[err.split(".")[0]].append(i)
            fail_cnt += 1

    for k, v in error_map.items():
        print("{:<80} {:>4} {}".format(k, len(v), v[:10]))

    success_cnt = len(data) - fail_cnt
    print("-" * 50)
    print(
        "MCS was successful on {} ({:.0%}) reactions.".format(
            success_cnt, success_cnt / len(data)
        )
    )


def print_rule_summary(data):
    rule_map = {
        r.name: set()
        for r in CompoundRule.get_all() + ExpandRule.get_all() + MergeRule.get_all()
    }
    rule_map["no rule"] = set()
    for i, item in enumerate(data):
        if "rules" not in item.keys() or len(item["rules"]) == 0:
            rule_map["no rule"].add(i)
        else:
            for r in item["rules"]:
                rule_map[r].add(i)

    for rule, ids in rule_map.items():
        print("{:<30} {}".format(rule, len(ids)))


def plot_reaction(entry, show_atom_numbers=False, figsize=(10, 7.5)):
    visualizer = ReactionVisualizer(figsize=figsize)
    visualizer.plot_reactions(
        entry,
        "old_reaction",
        "new_reaction",
        compare=True,
        show_atom_numbers=show_atom_numbers,
        pathname="./tmp",
        savefig=True,
    )
    print("ID:", entry["R-id"])
    print("Imbalance:", entry["carbon_balance_check"])
    print("Smiles:", entry["smiles"])
    print("Sorted Reactants:", entry["sorted_reactants"])
    print("MCS:", [len(x) for x in entry["mcs_results"]])
    print("Boundaries:", entry["boundary_atoms_products"])
    print("Neighbors:", entry["nearest_neighbor_products"])
    print("Issue:", entry["issue"])
    print("Rules:", entry["rules"])


s = "Clc1ccccc1CBr"  # "c1ccc(P(=O)(c2ccccc2)c2ccccc2)cc1"
s = Chem.CanonSmiles(s)
print(s)
mol = rdmolfiles.MolFromSmiles(s)
enumerator = rdMolStandardize.TautomerEnumerator()
mol = enumerator.Canonicalize(mol)
if True:
    for i, atom in enumerate(mol.GetAtoms()):
        atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
atom = mol.GetAtomWithIdx(0)
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
img = Draw.MolToImage(mol)
ax.imshow(img)
ax.set_title(s)
ax.axis("off")
plt.show()


# |%%--%%| <tx0z4CFgIc|aUE1hGnjdO>

path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("golden_dataset", "Final_Graph")
data = load_database(path)
clear_atom_nums(data)
print(data[0].keys())

# |%%--%%| <aUE1hGnjdO|kL4B2dKA6i>
from SynRBL.SynMCSImputer.model import MCSImputer, build_compounds

imputer = MCSImputer()
sample = data[298]  # 67, 88 catalysis | 73 not catalysis
cset = build_compounds(sample)
for c in cset.compounds:
    print(len(c.boundaries), c.src_smiles == c.smiles, c.smiles)
imputer.impute_reaction(sample)
clear_atom_nums([sample])
print("Issue:", sample["issue"])
plot_reaction(sample, show_atom_numbers=False)

# |%%--%%| <kL4B2dKA6i|U1toLALgcc>

path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("golden_dataset", "MCS_Impute")
results = load_database(path)
clear_atom_nums(results)

# |%%--%%| <U1toLALgcc|ftWMEjJznz>

print_error_summary(results)
i, rx = get_reaction_by_id(results, "golden_dataset_196")
# print(i)
#rx = results[204]

plot_reaction(rx, show_atom_numbers=False)

# |%%--%%| <ftWMEjJznz|T6IJBZXUlT>
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.Draw as Draw
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import rdkit.Chem.rdChemReactions as rdChemReactions
import matplotlib.pyplot as plt

index = 104
reaction_data = data[index]
reactant = rdmolfiles.MolFromSmiles(reaction_data["reactants"])
print(rdmolfiles.MolToSmiles(reactant))
reaction = rdChemReactions.ReactionFromSmarts(
    reaction_data["reactions"], useSmiles=True
)
fig, ax = plt.subplots(2, 1, figsize=(1, 1.5))
img = Draw.ReactionToImage(reaction)
ax[0].imshow(img)
for i, atom in enumerate(reactant.GetAtoms()):
    atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
img = Draw.MolToImage(reactant)
ax[1].imshow(img)

# |%%--%%| <T6IJBZXUlT|7zydJ3VZZW>


entry = get_reaction_by_id(data, "R190")
print(entry)
# |%%--%%| <7zydJ3VZZW|0ppeA6PwQO>
import SynRBL.SynVis.reaction_visualizer as visualizer

old_smiles = "CC(=O)C.CC(C)(C)OOC(C)(C)C>>CC(=O)OC"
new_smiles = "CC(=O)C.CC(C)(C)OOC(C)(C)C.O>>CC(=O)OC.CC(C)(C)O.CC(C)(C)O"
vis = visualizer.ReactionVisualizer(figsize=(10, 8))
vis.plot_reactions(
    {"o": old_smiles, "n": new_smiles},
    old_reaction_col="o",
    new_reaction_col="n",
    compare=True,
)
# |%%--%%| <0ppeA6PwQO|Ffoj2H07hg>
import collections


def append_rids(val_set, src_id_map, data):
    for row in data:
        id = row["id"]
        rid = row["R-id"]
        rxn = row["reactions"]
        val_index = None
        for vi in src_id_map[id]:
            if rxn == val_set[vi]["reaction"]:
                if val_index is not None:
                    assert False, "Found duplicate: {}".format(id)
                val_index = vi
        assert val_index is not None, "Not found in val_set."
        assert val_set[val_index]["src_id"] == id
        assert val_set[val_index]["reaction"] == rxn
        val_set[val_index]["R-id"] = rid


val_set = []
blacklist_ids = ["patent_246"]
src_id_map = collections.defaultdict(lambda: [])
# load src data
for ds in _DATASETS:
    src_path = "./Data/Validation_set/{}.csv".format(ds)
    src_df = pd.read_csv(src_path)
    for i, row in src_df.iterrows():
        src_id = row["id"]
        if src_id in blacklist_ids:
            print("Skip '{}'. (Blacklisted)".format(src_id))
            continue
        is_duplicate = False
        for vi in src_id_map[src_id]:
            val_item = val_set[vi]
            if (
                val_item["src_id"] == src_id
                and val_item["reaction"] == row["reactions"]
            ):
                is_duplicate = True
                break
        if not is_duplicate:
            src_id_map[src_id].append(len(val_set))
            val_set.append(
                {
                    "src_id": src_id,
                    "R-id": "",
                    "dataset": ds,
                    "reaction": row["reactions"],
                }
            )

# check that no duplicates exist (id and reaction duplicates)
for id, indices in src_id_map.items():
    rxns = []
    for index in indices:
        val_item = val_set[index]
        assert val_item["reaction"] not in rxns
        rxns.append(val_item["reaction"])

# load ids from rule-based sets
for ds in _DATASETS:
    src_path = "./Data/Validation_set/{}/reactions_clean.json.gz".format(ds)
    data = load_database(src_path)
    append_rids(val_set, src_id_map, data)

val_df = pd.DataFrame(val_set)
print("{} R-ids after rule-based datasets.".format(len(val_df[val_df["R-id"] != ""])))

# load ids from mcs-based sets
for ds in _DATASETS:
    src_path = "./Data/Validation_set/{}/mcs_based_reactions.json.gz".format(ds)
    data = load_database(src_path)
    append_rids(val_set, src_id_map, data)

val_df = pd.DataFrame(val_set)
print("{} R-ids after mcs-based datasets.".format(len(val_df[val_df["R-id"] != ""])))
assert len(val_df[val_df["R-id"] == ""]) == 0
print(
    "Validation set was built successfully containing {} reactions.".format(len(val_df))
)
val_df.to_csv("./Data/Validation_set/validation_set.csv")
#|%%--%%| <Ffoj2H07hg|QOuhqrXaXd>
import json
import copy
from SynRBL.SynCmd.cmd_run import _ID_COL
from SynRBL.rsmi_utils import load_database, save_database

path = "./tmp/91cc3931677445d798e53218a8329aa6244631a5.cache"
with open(path, "r") as f:
    data = json.load(f)

reactions = copy.deepcopy(data["reactions"])
merge_data = copy.deepcopy(data["mcs_based"])
mcs_data = copy.deepcopy(data["rule_based_unsolved"])
ids = [value[_ID_COL] for value in merge_data]
mcs_data = [value for value in mcs_data if value[_ID_COL] in ids]
assert len(merge_data) == len(mcs_data)
merge_data_exp = []
mcs_data_exp = []
test_set_ids = []
id_map = {r[_ID_COL]: r['R-id'] for r in reactions}

for i, (md, mcsd) in enumerate(zip(merge_data, mcs_data)):
    assert md[_ID_COL] == mcsd[_ID_COL]
    if i % 5 == 0:
        test_set_ids.append(id_map[md[_ID_COL]])
    else:
        merge_data_exp.append(md)
        mcs_data_exp.append(mcsd)

save_database(merge_data_exp, "Data/Validation_set/MCS_Impute_train.json.gz")
save_database(mcs_data_exp, "Data/Validation_set/mcs_based_reactions_train.json.gz")
with open("Data/Validation_set/test_set_ids.json", "w") as f:
    json.dump({"ids": test_set_ids}, f, indent=4)

print("Train set:", len(merge_data_exp))
print("Test set:", len(test_set_ids))
