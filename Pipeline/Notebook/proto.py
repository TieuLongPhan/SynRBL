import collections
import numpy as np
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


s = "Clc1cccccc1CBr"  # "c1ccc(P(=O)(c2ccccc2)c2ccccc2)cc1"
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
compounds = build_compounds(sample)
for c in compounds:
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
i, rx = get_reaction_by_id(results, "golden_dataset_266")
# print(i)
rx = results[204]

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
