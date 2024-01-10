import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.Draw as Draw
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import matplotlib.pyplot as plt
from SynRBL.SynVis.reaction_visualizer import ReactionVisualizer
import rdkit.Chem.MolStandardize.rdMolStandardize as rdMolStandardize
from SynRBL.rsmi_utils import load_database, save_database

path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("Jaworski", "Final_Graph")
org_data = load_database(path)


def plot_reaction(entry, show_atom_numbers=False, figsize=(10,7.5)):
    visualizer = ReactionVisualizer(figsize=figsize)
    visualizer.plot_reactions(
        entry,
        "old_reaction",
        "new_reaction",
        compare=True,
        show_atom_numbers=show_atom_numbers,
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


s = "c1ccccc1C(=S)OC"  # "CC[Si](C)(C)C"  # "c1ccc(P(=O)(c2ccccc2)c2ccccc2)cc1"
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


# |%%--%%| <PDHNfCjKgB|tx0z4CFgIc>
import collections
from SynRBL.SynMCSImputer.rules import CompoundRule, MergeRule
from SynRBL.SynMCSImputer.model import MCSImputer


def impute_new_reactions(data):
    imputer = MCSImputer()
    rule_map = {r.name: set() for r in CompoundRule.get_all() + MergeRule.get_all()}
    rule_map["no rule"] = set()
    for i, item in enumerate(data):
        imputer.impute_reaction(item)
        issue = item["issue"]
        if issue != "":
            print("[ERROR] [{}] {}".format(i, issue))
        rules = item["rules"]
        if len(rules) == 0:
            for r in rules:
                rule_map[r].add(i)
        else:
            rule_map["no rule"].add(i)
    return rule_map


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
    rule_map = {r.name: set() for r in CompoundRule.get_all() + MergeRule.get_all()}
    rule_map["no rule"] = set()
    for i, item in enumerate(data):
        if "rules" not in item.keys() or len(item["rules"]) == 0:
            rule_map["no rule"].add(i)
        else:
            for r in item["rules"]:
                rule_map[r].add(i)

    for rule, ids in rule_map.items():
        print("{:<30} {}".format(rule, len(ids)))


# |%%--%%| <tx0z4CFgIc|ftWMEjJznz>

path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("Jaworski", "MCS_Impute")
results = load_database(path)

print_error_summary(results)
plot_reaction(results[67], show_atom_numbers=True)

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


def get_reaction_by_id(id):
    path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("Jaworski", "Final_Graph")
    data = load_database(path)
    for i, item in enumerate(data):
        if item["R-id"] == id:
            return i, item
    return None


entry = get_reaction_by_id("R190")
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
