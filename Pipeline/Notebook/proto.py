import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.Draw as Draw
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import matplotlib.pyplot as plt
from SynRBL.SynVis.reaction_visualizer import ReactionVisualizer

s = "C(=O)C1CCN(Cc2ccccc2)CC1"  # "CC[Si](C)(C)C"  # "c1ccc(P(=O)(c2ccccc2)c2ccccc2)cc1"
s = Chem.CanonSmiles(s)
print(s)
mol = rdmolfiles.MolFromSmiles(s)
# for i, atom in enumerate(mol.GetAtoms()):
#    atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
atom = mol.GetAtomWithIdx(0)
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
img = Draw.MolToImage(mol)
ax.imshow(img)
ax.set_title(s)
ax.axis("off")
plt.show()


# |%%--%%| <PDHNfCjKgB|IXhnkWIUcu>
from SynRBL.SynMCS.merge import merge
import SynRBL.SynMCS.structure as structure
import SynRBL.SynMCS.utils as utils
import traceback


def impute_new_reaction(data):
    rule_map = {r.name: set() for r in CompoundRule.get_all() + MergeRule.get_all()}
    rule_map["no rule"] = set()
    for i, item in enumerate(data):
        data[i]["rules"] = []
        data[i]["new_reaction"] = data[i]["old_reaction"]
        if data[i]["issue"] != "":
            print("[ERROR] [{}]".format(i), "Skip because of previous issue.")
            continue
        try:
            compounds = structure.build_compounds(item)
            result = merge(compounds)
            new_reaction = "{}.{}".format(item["old_reaction"], result.smiles)
            data[i]["new_reaction"] = new_reaction
            rules = [r.name for r in result.rules]
            data[i]["rules"] = rules
            if len(rules) == 0:
                rule_map["no rule"].add(i)
            else:
                for r in data[i]["rules"]:
                    rule_map[r].add(i)
            utils.carbon_equality_check(new_reaction)
        except Exception as e:
            # traceback.print_exc()
            data[i]["issue"] = str(e)
            print("[ERROR] [{}]".format(i), e)
    return rule_map


# |%%--%%| <IXhnkWIUcu|tx0z4CFgIc>
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCS.rules import CompoundRule, MergeRule

path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("golden_dataset", "Final_Graph")
data = load_database(path)
impute_new_reaction(data)
rule_map = {r.name: set() for r in CompoundRule.get_all() + MergeRule.get_all()}
rule_map["no rule"] = set()
for i, item in enumerate(data):
    if len(item["rules"]) == 0:
        rule_map["no rule"].add(i)
    else:
        for r in item["rules"]:
            rule_map[r].add(i)


def print_rule_summary(rule_map):
    for rule, ids in rule_map.items():
        print("{:<30} {}".format(rule, len(ids)))


print_rule_summary(rule_map)

# |%%--%%| <tx0z4CFgIc|FtdgkzRxV9>


def plot_reaction(entry):
    visualizer = ReactionVisualizer(figsize=(10, 10))
    visualizer.plot_reactions(
        entry, "old_reaction", "new_reaction", compare=True, show_atom_numbers=True
    )
    print("Compounds:", entry["smiles"])
    print("Boundaries:", entry["boundary_atoms_products"])
    print("Neighbors:", entry["nearest_neighbor_products"])
    print("Issue:", entry["issue"])
    print("Rules:", entry["rules"])


plot_reaction(data[759])

# |%%--%%| <FtdgkzRxV9|mWahMtVBFr>
import SynRBL.SynMCS.structure as structure

for i, entry in enumerate(data):
    for compound in structure.build_compounds(entry):
        for boundary in compound.boundaries:
            atom = boundary.get_atom()
            expHs = atom.GetNumExplicitHs()
            if expHs > 0:
                print("{} | Exp. Hs={}".format(i, expHs))

# |%%--%%| <mWahMtVBFr|T6IJBZXUlT>
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
    path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("USPTO_50K", "Final_Graph")
    data = load_database(path)
    for item in data:
        if item["R-id"] == id:
            return item
    return None


entry = get_reaction_by_id("R26990")
print(entry)


# |%%--%%| <7zydJ3VZZW|mXrwYaTsEp>
from SynRBL.SynMCS.MissingGraph.find_missing_graphs import FindMissingGraphs
import rdkit.Chem.rdmolfiles as rdmolfiles

mol1 = rdmolfiles.MolFromSmiles("CON(C)C(=O)C1CCN(Cc2ccccc2)CC1")
mol2 = rdmolfiles.MolFromSmiles("C[Mg+]")
mol3 = rdmolfiles.MolFromSmiles("CC(=O)C1CCN(Cc2ccccc2)CC1")
mol_h1 = rdmolfiles.MolFromSmiles("C")
mol_h2 = rdmolfiles.MolFromSmiles("C(=O)C1CCN(Cc2ccccc2)CC1")
result = FindMissingGraphs.find_missing_parts_pairs([mol1, mol2], [mol_h2, mol_h1])

print([rdmolfiles.MolToSmiles(m) for m in result[0]])
print(result[1])
print(result[2])

plot_reaction(data[84])
