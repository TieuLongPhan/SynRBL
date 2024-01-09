import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.Draw as Draw
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import matplotlib.pyplot as plt
from SynRBL.SynVis.reaction_visualizer import ReactionVisualizer

def plot_reaction(entry):
    visualizer = ReactionVisualizer(figsize=(10, 10))
    visualizer.plot_reactions(
        entry, "old_reaction", "new_reaction", compare=True, show_atom_numbers=False
    )
    print("ID:", entry["R-id"])
    print("Imbalance:", entry['carbon_balance_check'])
    print("Smiles:", entry["smiles"])
    print("Sorted Reactants:", entry["sorted_reactants"])
    print("MCS:", [len(x) for x in entry["mcs_results"]])
    print("Boundaries:", entry["boundary_atoms_products"])
    print("Neighbors:", entry["nearest_neighbor_products"])
    print("Issue:", entry["issue"])
    print("Rules:", entry["rules"])
    print("Compounds:")
    for c in entry['compounds']:
        print("  Smiles: {}".format(c.smiles))
        for b in c.boundaries:
            print("    Boundary: {}   Neighbor: {}".format(b.symbol, b.neighbor_symbol))

s = "NC(=O)CC(N)C(=O)O"  # "CC[Si](C)(C)C"  # "c1ccc(P(=O)(c2ccccc2)c2ccccc2)cc1"
s = Chem.CanonSmiles(s)
print(s)
mol = rdmolfiles.MolFromSmiles(s)
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


# |%%--%%| <PDHNfCjKgB|IXhnkWIUcu>
import traceback
import SynRBL.SynMCSImputer.merge as merge
import SynRBL.SynMCSImputer.structure as structure
import SynRBL.SynMCSImputer.utils as utils
import copy


def impute_new_reaction(data):
    rule_map = {r.name: set() for r in CompoundRule.get_all() + MergeRule.get_all()}
    rule_map["no rule"] = set()
    for i, item in enumerate(data):
        item["rules"] = []
        new_reaction = item["old_reaction"]
        #if item["issue"] != "":
        #    print("[ERROR] [{}]".format(i), "Skip because of previous issue.")
        #    continue
        try:
            compounds = structure.build_compounds(item)
            item['compounds'] = copy.deepcopy(compounds) 
            result = merge.merge(compounds)
            imbalance = item['carbon_balance_check']
            if imbalance == 'products':
                new_reaction = "{}.{}".format(item["old_reaction"], result.smiles)
            elif imbalance == 'reactants':
                new_reaction = "{}.{}".format(result.smiles, item["old_reaction"])
            elif imbalance == 'balanced':
                #print("[INFO] [{}] Reaction is balanced.".format(i))
                pass
            else:
                raise ValueError("Carbon balance '{}' is not known.".format(imbalance))
            item["new_reaction"] = new_reaction
            rules = [r.name for r in result.rules]
            item["rules"] = rules
            if len(rules) == 0:
                rule_map["no rule"].add(i)
            else:
                for r in item["rules"]:
                    rule_map[r].add(i)
            utils.carbon_equality_check(new_reaction)
        except Exception as e:
            #traceback.print_exc()
            item["issue"] = str(e)
            print("[ERROR] [{}]".format(i), e)
        finally:
            item['new_reaction'] = new_reaction
    return rule_map


#|%%--%%| <IXhnkWIUcu|WavXFkZceG>
from SynRBL.rsmi_utils import load_database, save_database

path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("Jaworski", "Final_Graph")
data = load_database(path)

# |%%--%%| <WavXFkZceG|tx0z4CFgIc>
from SynRBL.SynMCSImputer.rules import CompoundRule, MergeRule
impute_new_reaction(data)
rule_map = {r.name: set() for r in CompoundRule.get_all() + MergeRule.get_all()}
rule_map["no rule"] = set()
for i, item in enumerate(data):
    if "rules" not in item.keys() or len(item["rules"]) == 0:
        rule_map["no rule"].add(i)
    else:
        for r in item["rules"]:
            rule_map[r].add(i)


def print_rule_summary(rule_map):
    for rule, ids in rule_map.items():
        print("{:<30} {}".format(rule, len(ids)))


print_rule_summary(rule_map)

# |%%--%%| <tx0z4CFgIc|FtdgkzRxV9>

print(data[5]['new_reaction'])
data[5]['new_reaction'] = "OCC(O)CC(O)O.O=CCCC=O>>OC1CC2C=C(CC2O1)C=O"
plot_reaction(data[5])

# |%%--%%| <FtdgkzRxV9|dZjkHmncQW>
import collections

error_map = collections.defaultdict(lambda: [])
for i, r in enumerate(data):
    err = r["issue"]
    if len(err) > 0:
        error_map[err[:20]].append(i)

for k, v in error_map.items():
    print("{:<20} {:>4} {}".format(k, len(v), v[:10]))


# |%%--%%| <dZjkHmncQW|mWahMtVBFr>
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
