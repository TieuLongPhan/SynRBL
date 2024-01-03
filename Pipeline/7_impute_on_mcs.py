import traceback
from SynRBL.rsmi_utils import load_database, save_database
from rdkit import Chem
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from rdkit import Chem
from rdkit.Chem import Draw, rdChemReactions
from collections import defaultdict


def plot_summary(reaction_data):
    fig = plt.figure(layout="constrained", figsize=(9, 9))
    fig.patch.set_facecolor("white")
    rows, cols = (3, 3)
    gs = GridSpec(rows, cols, figure=fig)
    ax_reaction = fig.add_subplot(gs[0, :])
    ax_reaction.set_title("Initial Reaction")

    reaction = rdChemReactions.ReactionFromSmarts(
        reaction_data["reactions"], useSmiles=True
    )
    reaction_img = Draw.ReactionToImage(reaction)
    ax_reaction.imshow(reaction_img)
    ax_reaction.axis("off")

    ax_idx = 0
    mcs_data = reaction_data["missing_parts"]
    mols = [Chem.MolFromSmiles(s) for s in mcs_data["smiles"]]
    bounds = mcs_data["boundary_atoms_products"]
    neighbors = mcs_data["nearest_neighbor_products"]
    for i, mol in enumerate(mols):
        ax = fig.add_subplot(gs[1 + int(ax_idx / cols), ax_idx % cols])
        img = Draw.MolToImage(mol)
        ax.set_title(
            "Mol {}: {} {}".format(i + 1, bounds[i], neighbors[i]),
            fontsize=10,
        )
        ax.imshow(img)
        ax.axis("off")
        ax_idx += 1

    merge_result = merge(mols, bounds, neighbors)
    m_smiles = []
    for i, result in enumerate(merge_result):
        mol = Chem.RemoveHs(result["mol"])
        m_smiles.append(Chem.MolToSmiles(mol))
        ax = fig.add_subplot(gs[1 + int(ax_idx / cols), ax_idx % cols])
        img = Draw.MolToImage(mol)
        ax.imshow(img)
        ax.set_title("Result {}".format(i + 1))
        ax.axis("off")
        ax_idx += 1

    if reaction_data["Unbalance"] in ["Products", "Both"]:
        reaction_smiles = [reaction_data["reactions"], *m_smiles]
    else:
        raise NotImplementedError(
            "Unbalance type '{}' is not implemented.".format(reaction_data["Unbalance"])
        )
    ax_new_reaction = fig.add_subplot(gs[2, :])
    reaction = rdChemReactions.ReactionFromSmarts(
        ".".join(reaction_smiles), useSmiles=True
    )
    reaction_img = Draw.ReactionToImage(reaction)
    ax_new_reaction.imshow(reaction_img)
    ax_new_reaction.axis("off")
    plt.show()


def load_data(dataset="3+"):
    if dataset == "3+":
        mcs_data = load_database("./Data/MCS/Final_Graph_macth_3+.json.gz")
        reactions = load_database(
            "./Data/MCS/Original_data_Intersection_MCS_3+_matching_ensemble.json.gz"
        )
    elif dataset == "0-50":
        mcs_data = load_database("./Data/MCS/Final_Graph_macth_under2-.json.gz")
        reactions = load_database(
            "./Data/MCS/Original_data_Intersection_MCS_0_50_largest.json.gz"
        )
    else:
        raise ValueError("Unknown dataset '{}'.".format(dataset))
    if len(mcs_data) != len(reactions):
        raise ValueError(
            "Graph data and reaction data must be of same length. ({} != {})".format(
                len(mcs_data), len(reactions)
            )
        )
    for i, mcs_item in enumerate(mcs_data):
        reactions[i]["missing_parts"] = mcs_item
    return reactions


def impute(reaction):
    mcs_data = reaction["missing_parts"]
    mols = [Chem.MolFromSmiles(s) for s in mcs_data["smiles"]]
    if len(mcs_data["smiles"]) == 0:
        # Skip reactions where finding MCS timed out.
        raise ValueError("Empty substructure.")
    bounds = mcs_data["boundary_atoms_products"]
    neighbors = mcs_data["nearest_neighbor_products"]
    merge_result = merge(mols, bounds, neighbors)
    m_smiles = []
    compound_rules = []
    merge_rules = []
    for result in merge_result:
        mol = Chem.RemoveHs(result["mol"])
        m_smiles.append(Chem.MolToSmiles(mol))
        if "compound_rules" in result.keys():
            compound_rules.extend([r.name for r in result["compound_rules"]])
        if "merge_rules" in result.keys():
            merge_rules.extend([r.name for r in result["merge_rules"]])

    if reaction["Unbalance"] in ["Products", "Both"]:
        reaction_smiles = [reaction["reactions"], *m_smiles]
    else:
        raise NotImplementedError(
            "Unbalance type '{}' is not implemented.".format(reaction["Unbalance"])
        )

    reaction["structure_impute_reaction"] = ".".join(reaction_smiles)
    reaction["compound_rules"] = compound_rules
    reaction["merge_rules"] = merge_rules
    reaction["old_reaction"] = mcs_data["old_reaction"]


def get_export_dict(
    reactions,
    export_key_map={
        "id": "id",
        "R-id": "R-id",
        "old_reaction": "old_reaction",
        "structure_impute_reaction": "new_reaction",
        "merge_rules": "merge_rules",
        "compound_rules": "compound_rules",
        "issue": "issue",
    },
):
    export = []
    for reaction in reactions:
        export_reaction = {}
        for k, v in export_key_map.items():
            if k in reaction.keys():
                export_reaction[v] = reaction[k]
            else:
                export_reaction[v] = None
        export.append(export_reaction)
    return export


def main():
    dataset = "3+"  # "0-50"
    reactions = load_data(dataset)
    print(reactions[0])
    failed = []
    missing_parts_lengths = defaultdict(lambda: 0)
    for i in range(len(reactions)):
        reaction = reactions[i]
        k = len(reaction["reactions"].split("."))
        print(reaction)
        break
        mol1 = Chem.MolFromSmiles(reaction["reactants"])
        mol2 = Chem.MolFromSmiles(reaction["products"])
        plot_mols([mol1, mol2], includeAtomNumbers=True)
        missing_parts = reaction["missing_parts"]
        # print(len(missing_parts['smiles']))
        missing_parts_lengths[k] += 1
        break
    print(missing_parts_lengths)
    plot_summary(reaction)
    return
    # try:
    #    impute(reaction)
    # except Exception as e:
    #    #traceback.print_exc()
    #    failed.append(i)
    #    reaction["issue"] = [str(e)]
    #    print("[ERROR] [{}] {}".format(i, e))

    # export_reactions = get_export_dict(reactions)
    # save_database(
    #    export_reactions,
    #    "./Data/MCS/After_Merge_and_Expansion_{}.json.gz".format(dataset),
    # )
    return
    id = 0
    try:
        plot_summary(reactions[id])
    except Exception as e:
        print("[ERROR] [{}] {}".format(id, e))
    for i, s in enumerate(reactions[id]["missing_parts"]["smiles"]):
        print("Mol {}: {}".format(i + 1, s))


if __name__ == "__main__":
    main()

# |%%--%%| <OWu15Qd2s0|ntYKIvfOQs>
import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.Draw as Draw
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import matplotlib.pyplot as plt

s = "CC(N)=O" # "CC[Si](C)(C)C"  # "c1ccc(P(=O)(c2ccccc2)c2ccccc2)cc1"
s = Chem.CanonSmiles(s)
print(s)
mol = rdmolfiles.MolFromSmiles(s)
match = mol.GetSubstructMatch(rdmolfiles.MolFromSmiles("C=O"))
print(match)
for i, atom in enumerate(mol.GetAtoms()):
    atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
img = Draw.MolToImage(mol)
plt.imshow(img)
plt.show()
# |%%--%%| <ntYKIvfOQs|4Efw41ErNz>
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynVis.reaction_visualizer import ReactionVisualizer
from SynRBL.SynMCS.structure import Compound
from SynRBL.SynMCS.merge import merge, NoCompoundRule

def get_reactant_src(reaction, smiles, neighbors):
    match_mol = rdmolfiles.MolFromSmiles(smiles)
    reactant = reaction.split(">>")[0]
    matches = []
    for compound in reactant.split("."):
        c_mol = rdmolfiles.MolFromSmiles(compound)
        match = c_mol.GetSubstructMatch(match_mol)
        if len(match) > 0:
            #found_all_neighbors = True
            #for n in neighbors:
            #    for ni_s, ni_i in n.items():
            #        if c_mol.GetAtomWithIdx(ni_i).GetSymbol() != ni_s:
            #            found_all_neighbors = False
            #            break
            #if found_all_neighbors:
            matches.append(compound)
    if len(matches) != 1:
        raise ValueError("Found {} matches for structure '{}'.".format(len(matches), smiles))
    return matches[0]


def build_compounds(item):
    reaction = item['old_reaction']
    smiles = item['smiles']
    boundaries = item['boundary_atoms_products']
    neighbors = item['nearest_neighbor_products']
    if len(smiles) != len(neighbors) or len(smiles) != len(boundaries):
        print(smiles, neighbors, boundaries)
        raise ValueError("Unequal leghts.")
    compounds = []
    for s, b, n in zip(smiles, boundaries, neighbors):
        src_mol = get_reactant_src(reaction, s, n)
        c = Compound(s, src_mol=src_mol) 
        if len(b) != len(n):
            raise ValueError("Boundary and neighbor missmatch.")
        for bi, ni in zip(b, n):
            bi_s, bi_i = list(bi.items())[0]
            ni_s, ni_i = list(ni.items())[0]
            c.add_boundary(bi_i, symbol=bi_s, neighbor_index=ni_i, neighbor_symbol=ni_s)
        compounds.append(c)
    return compounds


dataset = "USPTO_50K"
data = load_database("./Data/Validation_set/{}/MCS/Final_Graph.json.gz".format(dataset))
for i, item in enumerate(data):
    if data[i]['issue'] != '':
        print("[ERROR] [{}]".format(i), data[i]['issue'])
        continue
    data[i]["rules"] = []
    data[i]["new_reaction"] = data[i]["old_reaction"]
    try:
        compounds = build_compounds(item)
        result = merge(compounds)
        new_reaction = "{}.{}".format(item['old_reaction'], result.smiles)
        data[i]["new_reaction"] = new_reaction
        data[i]["rules"] = [r.name for r in result.rules]
    except NoCompoundRule as e:
        #print("[WARN] [{}]".format(i), e)
        pass
    except Exception as e:
        data[i]["issue"] = str(e)
        print("[ERROR] [{}]".format(i), e)

#|%%--%%| <4Efw41ErNz|gcHol4kj8O>

index = 8 
print(data[index])
visualizer = ReactionVisualizer(figsize=(10, 10))
visualizer.plot_reactions(data[index], "old_reaction", "new_reaction", compare=True, show_atom_numbers=True)
print("Rules:", data[index]['rules'])


# |%%--%%| <gcHol4kj8O|mqphgzX5mM>

import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.Draw as Draw
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import rdkit.Chem.rdChemReactions as rdChemReactions
import matplotlib.pyplot as plt

print(matches[0:5])
index = 104
reaction_data = data[index]
reactant = rdmolfiles.MolFromSmiles(reaction_data["reactants"])
print(rdmolfiles.MolToSmiles(reactant))
reaction = rdChemReactions.ReactionFromSmarts(reaction_data["reactions"], useSmiles=True)
fig, ax = plt.subplots(2, 1, figsize=(1, 1.5))
img = Draw.ReactionToImage(reaction)
ax[0].imshow(img)
for i, atom in enumerate(reactant.GetAtoms()):
    atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
img = Draw.MolToImage(reactant)
ax[1].imshow(img)
