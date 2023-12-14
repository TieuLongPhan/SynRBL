import traceback
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCS.mol_merge import merge
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
    dataset = "3+" #"0-50"
    reactions = load_data(dataset)[0:10]
    print(reactions[0])
    failed = []
    missing_parts_lengths = defaultdict(lambda: 0)
    for i in range(len(reactions)):
        reaction = reactions[i]
        missing_parts = reaction['missing_parts']
        print(reaction['Unbalance'])
        print(reaction['products'])
        missing_parts_lengths[len(missing_parts['smiles'])] += 1
        continue
    print(missing_parts_lengths)
    return
        try:
            impute(reaction)
        except Exception as e:
            # traceback.print_exc()
            failed.append(i)
            reaction["issue"] = [str(e)]
            print("[ERROR] [{}] {}".format(i, e))

    #export_reactions = get_export_dict(reactions)
    #save_database(
    #    export_reactions,
    #    "./Data/MCS/After_Merge_and_Expansion_{}.json.gz".format(dataset),
    #)
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
    
