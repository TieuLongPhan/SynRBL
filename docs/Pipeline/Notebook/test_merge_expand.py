from SynRBL.rsmi_utils import load_database, filter_data
from SynRBL.SynMCSImputer.mol_merge import merge, plot_mols, InvalidAtomDict
import matplotlib.pyplot as plt
from collections import defaultdict
from rdkit import Chem

mcs_data = load_database("./Data/MCS/Final_Graph_macth_3+.json.gz")
reactions = load_database(
    "./Data/MCS/Original_data_Intersection_MCS_3+_matching_ensemble.json.gz"
)
assert len(mcs_data) == len(reactions)
#|%%--%%| <HcDaOOboVX|HhPWmftW1t>
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from rdkit import Chem
from rdkit.Chem import Draw, rdChemReactions


def plot_mcs_summary(reaction_data, mcs_data, file=None, color="black"):
    fig = plt.figure(layout="constrained", figsize=(9, 9))
    fig.patch.set_facecolor("white")
    rows, cols = (3, 3)
    gs = GridSpec(rows, cols, figure=fig)
    ax_reaction = fig.add_subplot(gs[0, :])
    ax_reaction.set_title("Initial Reaction", color=color)

    reaction = rdChemReactions.ReactionFromSmarts(
        reaction_data["reactions"], useSmiles=True
    )
    reaction_img = Draw.ReactionToImage(reaction)
    ax_reaction.imshow(reaction_img)
    ax_reaction.axis("off")

    ax_idx = 0
    mols = [Chem.MolFromSmiles(s) for s in mcs_data["smiles"]]
    bounds = mcs_data["boundary_atoms_products"]
    neighbors = mcs_data["nearest_neighbor_products"]
    for i, mol in enumerate(mols):
        ax = fig.add_subplot(gs[1 + int(ax_idx / cols), ax_idx % cols])
        img = Draw.MolToImage(mol)
        ax.set_title(
            "Mol {}: {} {}".format(i + 1, bounds[i], neighbors[i]),
            fontsize=10,
            color=color,
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
        ax.set_title("Result {}".format(i + 1), color=color)
        ax.axis("off")
        ax_idx += 1

    if reaction_data["Unbalance"] == "Products":
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

    if file is not None:
        plt.savefig(file, facecolor=fig.get_facecolor())
    plt.show()


# |%%--%%| <HhPWmftW1t|C6tCEiQ7Tu>


def impute(mcs_data, reactions, idx, verbose=False):
    frags = [Chem.MolFromSmiles(s) for s in mcs_data[idx]["smiles"]]
    bounds = mcs_data[idx]["boundary_atoms_products"]
    neighbors = mcs_data[idx]["nearest_neighbor_products"]
    if verbose:
        print("----------")
        print(
            "Idx={}({}) boundary={} neighbors={}".format(
                idx, len(frags), bounds, neighbors
            )
        )
        for i, f in enumerate(frags):
            print("  Mol {}: {}".format(i, Chem.MolToSmiles(f)))
        plot_mols(
            frags,
            figsize=(4, 1),
            includeAtomNumbers=False,
            titles=["Input" for _ in range(len(frags))],
        )
        plt.show()
    mmol = merge(frags, bounds, neighbors)
    if verbose:
        plot_mols(
            [Chem.RemoveHs(m["mol"]) for m in mmol],
            figsize=(3, 1),
            titles=["Merged" for _ in range(len(mmol))],
        )
        plt.show()
    return mmol

#|%%--%%| <C6tCEiQ7Tu|B2osgdF1ll>

s = 100
n = 0
correct = []
incorrect = []
crules = defaultdict(lambda: [])
mrules = defaultdict(lambda: [])
for i in range(n, n + s):
    try:
        mmols = impute(mcs_data, reactions, i)
        for mmol in mmols:
            for r in mmol.get("compound_rules", []):
                crules[r.name].append(i)
            for r in mmol.get("merge_rules", []):
                mrules[r.name].append(i)
        correct.append(i)
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        print("[{}]".format(i), e)
        incorrect.append(i)

# |%%--%%| <B2osgdF1ll|3fjJ9252Be>

print("Compound rule useage:")

for k, v in crules.items():
    print("  Rule '{}' was used {} times.".format(k, len(v)))
print("Merge rule useage:")
for k, v in mrules.items():
    print("  Rule '{}' was used {} times.".format(k, len(v)))
print("Correct merges:", len(correct))
print("Extracted incorrect:", len(incorrect))

# |%%--%%| <3fjJ9252Be|A0A0bYceet>
import random
import numpy as np


n = 2
indices = crules["append O when next to O or N"]
# indices = crules['append O to Si']
# indices = crules["append O to C-C bond"]
# indices = mrules['halogen bond restriction']
# indices = mrules['silicium radical']
# indices = random.choices(correct, k=5)
# indices = incorrect
indices = random.choices(indices, k=np.min([len(indices), n]))
indices = [75]

for i in indices:
    try:
        impute(mcs_data, reactions, i, verbose=True)
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        print("[{}]".format(i), e)
# |%%--%%| <A0A0bYceet|CG6YuHk7mE>


def impute_new_reaction(reaction_data, mcs_data):
    mols = [Chem.MolFromSmiles(s) for s in mcs_data["smiles"]]
    bounds = mcs_data["boundary_atoms_products"]
    neighbors = mcs_data["nearest_neighbor_products"]
    merge_result = merge(mols, bounds, neighbors)
    m_smiles = []
    for i, result in enumerate(merge_result):
        mol = Chem.RemoveHs(result["mol"])
        m_smiles.append(Chem.MolToSmiles(mol))

    if reaction_data["Unbalance"] == "Products":
        reaction_smiles = [reaction_data["reactions"], *m_smiles]
    else:
        raise NotImplementedError(
            "Unbalance type '{}' is not implemented.".format(reaction_data["Unbalance"])
        )

    reaction_data["mcs_impute_reaction"] = ".".join(reaction_smiles)


for i in range(0, 100):
    try:
        impute_new_reaction(reactions[i], mcs_data[i])
    except Exception as e:
        print("[ERROR] [{}] {}".format(i, e))


# |%%--%%| <CG6YuHk7mE|UFSXR8zCrf>

idx = 75
export = True
export_file = "export_{}.png".format(reactions[idx]["id"]) if export else None
plot_mcs_summary(reactions[idx], mcs_data[idx])

# |%%--%%| <UFSXR8zCrf|j2FErxY9eI>
from SynRBL.SynVis import ReactionVisualizer

idx = 4
export = True

print('{')
for k, v in reactions[idx].items():
    val = "'{}'".format(v) if isinstance(v, str) else v
    print("  '{}': {}".format(k, val))
print('}')
export_file = "export_{}.jpg".format(reactions[idx]["id"]) if export else None
visualizer = ReactionVisualizer()
visualizer.plot_reactions(
    reactions[idx],
    "reactions",
    "mcs_impute_reaction",
    compare=True,
    savefig=export,
    pathname=export_file,
)

