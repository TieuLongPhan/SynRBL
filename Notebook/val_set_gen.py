import os
import pandas
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynVis.reaction_visualizer import ReactionVisualizer
from SynRBL.SynUtils.chem_utils import remove_atom_mapping, normalize_smiles
import io
import PIL.Image as Image
import matplotlib.pyplot as plt
import rdkit.Chem.rdChemReactions as rdChemReactions
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D

RESULT_PATH_FMT = "./Pipeline/Validation/Analysis/SynRBL - {}.csv"
DATA_PATH_FMT = "./Data/Validation_set/{}/MCS/MCS_Impute.json.gz"
VALSET_PATH_FMT = "./Data/Validation_set/{}/corrected.json.gz"
DATASETS = ["Jaworski", "USPTO_unbalance_class", "USPTO_random_class", "golden_dataset"]


def load_results(dataset):
    file = RESULT_PATH_FMT.format(dataset)
    return pandas.read_csv(file)


def load_data(dataset):
    return load_database(DATA_PATH_FMT.format(dataset))


def load_valset(dataset):
    path = VALSET_PATH_FMT.format(dataset)
    if os.path.exists(path):
        return load_database(path)
    else:
        return []


def save_valset(data, dataset):
    path = VALSET_PATH_FMT.format(dataset)
    save_database(data, path)


def get_by_id(data, id):
    for e in data:
        if e["R-id"] == id:
            return e
    return None


def build_validation_set(data, results):
    if len(results) != len(data):
        raise ValueError(
            "Data and results must be of same length. ({} != {})".format(
                len(results), len(data)
            )
        )
    vset = []
    for d, r in zip(data, results.iterrows()):
        _, row = r
        reaction = normalize_smiles(d["new_reaction"])
        correct_reaction = None
        wrong_reactions = []
        if row.Result:
            correct_reaction = reaction
        else:
            wrong_reactions.append(reaction)
        vset.append(
            {
                "R-id": d["R-id"],
                "reaction": d["old_reaction"],
                "correct_reaction": correct_reaction,
                "wrong_reactions": wrong_reactions,
            }
        )
    return vset


def merge_validation_sets(vset, new_vset):
    def _it(row):
        return row["correct_reaction"], row["wrong_reactions"]

    ovset = []
    ncnt = 0
    mcnt = 0
    for ne in new_vset:
        id = ne["R-id"]
        e = get_by_id(vset, id)
        if e is None:
            ovset.append(ne)
            ncnt += 1
        else:
            assert id == e["R-id"]
            cr, wrs = _it(e)
            ncr, nwrs = _it(ne)
            if ncr is not None:
                if len(cr) > 0 and cr != ncr:
                    print("[{}] Correct reaction changed.".format(id))
                    # e['correct_reaction'] = ncr
                    # mcnt += 1
                elif ncr in wrs:
                    print("[{}] New correct reaction was marked as wrong.".format(id))
            for nwr in nwrs:
                if len(nwr) > 0 and nwr not in wrs:
                    print("[{}] Found new wrong reaction.".format(id))
                    # e['wrong_reactions'].append(nwr)
                    # mcnt += 1
            ovset.append(e)
    print("Added {} new reactions and modified {} reactions.".format(ncnt, mcnt))
    return ovset

def get_reaction_img(smiles):
    rxn = rdChemReactions.ReactionFromSmarts(smiles, useSmiles=True)
    d = rdMolDraw2D.MolDraw2DCairo(2000, 500)
    d.DrawReaction(rxn)
    d.FinishDrawing()
    return Image.open(io.BytesIO(d.GetDrawingText()))


def plot_reactions(smiles, titles=None, suptitle=None):
    if not isinstance(smiles, list):
        smiles = [smiles]
    l = len(smiles)
    fig, axs = plt.subplots(l, 1, figsize=(10, l * 3))
    if suptitle is not None:
        fig.suptitle(suptitle, color="gray")
    if l == 1:
        axs = [axs]
    if titles is None:
        titles = ["" for _ in range(l)]
    for s, ax, title in zip(smiles, axs, titles):
        img = get_reaction_img(remove_atom_mapping(s))
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_reaction(data, index, new_data=None):
    smiles = [data[index]["reaction"], data[index]["correct_reaction"]]
    titles = ["Initial Reaction", '"Correct" Reaction']
    if new_data is not None:
        smiles.append(new_data[index]["correct_reaction"])
        titles.append("New Correct Reaction")
    plot_reactions(
        smiles,
        titles,
        suptitle="Reaction index: {}   ID: {}".format(index, data[index]["R-id"]),
    )

#|%%--%%| <fQfIoeoE9J|OqsCzC6wdl>

dataset = DATASETS[3]
save = False
# for dataset in DATASETS:
print("Start: {}.".format(dataset))
results = load_results(dataset)
data = load_data(dataset)
vset = load_valset(dataset)

new_vset = build_validation_set(data, results)

mvset = merge_validation_sets(vset, new_vset)
if save:
    save_valset(mvset, dataset)

# |%%--%%| <OqsCzC6wdl|Ubskix1QjQ>

plot_reaction(vset, 266, new_data=new_vset)
