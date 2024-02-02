import re
import io
import pandas as pd
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import rdkit.Chem.rdChemReactions as rdChemReactions
import PIL.Image as Image
from SynRBL.rsmi_utils import load_database

DATA_PATH_FMT = (
    "/homes/biertank/klaus/repos/SynRBL/Data/Validation_set/{}/MCS/MCS_Impute.json.gz"
)
FINAL_VAL_PATH = "/homes/biertank/klaus/Documents/final_validation.csv"
DATASETS = [
    "Jaworski",
    "golden_dataset",
    "USPTO_unbalance_class",
    "USPTO_random_class",
    "USPTO_diff",
]


def load_data(dataset):
    return load_database(DATA_PATH_FMT.format(dataset))


def load_final_validation():
    return pd.read_csv(FINAL_VAL_PATH)


def remove_atom_mapping(smiles: str) -> str:
    pattern = re.compile(r":\d+")
    smiles = pattern.sub("", smiles)
    pattern = re.compile(r"\[(?P<atom>(B|C|N|O|P|S|F|Cl|Br|I){1,2})(?:H\d?)?\]")
    smiles = pattern.sub(r"\g<atom>", smiles)
    return smiles


def normalize_smiles(smiles: str) -> str:
    if ">>" in smiles:
        return ">>".join([normalize_smiles(t) for t in smiles.split(">>")])
    elif "." in smiles:
        token = sorted(
            smiles.split("."),
            key=lambda x: (sum(1 for c in x if c.isupper()), sum(ord(c) for c in x)),
            reverse=True,
        )
        return ".".join([normalize_smiles(t) for t in token])
    else:
        return Chem.CanonSmiles(remove_atom_mapping(smiles))


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

def plot_reaction(diff, index):
    e = diff[index]
    smiles = [e["initial"], e["correct"], e["new"]]
    titles = ["Initial Reaction", "Marked as Correct", "New Imputation"]
    plot_reactions(smiles, titles, e["title"])

df = load_final_validation()
diff = []
for dataset in DATASETS:
    data = load_data(dataset)

    for i, item in enumerate(data):
        id = item["R-id"]
        df_index = df.index[df["R-id"] == id].to_list()
        if len(df_index) == 0:
            continue
        assert len(df_index) == 1
        df_index = df_index[0]
        df_row = df.iloc[df_index]
        is_correct = df_row["Result"]
        if is_correct:
            correct_reaction = normalize_smiles(df_row["correct_reaction"])
            reaction_result = normalize_smiles(item["new_reaction"])
            if correct_reaction != reaction_result:
                print(
                    "[{}:{},{}] Reaction was correct but differs now.".format(
                        id, i, len(diff)
                    )
                )
                initial_reaction = normalize_smiles(item["old_reaction"])
                diff.append(
                    {
                        "correct": correct_reaction,
                        "new": reaction_result,
                        "initial": initial_reaction,
                        "title": "{}:{}".format(id, i)
                    }
                )

print("Found {} reactions that were correct but differ now.".format(len(diff)))

# |%%--%%| <nwsAT431AF|z1FxZYTB7T>

index = 3
print(len(diff))
plot_reaction(diff, index)

