import os
import re
import json
import pandas as pd
from SynRBL.rsmi_utils import load_database
import io
import matplotlib.pyplot as plt
import rdkit.Chem as Chem
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import rdkit.Chem.rdChemReactions as rdChemReactions
import PIL.Image as Image
from SynRBL.rsmi_utils import load_database

FINAL_VALIDATION_PATH = "./Pipeline/Validation/Analysis/final_validation.csv"
# FINAL_VALIDATION_PATH = "/homes/biertank/klaus/Documents/final_validation.csv"
DATASET_PATH_FMT = "./Data/Validation_set/{}/MCS/MCS_Impute.json.gz"
SNAPSHOT_PATH = "./Data/Validation_set/snapshot.json"
# SNAPSHOT_PATH = "/homes/biertank/klaus/Documents/snapshot.json"
DATASETS = [
    "Jaworski",
    "golden_dataset",
    "USPTO_unbalance_class",
    "USPTO_random_class",
    "USPTO_diff",
]


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


def plot_reactions(smiles, titles=None, suptitle=None, filename=None, dpi=300):
    if not isinstance(smiles, list):
        smiles = [smiles]
    l = len(smiles)
    fig, axs = plt.subplots(l, 1, figsize=(10, l * 3), dpi=dpi)
    if suptitle is not None:
        fig.suptitle(suptitle, color="gray")
    if l == 1:
        axs = [axs]
    if titles is None:
        titles = ["" for _ in range(l)]
    for s, ax, title in zip(smiles, axs, titles):
        img = get_reaction_img(s)
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)
    else:
        plt.show()


def plot_reaction(item, path=None, dpi=300):
    smiles = [normalize_smiles(item["initial_reaction"])]
    titles = ["Initial Reaction"]
    correct_r = item["correct_reaction"]
    checked_r = item["checked_reaction"]
    if correct_r is not None:
        smiles.append(normalize_smiles(correct_r))
        titles.append("Correct Reaction")
    elif checked_r is not None:
        smiles.append(normalize_smiles(checked_r))
        titles.append("Checked but WRONG")
    smiles.append(normalize_smiles(item["result_reaction"]))
    titles.append("New Imputation")
    filename = None 
    if path is not None:
        if not os.path.exists(path):
            os.makedirs(path)
        filename = os.path.join(path, "{}.jpg".format(item["R-id"]))
    plot_reactions(smiles, titles, item["R-id"], filename=filename, dpi=300)

def load_data(dataset):
    data = load_database(os.path.abspath(DATASET_PATH_FMT.format(dataset)))
    df = pd.read_csv(FINAL_VALIDATION_PATH)
    with open(SNAPSHOT_PATH, "r") as f:
        snapshot = json.load(f)
    return data, df, snapshot

def verify_dataset(dataset):
    def _fmt(id, initial_r, result_r, correct_r=None, checked_r=None):
        return {
            "initial_reaction": initial_r,
            "result_reaction": result_r,
            "correct_reaction": correct_r,
            "checked_reaction": checked_r,
            "R-id": id,
        }

    data, df, snapshot = load_data(dataset)

    new_wrong = []
    new_unknown = []
    for item in data:
        id = item["R-id"]
        df_index = df.index[df["R-id"] == id].to_list()
        if len(df_index) == 0:
            continue
        assert len(df_index) == 1
        sn_item = snapshot[id]
        assert id in snapshot.keys(), "Id not in snapshot."
        df_index = df_index[0]
        df_row = df.iloc[df_index]
        is_correct = df_row["Result"]
        initial_reaction = df_row["reactions"]
        result_reaction = item["new_reaction"]
        result_reaction_n = normalize_smiles(result_reaction)
        if is_correct:
            correct_reaction = df_row["correct_reaction"]
            if result_reaction_n != normalize_smiles(correct_reaction):
                new_wrong.append(
                    _fmt(
                        id,
                        initial_reaction,
                        result_reaction,
                        correct_r=correct_reaction,
                    )
                )
        else:
            wrong_reactions = sn_item["wrong_reactions"]
            wrong_reactions_n = [normalize_smiles(r) for r in wrong_reactions]
            if result_reaction_n not in wrong_reactions_n:
                new_unknown.append(
                    _fmt(
                        id,
                        initial_reaction,
                        result_reaction,
                        checked_r=wrong_reactions[0],
                    )
                )
    return new_wrong, new_unknown

new_wrong, new_unknown = [], []
for dataset in DATASETS:
    w, u = verify_dataset(dataset)
    new_wrong.extend(w)
    new_unknown.extend(u)

good = True
if len(new_wrong) > 0:
    print(
        "[WARNING] {} reactions that were correct are now wrong!".format(
            len(new_wrong)
        )
    )
    good = False
if len(new_unknown) > 0:
    print(
        (
            "[INFO] {} reactions that were wrong changed. "
            + "Please check if they are correct now."
        ).format(len(new_unknown))
    )
    good = False
if good:
    print("[INFO] All good!")


# |%%--%%| <5Q5RDfNoyT|L868S7wclK>

path = "./test_export/"
for item in new_unknown:
    plot_reaction(item, path=path)
for item in new_wrong:
    plot_reaction(item, path=path)

#|%%--%%| <L868S7wclK|wGrhho1wW7>

def load_reaction_data(id):
    for dataset in DATASETS:
        data, df, snapshot = load_data(dataset)
        for item in data:
            _id = item["R-id"]
            if id == _id:
                assert id in snapshot.keys(), "Id not in snapshot."
                df_index = df.index[df["R-id"] == id].to_list()
                assert len(df_index) == 1
                df_index = df_index[0]
                return item, df, df_index, snapshot
    raise KeyError("Reaction '{}' not found.".format(id))


def set_reaction_correct(id, save=False):
    item, df, df_index, snapshot = load_reaction_data(id)
    row = df.iloc[df_index]
    correct_reaction = item["new_reaction"]
    if row["Result"] == True:
        raise RuntimeError("Reaction '{}' is already marked correct.".format(id))
    with open(SNAPSHOT_PATH, "r") as f:
        snapshot = json.load(f)
    if id not in snapshot.keys():
        raise KeyError("Reaction '{}' not found in snapshot.".format(id))
    df = pd.read_csv(FINAL_VALIDATION_PATH)
    df.at[df_index, "correct_reaction"] = correct_reaction
    df.at[df_index, "Result"] = True
    snapshot[id]["checked_reaction"] = correct_reaction
    if save:
        df.to_csv(FINAL_VALIDATION_PATH)
        with open(SNAPSHOT_PATH, "w") as f:
            json.dump(snapshot, f, indent=4)
    
def set_reaction_wrong(id, save=False):
    item, df, df_index, snapshot = load_reaction_data(id)
    row = df.iloc[df_index]
    wrong_reaction = item["new_reaction"]
    if row["Result"] == True:
        raise RuntimeError("Reaction '{}' has already a correct result.".format(id))
    with open(SNAPSHOT_PATH, "r") as f:
        snapshot = json.load(f)
    if id not in snapshot.keys():
        raise KeyError("Reaction '{}' not found in snapshot.".format(id))
    df = pd.read_csv(FINAL_VALIDATION_PATH)
    snapshot[id]["wrong_reactions"].insert(0, wrong_reaction)
    if save:
        with open(SNAPSHOT_PATH, "w") as f:
            json.dump(snapshot, f, indent=4)

set_reaction_wrong("Jaworski_221")

# |%%--%%| <wGrhho1wW7|WaEMB9CVpa>

def create_snapshot(data, df, snapshot=None):
    snapshot = {} if snapshot is None else snapshot
    for i, item in enumerate(data):
        id = item["R-id"]
        df_index = df.index[df["R-id"] == id].to_list()
        if len(df_index) == 0:
            continue
        assert len(df_index) == 1
        df_index = df_index[0]
        df_row = df.iloc[df_index]
        is_correct = df_row["Result"]
        result_reaction = item["new_reaction"]

        if is_correct:
            correct_reaction = df_row["correct_reaction"]
            sn_item = {"checked_reaction": correct_reaction, "wrong_reactions": []}
        else:
            sn_item = {
                "checked_reaction": result_reaction,
                "wrong_reactions": [result_reaction],
            }

        snapshot[id] = sn_item
    return snapshot


dataset = DATASETS[4]
data = load_dataset(dataset)
df = pd.read_csv(FINAL_VALIDATION_PATH)

with open(SNAPSHOT_PATH, "r") as f:
    snapshot = json.load(f)

snapshot = create_snapshot(data, df, snapshot)
print("Snapshot entries: {}".format(len(snapshot)))

# with open(SNAPSHOT_PATH, "w") as f:
#    json.dump(snapshot, f, indent=4)
