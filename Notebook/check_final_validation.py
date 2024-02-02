import pandas as pd
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynVis.reaction_visualizer import ReactionVisualizer
from SynRBL.SynUtils.chem_utils import remove_atom_mapping, normalize_smiles
import io
import PIL.Image as Image
import matplotlib.pyplot as plt
import rdkit.Chem.rdChemReactions as rdChemReactions
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D

DATA_PATH_FMT = "/homes/biertank/klaus/repos/SynRBL/Data/Validation_set/{}/MCS/MCS_Impute.json.gz"
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

df = load_final_validation()
for dataset in DATASETS:
    data = load_data(dataset)

    for i, item in enumerate(data):
        id = item["R-id"]
        df_index = df.index[df["R-id"] == id].to_list()
        assert len(df_index) == 1
        df_index = df_index[0]
        df_row = df.iloc[df_index]
        is_correct = df_row["Result"]
        if is_correct:
            correct_reaction = normalize_smiles(df_row["correct_reaction"])
            reaction_result = normalize_smiles(item["new_reaction"])
            if correct_reaction != reaction_result:
                print("[{}, {}] Reaction was correct but differs now.".format(i, id))
