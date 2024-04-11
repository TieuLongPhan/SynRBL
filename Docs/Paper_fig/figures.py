import matplotlib.pyplot as plt

from synrbl.SynVis import ReactionVisualizer
from synrbl.rsmi_utils import load_database
from synrbl.SynUtils.chem_utils import remove_atom_mapping


def clear_atom_nums(dataset):
    for k in ["new_reaction", "old_reaction"]:
        for i in range(len(dataset)):
            dataset[i][k] = remove_atom_mapping(dataset[i][k])


path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("USPTO_50K", "MCS_Impute")
data = load_database(path)
clear_atom_nums(data)

# |%%--%%| <PE8YO5X0qq|qiP5pkTCwv>

# Dataset                      Reactions   C balanced      Rate
# -------------------------------------------------------------
# golden_dataset                     887          530    59.75%
# Jaworski                           154          133    86.36%
# artificial_data_1                 1177         1113    94.56%
# artificial_data_2                 1178         1124    95.42%
# USPTO_50K                        16728        16721    99.96%

# |%%--%%| <qiP5pkTCwv|DUqrOw4SKU>


def get_ids_by_rule(data, rule, n_rules=None):
    ids = []
    for i, e in enumerate(data):
        if rule in e["rules"]:
            if n_rules is None or len(e["rules"]) == n_rules:
                ids.append(i)
    return ids


def get_reaction_by_id(data, id):
    for i, item in enumerate(data):
        if item["R-id"] == id:
            return i, item
    return None


rids = get_ids_by_rule(data, "phosphor double bond", n_rules=1)

# |%%--%%| <DUqrOw4SKU|5Fdy81dKgB>

ridl = []
for idx in rids:
    ridl.append((idx, len(data[idx]["new_reaction"])))

ridl = sorted(ridl, key=lambda e: e[1])
print(ridl[0:20])
# idx, _ = get_reaction_by_id(data, "golden_dataset_568")
idx = 198
print(idx, data[idx]["issue"], data[idx]["rules"])
rvis = ReactionVisualizer(figsize=(10, 8))
rvis.plot_reactions(
    data[idx],
    "old_reaction",
    "new_reaction",
    compare=True,
    show_atom_numbers=False,
    # new_reaction_title="Balanced Reaction",
    # old_reaction_title="Initial Reaction",
)

# |%%--%%| <5Fdy81dKgB|w3XZRXxvhi>

export_config = [
    (9453, "phosphor_bond"),
    (8346, "bond_restriction"),
    (7156, "single_bond"),
    (11066, "C-O_ether_break"),
    (8128, "C-O_ester_break"),
    (4944, "C-N_amide_break"),
    (11884, "form_M-OH"),
    (10382, "append_O"),
    (1282, "append_O_CC_break"),
]

path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("USPTO_50K", "MCS_Impute")
_data = load_database(path)
clear_atom_nums(_data)

for idx, fname in export_config:
    rvis = ReactionVisualizer(figsize=(8, 4))
    rvis.plot_reactions(
        _data[idx],
        "old_reaction",
        "new_reaction",
        compare=True,
        savefig=True,
        pathname="./figs/{}.png".format(fname),
        show_atom_numbers=False,
        new_reaction_title="Balanced Reaction",
        old_reaction_title="Initial Reaction",
    )
# |%%--%%| <w3XZRXxvhi|vk7u8hsFMR>

plt.rcParams.update(
    {
        # "figure.facecolor":  (1.0, 0.0, 0.0, 0.3),  # red   with alpha = 30%
        "axes.facecolor": (0.0, 1.0, 1.0, 1.0),  # green with alpha = 50%
        # "savefig.facecolor": (0.0, 0.0, 1.0, 0.2),  # blue  with alpha = 20%
    }
)

export_config = [(140, "oxidation_reaction"), (172, "ring_formation")]

path = "./Data/Validation_set/{}/MCS/{}.json.gz".format("golden_dataset", "MCS_Impute")
_data = load_database(path)
clear_atom_nums(_data)

for idx, fname in export_config:
    rvis = ReactionVisualizer(figsize=(8, 4))
    rvis.plot_reactions(
        _data[idx],
        "old_reaction",
        "new_reaction",
        compare=False,
        savefig=True,
        pathname="./figs/{}.png".format(fname),
        show_atom_numbers=False,
        new_reaction_title="",
        old_reaction_title="",
    )
