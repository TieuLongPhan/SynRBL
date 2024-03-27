import itertools
from numpy import who
import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles

from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynVis.reaction_visualizer import ReactionVisualizer

GT_REACTION_KEY = "gt_reaction"

datasets = [
    "artificial_data_1",
    "artificial_data_2",
]


def plot_reaction(entry, show_atom_numbers=False, figsize=(10, 7.5)):
    visualizer = ReactionVisualizer(figsize=figsize)
    visualizer.plot_reactions(
        entry,
        GT_REACTION_KEY,
        "new_reaction",
        compare=True,
        show_atom_numbers=show_atom_numbers,
    )
    print("ID:", entry["R-id"])
    print("Issue:", entry["issue"])
    print("Rules:", entry["rules"])
    print("Boundaries:", entry["boundary_atoms_products"])
    print("Neighbors:", entry["nearest_neighbor_products"])

def get_gt_reaction(entry):
    reactants = entry["reactants"]
    products = entry["products"]
    gt = entry["ground truth"]
    cbc = entry["carbon_balance_check"]
    if cbc == "reactants":
        gt_reaction = "{}.{}>>{}".format(reactants, gt, products)
    elif cbc == "products":
        gt_reaction = "{}>>{}.{}".format(reactants, gt, products)
    else:
        raise ValueError("Invalid carbon balance check value '{}'.".format(cbc))
    return gt_reaction


def get_canonical_reaction_permutations(reaction_smiles):
    tokens = reaction_smiles.split(">>")
    r_smiles = [Chem.CanonSmiles(s) for s in tokens[0].split(".")]
    p_smiles = [Chem.CanonSmiles(s) for s in tokens[1].split(".")]
    reactions = []
    for rp in itertools.permutations(r_smiles):
        for pp in itertools.permutations(p_smiles):
            reactions.append("{}>>{}".format(".".join(rp), ".".join(pp)))
    return reactions


def append_gt_reactions(data):
    for entry in data:
        entry[GT_REACTION_KEY] = get_gt_reaction(entry)


def build_eval_dict(data):
    edict = {}
    for entry in data:
        gt_reaction = entry[GT_REACTION_KEY]
        edict[entry["R-id"]] = get_canonical_reaction_permutations(gt_reaction)
    return edict


def evaluate(mcs_impute, eval_dict):
    success_cnt = 0
    for entry in mcs_impute:
        id = entry["R-id"]
        exp_results = eval_dict[id]
        result = entry["new_reaction"]
        is_correct = False
        if result in exp_results:
            success_cnt += 1
            is_correct = True
        entry["eval"] = {"correct": is_correct, "expected": exp_results}
        entry[GT_REACTION_KEY] = exp_results[0]
    return {"success_cnt": success_cnt, "accuracy": success_cnt / len(mcs_impute)}


dataset = datasets[0]
gt_data_path = "./Data/Validation_set/{}/{}.json.gz".format(
    dataset, "mcs_based_reactions"
)
gt_data = load_database(gt_data_path)
append_gt_reactions(gt_data)
eval_dict = build_eval_dict(gt_data)

mcs_data_path = "./Data/Validation_set/{}/MCS/{}.json.gz".format(dataset, "MCS_Impute")
mcs_data = load_database(mcs_data_path)

eval_result = evaluate(mcs_data, eval_dict)
failed_results = [e for e in mcs_data if not e['eval']['correct']]
print(eval_result)

idx = 21
nr = failed_results[idx]['new_reaction']
for x, y in [(r, nr) for r in failed_results[idx]['eval']['expected']]:
    print('-----')
    print(x)
    print(y)
plot_reaction(failed_results[idx])
