from SynRBL.rsmi_utils import load_database, filter_data
from SynRBL.SynMCS.mol_merge import merge, plot_mols, InvalidAtomDict
import matplotlib.pyplot as plt
from collections import defaultdict
from rdkit import Chem

reactions = load_database("./Data/MCS/Final_Graph_macth_3+.json.gz")
# |%%--%%| <dHaIny7QxH|C6tCEiQ7Tu>


def impute(data, idx, verbose=False):
    frags = [Chem.MolFromSmiles(s) for s in data[idx]["smiles"]]
    bounds = data[idx]["boundary_atoms_products"]
    neighbors = data[idx]["nearest_neighbor_products"]
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
            includeAtomNumbers=True,
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


s = 1000
n = 0
correct = []
incorrect = []
crules = defaultdict(lambda: [])
mrules = defaultdict(lambda: [])
for i in range(n, n + s):
    try:
        mmols = impute(reactions, i)
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

# |%%--%%| <C6tCEiQ7Tu|3fjJ9252Be>

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

# indices = crules['append O to Si'][0:3]
indices = crules["append O to C-C bond"][0:5]
# indices = mrules['halogen bond restriction']
# indices = mrules['silicium radical'][0:5]
# indices = random.choices(correct, k=5)
# indices = incorrect
indices = [24]

for i in indices:
    try:
        impute(reactions, i, verbose=True)
    except Exception as e:
        # import traceback
        # traceback.print_exc()
        print("[{}]".format(i), e)
