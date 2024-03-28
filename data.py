import io
import matplotlib.pyplot as plt
from PIL import Image
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import rdkit.Chem.rdChemReactions as rdChemReactions

path = "/home/klaus/Documents/Ecoli_corrAAM_forATN.txt"


def plot_rxn(smiles):
    r = rdChemReactions.ReactionFromSmarts(smiles, useSmiles=True)
    d = rdMolDraw2D.MolDraw2DCairo(1600, 900)
    d.DrawReaction(r)
    d.FinishDrawing()
    img = Image.open(io.BytesIO(d.GetDrawingText()))

    plt.imshow(img)
    plt.axis("off")
    plt.show()


ecoli_reactions = []
with open(path, "r") as f:
    for i, line in enumerate(f.readlines()):
        c = i % 4
        if c == 1 or c == 2:
            ecoli_reactions.append(line.replace("\n", ""))

# |%%--%%| <IwYFjlReDK|w5kVVS108c>
import pandas as pd
import collections

stats = collections.defaultdict(lambda: 0)
benchmark_set = []
for n, r_smiles in enumerate(ecoli_reactions):
    token = r_smiles.split(">>")
    reactants = [t for t in token[0].split(".") if len(t) > 0]
    products = [t for t in token[1].split(".") if len(t) > 0]
    stats[len(reactants)] += 1
    for i in range(len(reactants)):
        _reactant_str = ".".join([r for r in reactants])
        _product_str = ".".join([p for j, p in enumerate(products) if i != j])
        benchmark_reaction = "{}>>{}".format(_reactant_str, _product_str)
        benchmark_set.append(
            {
                "reaction": benchmark_reaction,
                "expected_reaction": r_smiles,
            }
        )


print("Nr. of products: {}".format(dict(stats)))
df = pd.DataFrame(benchmark_set)
print(len(df))
df.to_csv("ecoli_benchmark.csv")
