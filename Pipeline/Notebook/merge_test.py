import rdkit.Chem as Chem
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.Draw as Draw
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import matplotlib.pyplot as plt

s = "CC(N)=O"  # "CC[Si](C)(C)C"  # "c1ccc(P(=O)(c2ccccc2)c2ccccc2)cc1"
s = Chem.CanonSmiles(s)
mol = rdmolfiles.MolFromSmiles(s)
for i, atom in enumerate(mol.GetAtoms()):
    atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
fig, ax = plt.subplots(1, 1, figsize=(10, 3))
img = Draw.MolToImage(mol)
ax.imshow(img)
ax.set_title(s)
ax.axis("off")
plt.show()

#|%%--%%| <PDHNfCjKgB|jNxzbfeXHa>
import os
import shutil
import itertools
import rdkit.Chem.Draw as Draw

def export_reaction_imgs(data, rule_map, n=3, path="./tmp/rule_samples"):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    visualizer = ReactionVisualizer(figsize=(16, 8))
    for rule, indices in rule_map.items():
        dir = os.path.join(path, rule)
        os.mkdir(dir)
        for index in itertools.islice(indices, n):
            print(rule, index)
            filename = os.path.join(dir, "{}.{}".format(data[index]["R-id"], "jpg"))
            visualizer.plot_reactions(
                data[index],
                "old_reaction",
                "new_reaction",
                savefig=True,
                pathname=filename,
                compare=True,
                show_atom_numbers=True,
            )

export_reaction_imgs(data, rule_map)

#|%%--%%| <jNxzbfeXHa|FtdgkzRxV9>

index = 866
print(data[index])
visualizer = ReactionVisualizer(figsize=(10, 10))
visualizer.plot_reactions(
    data[index], "old_reaction", "new_reaction", compare=True, show_atom_numbers=True
)
print("Rules:", data[index]["rules"])

#|%%--%%| <FtdgkzRxV9|T6IJBZXUlT>
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.Draw as Draw
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import rdkit.Chem.rdChemReactions as rdChemReactions
import matplotlib.pyplot as plt

print(matches[0:5])
index = 104
reaction_data = data[index]
reactant = rdmolfiles.MolFromSmiles(reaction_data["reactants"])
print(rdmolfiles.MolToSmiles(reactant))
reaction = rdChemReactions.ReactionFromSmarts(
    reaction_data["reactions"], useSmiles=True
)
fig, ax = plt.subplots(2, 1, figsize=(1, 1.5))
img = Draw.ReactionToImage(reaction)
ax[0].imshow(img)
for i, atom in enumerate(reactant.GetAtoms()):
    atom.SetProp("molAtomMapNumber", str(atom.GetIdx()))
img = Draw.MolToImage(reactant)
ax[1].imshow(img)

