import synrbl
import rdkit.Chem.rdmolfiles as rdmolfiles

import rdkit.Chem.Draw as Draw
import matplotlib.pyplot as plt
import rdkit.Chem.rdchem as rdchem
import rdkit.Chem.rdmolops as rdmolops

from .rxnvis import RxnVis


def get_mcs_mol(mol, mcs_indices):
    rwmol = rdchem.EditableMol(mol)
    atoms_to_remove = sorted(
        [a.GetIdx() for a in mol.GetAtoms() if a.GetIdx() not in mcs_indices],
        reverse=True,
    )
    for a in atoms_to_remove:
        rwmol.RemoveAtom(a)
    mol = rwmol.GetMol()
    for a in mol.GetAtoms():
        if not a.IsInRing() and a.GetIsAromatic():
            a.SetIsAromatic(False)
    rdmolops.SanitizeMol(mol)
    return mol


def crop(img):
    nonwhite_positions = [
        (x, y)
        for x in range(img.size[0])
        for y in range(img.size[1])
        if img.getdata()[x + y * img.size[0]] != (255, 255, 255)  # type: ignore
    ]
    rect = (
        min([x - 10 for x, y in nonwhite_positions]),
        min([y - 10 for x, y in nonwhite_positions]),
        max([x + 10 for x, y in nonwhite_positions]),
        max([y + 10 for x, y in nonwhite_positions]),
    )
    return img.crop(rect)


class MCSDebug:
    def __init__(self):
        self.fontsize = 9
        self.balancer = synrbl.Balancer()
        self.balancer.columns.append("mcs")
        self.cairosize = (1600, 900)
        self.highlight_color = (0.4, 0.9, 0.6, 1)

    def plot(self, smiles, verbose=True):
        result = self.balancer.rebalance(smiles, output_dict=True)[0]

        if verbose:
            for k, v in result.items():
                print("{}: {}".format(k, v))

        mcs = result["mcs"]
        sorted_reactants = mcs["sorted_reactants"]
        mols = [
            rdmolfiles.MolFromSmiles(s)
            for s in sorted_reactants
            + result["input_reaction"].split(">>")[1].split(".")
        ]
        mcs_mols = [rdmolfiles.MolFromSmarts(s) for s in mcs["mcs_results"]]
        mcs_index_list = list(
            r_mol.GetSubstructMatch(mcs_mol)
            for r_mol, mcs_mol in zip(mols, mcs_mols + [mcs_mols[0]])
        )

        fig = plt.figure()
        gs = fig.add_gridspec(3, len(mols))
        ax1 = fig.add_subplot(gs[0, :])
        axs2 = [fig.add_subplot(gs[1, i]) for i in range(len(mols))]
        ax3 = fig.add_subplot(gs[2, :])

        rxnvis = RxnVis(cairosize=self.cairosize)
        img = rxnvis.get_rxn_img(result["input_reaction"])
        ax1.imshow(img)
        ax1.set_title(
            "Input (Id: {})\nSMILES: {}".format(mcs["id"], result["input_reaction"]),
            fontsize=self.fontsize,
        )
        ax1.axis("off")

        highlight_colors = [
            {i: self.highlight_color for i in mcs_i} for mcs_i in mcs_index_list
        ]
        titles = [
            "Reactant {}\nBoundaries: {}\nNeighbors: {}".format(
                i + 1,
                (
                    [list(b.keys())[0] for b in mcs["boundary_atoms_products"][i]]
                    if mcs["boundary_atoms_products"][i] is not None
                    else "None"
                ),
                (
                    [list(n.keys())[0] for n in mcs["nearest_neighbor_products"][i]]
                    if mcs["nearest_neighbor_products"][i] is not None
                    else "None"
                ),
            )
            for i in range(len(sorted_reactants))
        ] + ["Product"]
        for ax, mol, mcs_indices, colors, title in zip(
            axs2, mols, mcs_index_list, highlight_colors, titles
        ):
            img = Draw.MolsToGridImage(
                [mol],
                molsPerRow=1,
                subImgSize=(int(self.cairosize[0] / len(mols)), self.cairosize[1]),
                highlightAtomLists=[mcs_indices],
                highlightAtomColors=[colors],
            )
            img = crop(img)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(title, fontsize=self.fontsize)

        rxnvis = RxnVis(cairosize=self.cairosize)
        img = rxnvis.get_rxn_img(result["reaction"])
        ax3.imshow(img)
        ax3.set_title(
            "Result (Confidence: {:.1%})\nRules: {}\nIssue: {}".format(
                result["confidence"], result["rules"], result["issue"]
            ),
            fontsize=self.fontsize,
        )
        ax3.axis("off")

        plt.tight_layout()
        plt.show()
