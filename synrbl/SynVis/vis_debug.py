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
        self.balancer.columns.extend(["mcs", "mcs_based_result"])
        self.balancer.mcs_method.output_col.append("mcs_based_result")
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
        ax_mcs = None
        if "mcs_based_result" in result.keys():
            gs = fig.add_gridspec(4, len(mols))
            ax1 = fig.add_subplot(gs[0, :])
            axs2 = [fig.add_subplot(gs[1, i]) for i in range(len(mols))]
            ax_mcs = fig.add_subplot(gs[2, :])
            ax_final = fig.add_subplot(gs[3, :])
        else:
            gs = fig.add_gridspec(3, len(mols))
            ax1 = fig.add_subplot(gs[0, :])
            axs2 = [fig.add_subplot(gs[1, i]) for i in range(len(mols))]
            ax_final = fig.add_subplot(gs[2, :])

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
        if ax_mcs is not None:
            img = rxnvis.get_rxn_img(result["mcs_based_result"])
            ax_mcs.imshow(img)
            ax_mcs.set_title(
                "MCS-Based Result\nRules: {}".format(result.get("rules", None)),
                fontsize=self.fontsize,
            )
            ax_mcs.axis("off")

        rxnvis = RxnVis(cairosize=self.cairosize)
        img = rxnvis.get_rxn_img(result["reaction"])
        ax_final.imshow(img)
        ax_final.set_title(
            "Result (Confidence: {:.1%})\nIssue: {}".format(
                result.get("confidence", 0), result["issue"]
            ),
            fontsize=self.fontsize,
        )
        ax_final.axis("off")

        plt.tight_layout()
        plt.show()
