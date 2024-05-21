import io
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.Draw.rdMolDraw2D as rdMolDraw2D
import rdkit.Chem.rdChemReactions as rdChemReactions
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf

from PIL import Image
from synrbl.SynUtils.chem_utils import remove_atom_mapping, normalize_smiles


class RxnVis:
    def __init__(
        self,
        nrows=1,
        ncols=1,
        dpi=400,
        figsize=(16, 9),
        cairosize=(1600, 900),
        border=10,
        show=True,
        remove_aam=False,
        normalize=False,
        close_fig=True,
    ):
        self.nrows = nrows
        self.ncols = ncols
        self.dpi = dpi
        self.figsize = figsize
        self.cairosize = cairosize
        self.border = border
        self.show = show
        self.remove_aam = remove_aam
        self.normalize = normalize
        self.close_fig = close_fig

    def get_rxn_img(self, smiles):
        drawer = rdMolDraw2D.MolDraw2DCairo(*self.cairosize)
        if ">>" in smiles:
            rxn = rdChemReactions.ReactionFromSmarts(smiles, useSmiles=True)
            drawer.DrawReaction(rxn)
        else:
            mol = rdmolfiles.MolFromSmiles(smiles)
            if mol is None:
                mol = rdmolfiles.MolFromSmarts(smiles)
            drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        img = Image.open(io.BytesIO(drawer.GetDrawingText()))
        nonwhite_positions = [
            (x, y)
            for x in range(img.size[0])
            for y in range(img.size[1])
            if img.getdata()[x + y * img.size[0]] != (255, 255, 255)  # type: ignore
        ]
        rect = (
            min([x - self.border for x, y in nonwhite_positions]),
            min([y - self.border for x, y in nonwhite_positions]),
            max([x + self.border for x, y in nonwhite_positions]),
            max([y + self.border for x, y in nonwhite_positions]),
        )
        return img.crop(rect)

    def __parse_input(self, smiles, titles):
        if isinstance(smiles, str):
            smiles = [smiles]
        if titles is None:
            titles = [None for _ in range(len(smiles))]
        return smiles, titles

    def __get_fig(self):
        fig, axs = plt.subplots(
            self.nrows, self.ncols, dpi=self.dpi, figsize=self.figsize
        )
        if self.ncols * self.nrows == 1:
            axs = [[axs]]
        elif self.nrows == 1:
            axs = [axs]
        elif self.ncols == 1:
            axs = [[a] for a in axs]
        return fig, axs

    def __get_ax(self, axs, i, title=None):
        i_r = int(i / self.ncols)
        i_c = int(i % self.ncols)
        ax = axs[i_r][i_c]
        if title is not None:
            ax.set_title(title)
        ax.axis("off")
        return ax

    def plot(
        self,
        smiles: str | list[str],
        titles=None,
        savefig=None,
        show=None,
        remove_aam=None,
        normalize=None,
        close_fig=None,
    ):
        smiles, titles = self.__parse_input(smiles, titles)
        show = show if show is not None else self.show
        remove_aam = remove_aam if remove_aam is not None else self.remove_aam
        normalize = normalize if normalize is not None else self.normalize
        close_fig = close_fig if close_fig is not None else self.close_fig

        if normalize:
            smiles = [normalize_smiles(s) for s in smiles]
        elif remove_aam:
            smiles = [remove_atom_mapping(s) for s in smiles]

        fig, axs = self.__get_fig()
        for i, (s, t) in enumerate(zip(smiles, titles)):
            if i == self.nrows * self.ncols:
                print(
                    "[WARN] {} reactions will not be displayed.".format(len(smiles) - i)
                )
                break
            ax = self.__get_ax(axs, i, title=t)
            if s is not None and len(s) > 0:
                img = self.get_rxn_img(s)
                ax.imshow(img)
        fig.tight_layout()
        if savefig is not None:
            fig.savefig(savefig)
        if show is True:
            plt.show()
        elif close_fig:
            plt.close(fig)
            fig = None
        return fig, axs


class Rxn2Pdf:
    def __init__(self, file, **kwargs):
        kwargs = Rxn2Pdf.__override_kwargs(**kwargs)
        self.rxnvis = RxnVis(**kwargs)
        self.pdf = matplotlib.backends.backend_pdf.PdfPages(file)

    @staticmethod
    def __override_kwargs(**kwargs):
        kwargs["close_fig"] = False
        kwargs["show"] = False
        return kwargs

    def add(self, smiles, **kwargs):
        if self.pdf is None:
            raise RuntimeError("Pdf is already closed.")
        kwargs = Rxn2Pdf.__override_kwargs(**kwargs)
        fig, _ = self.rxnvis.plot(smiles, **kwargs)
        self.pdf.savefig(fig)
        plt.close(fig)

    def close(self):
        if self.pdf is None:
            raise RuntimeError("Pdf is already closed.")
        self.pdf.close()
        self.pdf = None
