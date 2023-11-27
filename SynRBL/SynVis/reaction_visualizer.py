import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import io
from PIL import Image

class ReactionVisualizer:
    """
    A class for visualizing chemical reactions using RDKit and Matplotlib.

    Parameters
    ----------
    compare : bool, optional
        Whether to compare old and new reactions side by side. Default is True.
    orientation : str, optional
        Orientation of comparison ('vertical' or 'horizontal'). Default is 'vertical'.
    figsize : tuple, optional
        Figure size (width, height) in inches. Default is (10, 5).
    label_position : str, optional
        Position of labels ('above' or 'below') the reaction images. Default is 'below'.
    dpi : int, optional
        Dots per inch for image resolution. Default is 300.
    bond_line_width : float, optional
        Width of bond lines in the reaction image. Default is 6.
    atom_label_font_size : int, optional
        Font size for atom labels in the reaction image. Default is 50.
    padding : float, optional
        Padding around the drawing in the reaction image. Default is 0.001.

    Examples
    --------
    Create a ReactionVisualizer instance:

    >>> visualizer = ReactionVisualizer(
    ...     compare=True,
    ...     orientation='horizontal',
    ...     figsize=(12, 6),
    ...     label_position='above',
    ...     dpi=300,
    ...     bond_line_width=6,
    ...     atom_label_font_size=50,
    ...     padding=0.001
    ... )

    Visualize a single reaction:

    >>> reaction_str = 'C1=CC=CC=C1>>CCO'
    >>> visualizer.plot_reactions(reaction_str, '')

    Compare two reactions:

    >>> old_reaction_str = 'C1=CC=CC=C1>>CCO'
    >>> new_reaction_str = 'C1=CC=CC=C1>>CCO.CN'
    >>> visualizer.plot_reactions(old_reaction_str, new_reaction_str)
    """

    def __init__(self, compare=True, orientation='vertical', figsize=(10, 5), label_position='below', dpi=300,
                 bond_line_width=6, atom_label_font_size=50, padding=0.001):
        """
        Initialize the ReactionVisualizer.

        Parameters
        ----------
        compare : bool, optional
            Whether to compare old and new reactions side by side. Default is True.
        orientation : str, optional
            Orientation of comparison ('vertical' or 'horizontal'). Default is 'vertical'.
        figsize : tuple, optional
            Figure size (width, height) in inches. Default is (10, 5).
        label_position : str, optional
            Position of labels ('above' or 'below') the reaction images. Default is 'below'.
        dpi : int, optional
            Dots per inch for image resolution. Default is 300.
        bond_line_width : float, optional
            Width of bond lines in the reaction image. Default is 6.
        atom_label_font_size : int, optional
            Font size for atom labels in the reaction image. Default is 50.
        padding : float, optional
            Padding around the drawing in the reaction image. Default is 0.001.
        """
        self.compare = compare
        self.orientation = orientation
        self.figsize = figsize
        self.label_position = label_position
        self.dpi = dpi
        self.bond_line_width = bond_line_width
        self.atom_label_font_size = atom_label_font_size
        self.padding = padding

    def visualize_reaction(self, reaction_str):
        """
        Visualize a single chemical reaction and return the image.

        Parameters
        ----------
        reaction_str : str
            A string representation of the reaction (e.g., 'C1=CC=CC=C1>>CCO').

        Returns
        -------
        PIL.Image.Image
            An image of the chemical reaction.
        """
        # Parse reactants and products from the reaction string
        reactants_str, products_str = reaction_str.split('>>')
        reactants = [Chem.MolFromSmiles(smiles) for smiles in reactants_str.split('.')]
        products = [Chem.MolFromSmiles(smiles) for smiles in products_str.split('.')]
        
        rxn = AllChem.ChemicalReaction()
        for reactant in reactants:
            rxn.AddReactantTemplate(reactant)
        for product in products:
            rxn.AddProductTemplate(product)

        # Set up RDKit drawer with customizable parameters
        drawer = rdMolDraw2D.MolDraw2DCairo(2000, 600)  # Adjust canvas size as needed
        opts = drawer.drawOptions()
        opts.bondLineWidth = self.bond_line_width  # Increase bond line width
        opts.atomLabelFontSize = self.atom_label_font_size  # Increase font size for atom labels
        opts.padding = self.padding  # Adjust padding around the drawing

        drawer.DrawReaction(rxn)
        drawer.FinishDrawing()
        return Image.open(io.BytesIO(drawer.GetDrawingText()))

    def plot_reactions(self, old_reaction_str, new_reaction_str, savefig=False, pathname = None, dpi=300):
        """
        Plot one or two chemical reactions for visualization.

        Parameters
        ----------
        old_reaction_str : str
            A string representation of the old reaction.
        new_reaction_str : str
            A string representation of the new reaction.
        """
        new_reaction_image = self.visualize_reaction(new_reaction_str)
        if self.compare:
            old_reaction_image = self.visualize_reaction(old_reaction_str)
            nrows, ncols = (2, 1) if self.orientation == 'vertical' else (1, 2)
            fig, axs = plt.subplots(nrows, ncols, figsize=self.figsize, dpi=self.dpi)
            ax_new = axs[1] if self.orientation == 'vertical' else axs[0]
            ax_old = axs[0] if self.orientation == 'vertical' else axs[1]
            ax_old.imshow(old_reaction_image)
            ax_old.axis('off')
            ax_new.imshow(new_reaction_image)
            ax_new.axis('off')
            label_y_position = -0.1 if self.label_position == 'below' else 1.1
            ax_old.set_title('Old Reaction', position=(0.5, label_y_position), weight='bold')
            ax_new.set_title('New Reaction', position=(0.5, label_y_position), weight='bold')
        else:
            fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
            ax.imshow(new_reaction_image)
            ax.axis('off')
            label_y_position = -0.1 if self.label_position == 'below' else 1.1
            ax.set_title('New Reaction', position=(0.5, label_y_position), weight='bold')
        if savefig == True:
            fig.savefig(pathname, dpi=self.dpi)

        plt.tight_layout()