from typing import List, Dict, Set, Any
from typing import Optional, Union, Callable, Tuple
from rdkit import Chem
import io
import re
import tempfile
import matplotlib.pyplot as plt
from PIL import Image
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, letter
from SynRBL.SynVis.reaction_visualizer import ReactionVisualizer


class CheckCarbonBalance:
    def __init__(
        self, reactions_data: List[Dict[str, str]], rsmi_col="reactions", symbol=">>"
    ):
        """
        Initialize the CheckCarbonBalance class with reaction data.

        Parameters:
        reactions_data (List[Dict[str, str]]): A list of dictionaries, each containing reaction information.
        """
        self.reactions_data = reactions_data
        self.rsmi_col = rsmi_col
        self.symbol = symbol

    @staticmethod
    def count_carbon_atoms(smiles: str) -> int:
        """
        Count the number of carbon atoms in a molecule represented by a SMILES string.

        Parameters:
        smiles (str): A SMILES string.

        Returns:
        int: The number of carbon atoms in the molecule. Returns 0 if the SMILES string is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        return (
            sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C") if mol else 0
        )

    def check_carbon_balance(self) -> None:
        """
        Check and update the carbon balance status for each reaction in the reactions data.

        The method updates each reaction dictionary in the reactions data with a new key 'carbon_balance_check'.
        This key will have the value 'products' if the number of carbon atoms in the products is greater than or equal to the reactants,
        and 'reactants' otherwise.
        """
        for reaction in self.reactions_data:
            try:
                reactants_smiles, products_smiles = reaction[self.rsmi_col].split(
                    self.symbol
                )
                reactants_carbon = sum(
                    self.count_carbon_atoms(smiles)
                    for smiles in reactants_smiles.split(".")
                )
                products_carbon = sum(
                    self.count_carbon_atoms(smiles)
                    for smiles in products_smiles.split(".")
                )

                if reactants_carbon >= products_carbon:
                    reaction["carbon_balance_check"] = "products"
                else:
                    reaction["carbon_balance_check"] = "reactants"
            except KeyError as e:
                print(f"Key error: {e}")
            except ValueError as e:
                print(f"Value error: {e}")

    def is_carbon_balance(self) -> None:
        """
        Check and update the carbon balance status for each reaction in the reactions data.

        The method updates each reaction dictionary in the reactions data with a new key 'carbon_balance_check'.
        This key will have the value 'products' if the number of carbon atoms in the products is greater than or equal to the reactants,
        and 'reactants' otherwise.
        """
        for reaction in self.reactions_data:
            try:
                reactants_smiles, products_smiles = reaction[self.rsmi_col].split(
                    self.symbol
                )
                reactants_carbon = sum(
                    self.count_carbon_atoms(smiles)
                    for smiles in reactants_smiles.split(".")
                )
                products_carbon = sum(
                    self.count_carbon_atoms(smiles)
                    for smiles in products_smiles.split(".")
                )
                reaction["is_carbon_balance"] = reactants_carbon == products_carbon

            except KeyError as e:
                print(f"Key error: {e}")
            except ValueError as e:
                print(f"Value error: {e}")


def calculate_net_charge(sublist: list[dict[str, Union[str, int]]]) -> int:
    """
    Calculate the net charge from a list of molecules represented as SMILES strings.

    Args:
        sublist: A list of dictionaries, each with a 'smiles' string and a 'Ratio' integer.

    Returns:
        The net charge of the sublist as an integer.
    """
    total_charge = 0
    for item in sublist:
        if "smiles" in item and "Ratio" in item:
            mol = Chem.MolFromSmiles(item["smiles"])
            if mol:
                charge = (
                    sum(abs(atom.GetFormalCharge()) for atom in mol.GetAtoms())
                    * item["Ratio"]
                )
                total_charge += charge
    return total_charge


def save_reactions_to_pdf(
    data: List[Dict[str, str]],
    old_reaction_col: str,
    new_reaction_col: str,
    pdf_filename: str,
    compare: bool = False,
    orientation: str = "vertical",
    show_atom_numbers: bool = False,
):
    """
    Save a list of reaction visualizations to a PDF file.

    Parameters
    ----------
    data : List[Dict[str, str]]
        A list of dictionaries containing reaction data.
    old_reaction_col : str
        The column name containing the string representation of the old reaction.
    new_reaction_col : str
        The column name containing the string representation of the new reaction.
    pdf_filename : str
        The filename of the PDF to be saved.
    compare : bool, optional
        If True, both the old and new reactions are plotted. Default is False.
    orientation : str, optional
        The layout orientation of the plots ('vertical' or 'horizontal'). Default is 'vertical'.
    show_atom_numbers : bool, optional
        Whether to show atom numbers in the reaction visualizations. Default is False.
    scale_factor : float, optional
        Factor to scale the reaction image size in the PDF. Default is 1.0.
    title_font_size : int, optional
        Font size for the title. Default is 14.

    Notes
    -----
    The method plots each reaction using the plot_reactions method and saves it to a PDF file.
    Each reaction is plotted on a separate page. The method also handles scaling of the reaction
    image and includes a customizable title for each reaction page in the PDF.
    """
    c = canvas.Canvas(pdf_filename, pagesize=landscape(letter))
    page_width, page_height = landscape(letter)

    for reaction_data in data:
        # Create a figure using plot_reactions method
        fig = ReactionVisualizer(figsize=(10, 5)).plot_reactions(
            reaction_data,
            old_reaction_col,
            new_reaction_col,
            compare,
            orientation,
            show_atom_numbers,
        )

        # Save figure to a temporary buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf)

        # Save the image to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png") as img_temp:
            img.save(img_temp, format="PNG")
            img_temp.flush()

            # Image dimensions
            img_width, img_height = img.size
            scale_factor = min(
                (page_width - 100) / img_width, (page_height - 100) / img_height
            )
            scaled_width, scaled_height = (
                img_width * scale_factor,
                img_height * scale_factor,
            )

            # Calculate coordinates to center the image
            x = (page_width - scaled_width) / 2
            y = (page_height - scaled_height) / 2

            # Draw the centered image
            c.drawImage(
                img_temp.name,
                x,
                y,
                width=scaled_width,
                height=scaled_height,
                mask="auto",
            )
            c.showPage()

    c.save()
    print(f"Saved reactions to {pdf_filename}")


def remove_atom_mapping(smiles: str) -> str:
    pattern = re.compile(r":\d+")
    smiles = pattern.sub("", smiles)
    pattern = re.compile(r"\[(?P<atom>(B|C|N|O|P|S|F|Cl|Br|I){1,2})(?:H\d?)?\]")
    smiles = pattern.sub(r"\g<atom>", smiles)
    return smiles


def normalize_smiles(smiles: str) -> str:
    if ">>" in smiles:
        return ">>".join([normalize_smiles(t) for t in smiles.split(">>")])
    elif "." in smiles:
        token = sorted(
            smiles.split("."),
            key=lambda x: (sum(1 for c in x if c.isupper()), sum(ord(c) for c in x)),
            reverse=True,
        )
        token = [normalize_smiles(t) for t in token]
        token.sort(
            key=lambda x: (sum(1 for c in x if c.isupper()), sum(ord(c) for c in x)),
            reverse=True,
        )
        return ".".join(token)
    else:
        return Chem.CanonSmiles(remove_atom_mapping(smiles))
