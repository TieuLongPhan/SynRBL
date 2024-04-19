import io
import tempfile
import matplotlib.pyplot as plt

from typing import List, Dict
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import landscape, letter
from synrbl.SynVis.reaction_visualizer import ReactionVisualizer


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
        The layout orientation of the plots ('vertical' or 'horizontal').
        Default is 'vertical'.
    show_atom_numbers : bool, optional
        Whether to show atom numbers in the reaction visualizations.
        Default is False.
    scale_factor : float, optional
        Factor to scale the reaction image size in the PDF. Default is 1.0.
    title_font_size : int, optional
        Font size for the title. Default is 14.

    Notes
    -----
    The method plots each reaction using the plot_reactions method and saves
    it to a PDF file. Each reaction is plotted on a separate page. The method
    also handles scaling of the reaction image and includes a customizable
    title for each reaction page in the PDF.
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
