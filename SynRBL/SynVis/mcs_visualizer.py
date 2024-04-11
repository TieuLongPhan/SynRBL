import io

from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image


class MCSVisualizer:
    """
    A class for visualizing molecules and highlighting their Maximum Common
    Substructure (MCS).

    Methods
    -------
    draw_molecule_with_atom_numbers(mol):
        Annotates a molecule with atom indices.

    def highlight_molecule(
        molecule_smiles,
        mcs_smiles,
        show_atom_numbers=False,
        compare=False,
        missing_graph_smiles=None,
    ):
        Highlights the specified substructure within a molecule and optionally
        compares it with another molecule.
    """

    def __init__(self):
        pass

    def draw_molecule_with_atom_numbers(self, mol):
        """
        Annotates a molecule with atom indices.

        Parameters
        ----------
        mol : RDKit Molecule object
            The molecule to be annotated with atom numbers.

        Returns
        -------
        RDKit Mol object
            The annotated molecule.
        """
        mol_with_atom_numbers = Chem.Mol(mol)
        for atom in mol_with_atom_numbers.GetAtoms():
            atom.SetProp("atomLabel", str(atom.GetIdx()))
        return mol_with_atom_numbers

    def highlight_molecule(
        self,
        molecule_smiles,
        mcs_smiles,
        show_atom_numbers=False,
        compare=False,
        missing_graph_smiles=None,
    ):
        """
        Highlights the specified substructure (MCS) within a molecule and
        optionally compares it with another molecule.

        Parameters
        ----------
        molecule_smiles : str
            SMILES representation of the molecule.
        mcs_smiles : str
            SMILES representation of the substructure (MCS).
        show_atom_numbers : bool, optional
            If True, shows atom numbers on the molecules.
        compare : bool, optional
            If True, shows a comparison with another molecule specified by
            missing_graph_smiles.
        missing_graph_smiles : str, optional
            SMILES representation of the second molecule for comparison.

        Returns
        -------
        PIL.Image.Image
            An image of the molecule with the specified substructure
            highlighted, and optionally the second molecule.

        Example
        -------
        visualizer = MCSVisualizer()
        img = visualizer.highlight_molecule(
            "CCO", "CO", compare=True, missing_graph_smiles="CCC"
        )
        img.show()
        """
        molecule = Chem.MolFromSmiles(molecule_smiles)
        pattern = Chem.MolFromSmarts(Chem.CanonSmiles(mcs_smiles))
        matching = molecule.GetSubstructMatch(pattern)

        if not matching:
            raise ValueError("Substructure not found in molecule.")

        if show_atom_numbers:
            molecule = self.draw_molecule_with_atom_numbers(molecule)

        mols = [molecule]
        highlight_lists = [matching] if matching else [None]
        legends = ["Molecule with MCS Highlighted"]

        # Check if a comparison molecule is needed
        if compare and missing_graph_smiles:
            missing_graph = Chem.MolFromSmiles(missing_graph_smiles)
            if show_atom_numbers:
                missing_graph = self.draw_molecule_with_atom_numbers(missing_graph)
            mols.append(missing_graph)
            highlight_lists.append(None)
            legends.append("Missing Molecule")

        # Image dimensions
        width, height = 600, 600
        molsPerRow = 2 if compare else 1
        img_width = width * molsPerRow
        img_height = height * len(mols) // molsPerRow

        # Create a drawing object
        drawer = rdMolDraw2D.MolDraw2DCairo(img_width, img_height, width, height)

        # Draw each molecule
        for i, mol in enumerate(mols):
            row = i // molsPerRow
            col = i % molsPerRow
            x_offset = width * col
            y_offset = height * row
            drawer.SetOffset(x_offset, y_offset)
            drawer.DrawMolecule(
                mol, legend=legends[i], highlightAtoms=highlight_lists[i]
            )

        drawer.FinishDrawing()

        return Image.open(io.BytesIO(drawer.GetDrawingText()))
