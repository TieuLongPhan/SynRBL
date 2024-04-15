from rdkit import Chem
from typing import List, Union
from fgutils import FGQuery


class MoleculeStandardizer:
    """
    A class to standardize molecules by converting specific functional groups to their more
    common forms using RDKit for molecule manipulation.

    Parameters:
        smiles (str): SMILES representation of the molecule.
        query (FGQuery): An instance of FGQuery to handle functional group identification and queries.

    Methods:
        fit() -> str: Standardizes the molecule by iteratively converting identified functional groups.
        standardize_enol(smiles: str, atom_indices: List[int]) -> str: Converts ketones or aldehydes to enols.
        standardize_hemiketal(smiles: str, atom_indices: List[int]) -> str: Converts ketones to hemiketals.
    """

    def __init__(self, smiles: str):
        """
        Initializes the MoleculeStandardizer with a SMILES string and a functional group query object.

        Args:
            smiles (str): SMILES string of the molecule to be standardized.
        """
        self.smiles = smiles
        self.query = FGQuery(use_smiles=True)

    def fit(self) -> str:
        """
        Performs the standardization process by converting all relevant functional groups to their
        target forms based on predefined rules and updates the SMILES string accordingly.

        Returns:
            str: Canonical SMILES string of the standardized molecule.
        """
        self.fg = self.query.get(self.smiles)
        for dict in self.fg:
            if "hemiketal" in dict:
                atom_indices = dict[1]
                self.smiles = self.standardize_hemiketal(self.smiles, atom_indices)
                self.fg = self.query.get(self.smiles)
            elif "enol" in dict:
                atom_indices = dict[1]
                self.smiles = self.standardize_enol(self.smiles, atom_indices)
                self.fg = self.query.get(self.smiles)
        return Chem.CanonSmiles(self.smiles)

    @staticmethod
    def standardize_enol(smiles: str, atom_indices: List[int] = [0, 1, 2]) -> str:
        """
        Converts a given ketone (or aldehyde) to its enol form based on specified atom indices.

        Args:
        smiles (str): The SMILES string of the molecule.
        atom_indices (List[int]): List containing indices of two carbons and one oxygen atom
                                involved in the enol formation. Order does not initially matter.

        Returns:
        str: The SMILES string of the molecule after conversion to the enol form. If the indices are invalid,
            a string message indicating an error is returned.

        Raises:
        Exception: If an error occurs in bond manipulation or molecule sanitization, it prints an error message.

        Example:
        >>> enol("C=C(-O)C", [0, 1, 2])
        'CC(=O)-C'
        """
        # Initialize molecule and editable molecule
        mol = Chem.MolFromSmiles(smiles)
        emol = Chem.EditableMol(mol)

        c1_idx, c2_idx, o_idx = None, None, None
        # Identify the oxygen index and remove it from atom_indices to simplify carbon handling
        for i in atom_indices[:]:
            if mol.GetAtomWithIdx(i).GetSymbol() == "O":
                o_idx = i
                atom_indices.remove(i)

        # Distinguish between the two carbons based on their proximity to the oxygen
        for i in atom_indices:
            if abs(i - o_idx) == 1:
                c2_idx = i
            else:
                c1_idx = i

        # Check if indices are correctly assigned
        if None in [c1_idx, c2_idx, o_idx]:
            return "Invalid atom indices provided. Please check the input."

        # Try to modify the bonds to create the enol form
        try:
            emol.RemoveBond(c1_idx, c2_idx)
            emol.RemoveBond(c2_idx, o_idx)
            emol.AddBond(c1_idx, c2_idx, order=Chem.rdchem.BondType.SINGLE)
            emol.AddBond(c2_idx, o_idx, order=Chem.rdchem.BondType.DOUBLE)
        except Exception as e:
            return f"Error in modifying molecule: {str(e)}"

        # Generate the new molecule and sanitize it
        new_mol = emol.GetMol()
        try:
            Chem.SanitizeMol(new_mol)
        except Exception as e:
            return f"Error in sanitizing molecule: {str(e)}"

        # Return the new SMILES representation
        return Chem.MolToSmiles(new_mol)

    @staticmethod
    def standardize_hemiketal(smiles: str, atom_indices: List[int]) -> str:
        """
        Modifies a ketone molecule (represented by a SMILES string) to its hemiketal form
        based on specified atom indices for carbon and two oxygens.

        Args:
        smiles (str): SMILES representation of the original molecule.
        atom_indices (List[int]): Indices of the carbon and two oxygen atoms involved in the transformation.
                                The order of indices does not need to be [C, O1, O2].

        Returns:
        str: SMILES string of the modified molecule if successful. If the indices are invalid or an error occurs,
            a string message indicating the error is returned.

        Example:
        >>> standardize_hemiketal("C(O)(O)C", [1, 2, 0])
        'C(=O)C'

        Raises:
        Exception: If bond modifications or molecule sanitization fail, it captures and returns the exception message.
        """
        # Load the molecule from SMILES and create an editable molecule object
        mol = Chem.MolFromSmiles(smiles)
        emol = Chem.EditableMol(mol)

        # Initialize indices
        c_idx, o1_idx, o2_idx = None, None, None
        # Determine the indices for carbon and oxygens
        for i in atom_indices:
            atom_symbol = mol.GetAtomWithIdx(i).GetSymbol()
            if atom_symbol == "C":
                c_idx = i
            elif atom_symbol == "O":
                if o1_idx is None:
                    o1_idx = i  # Assume the first oxygen encountered is O1
                else:
                    o2_idx = i  # The next oxygen is O2

        # Check if all indices are assigned
        if None in [c_idx, o1_idx, o2_idx]:
            return "Invalid atom indices provided. Please check the input."

        # Attempt to modify the molecule structure
        try:
            # Remove existing bonds if they exist
            emol.RemoveBond(c_idx, o1_idx)
            emol.RemoveBond(c_idx, o2_idx)
            # Add new bonds to form hemiketal
            emol.AddBond(c_idx, o1_idx, order=Chem.rdchem.BondType.DOUBLE)
        except Exception as e:
            return f"Error in modifying molecule: {str(e)}"

        # Generate the new molecule and sanitize it
        new_mol = emol.GetMol()
        try:
            Chem.SanitizeMol(new_mol)
        except Exception as e:
            return f"Error in sanitizing molecule: {str(e)}"

        return Chem.MolToSmiles(new_mol)
