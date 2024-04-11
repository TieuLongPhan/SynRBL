from rdkit import Chem
from synrbl.SynProcessor import RSMIDecomposer
import pandas as pd
from typing import List, Optional, Dict


class RuleImputeManager(RSMIDecomposer):
    """
    A class for managing a database of chemical compounds, allowing for the
    addition, removal, and standardization of compound entries.

    Parameters
    ----------
    database : list
        A list of dictionaries representing chemical compounds. Each dictionary
        has keys: 'formula' (str), 'smiles' (str), and 'Composition' (dict).

    Methods
    -------
    add_entry(formula, smiles):
        Adds a new entry to the database.
    add_entries(entries):
        Adds multiple entries to the database.
    remove_entry(formula):
        Removes an entry from the database based on its formula.
    canonicalize_smiles(smiles):
        Returns the canonical form of the SMILES string.
    is_valid_smiles(smiles):
        Checks if the SMILES string is valid.
    """

    def __init__(self, database: Optional[List[dict]] = None) -> None:
        """
        Initializes the database with existing data if provided.

        Args:
        database (List[dict], optional): An existing database.

        Example:
        >>> existing_db = [{'formula': 'H2O', 'smiles': 'O'}]
        >>> db = RSMIDataImpute(existing_db)
        """
        super().__init__()
        if isinstance(database, pd.DataFrame):
            self.database = database.to_dict("records")
        else:
            self.database = database or []

    def add_entry(self, formula: str, smiles: str) -> None:
        """
        Adds a new entry to the database, ensuring no duplicates and valid
        SMILES.

        Args:
            formula: The formula of the compound.
            smiles: The SMILES string of the compound.

        Raises:
            ValueError: If the SMILES string is invalid or if the entry already
                exists.

        Example:
            >>> db = RSMIDataImpute()
            >>> db.add_entry('H2O', 'O')
            Entry with formula 'H2O' and smiles 'O' added to the database.
        """
        if any(d["formula"] == formula for d in self.database):
            raise ValueError(f"Entry with formula '{formula}' already exists.")

        if any(d["smiles"] == smiles for d in self.database):
            raise ValueError(f"Entry with SMILES '{smiles}' already exists.")

        if not self.is_valid_smiles(smiles):
            raise ValueError(f"Invalid SMILES string: {smiles}")

        composition = self.decompose(smiles)
        if "Q" not in composition:
            composition["Q"] = 0

        self.database.append(
            {"formula": formula, "smiles": smiles, "Composition": composition}
        )

        print(
            f"Entry with formula '{formula}' and smiles '{smiles}' added "
            + "to the database."
        )

    def add_entries(self, entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Adds multiple entries to the database.

        Args:
            entries: A list of dictionaries with 'formula' and 'smiles' keys.

        Returns:
            A list of entries that were not added due to errors.

        Example:
        >>> db = RSMIDataImpute()
        >>> invalid_entries = db.add_entries(
            [
                {"formula": "CO2", "smiles": "C=O"},
                {"formula": "Invalid", "smiles": "Invalid"},
            ]
        )
        Invalid entries: [{'formula': 'Invalid', 'smiles': 'Invalid'}]
        """
        invalid_entries = []
        for entry in entries:
            try:
                self.add_entry(entry["formula"], entry["smiles"])
            except ValueError:
                invalid_entries.append(entry)

        return invalid_entries

    def remove_entry(self, formula: str) -> None:
        """
        Removes an entry from the database based on its formula.

        Args:
            formula (str): The formula of the compound to remove.

        Example:
            >>> db = RSMIDataImpute()
            >>> db.add_entry('H2O', 'O')
            >>> db.remove_entry('H2O')
            Entry with formula 'H2O' removed from the database.
        """
        entry = next((d for d in self.database if d["formula"] == formula), None)
        if entry:
            self.database.remove(entry)
            print(f"Entry with formula '{formula}' removed from the database.")
        else:
            print(f"No entry found with formula '{formula}'.")

    @staticmethod
    def canonicalize_smiles(smiles: str) -> str:
        """
        Converts a SMILES string to its canonical form.

        Args:
        - smiles (str): The SMILES string to canonicalize.

        Returns:
        - str: The canonicalized SMILES string.

        Example:
        >>> canonical_smiles = RSMIDataImpute.canonicalize_smiles('O')
        >>> print(canonical_smiles)
        O
        """
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=True) if mol else None

    @staticmethod
    def is_valid_smiles(smiles: str) -> bool:
        """
        Checks the validity of a SMILES string.

        Args:
        - smiles (str): The SMILES string to check.

        Returns:
        - bool: True if valid, False otherwise.

        Example:
        >>> is_valid = RSMIDataImpute.is_valid_smiles('O')
        >>> print(is_valid)
        True
        """
        return Chem.MolFromSmiles(smiles) is not None
