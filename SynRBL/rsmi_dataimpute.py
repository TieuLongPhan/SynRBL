from rdkit import Chem
import pandas as pd

class RSMIDataImpute:
    """
    A class used to represent a database of chemical compounds.

    Attributes
    ----------
    database : list
        A list of dictionaries representing chemical compounds. Each dictionary has two keys: 'formula' (str) and 'smiles' (str).

    Methods
    -------
    add_entry(formula: str, smiles: str):
        Adds a new entry to the database. Raises a ValueError if the SMILES string is invalid.
    add_entries(entries: list):
        Adds multiple entries to the database. Returns a list of invalid entries.
    canonicalize_smiles(smiles: str) -> str:
        Returns the canonical form of the SMILES string.
    is_valid_smiles(smiles: str) -> bool:
        Checks if the SMILES string is valid.
    """

    def __init__(self, database=None):
        """
        Initializes the database.

        Parameters:
        database (list or pd.DataFrame, optional): An existing database. If a DataFrame is provided, it will be converted to a list of dictionaries. Default is None, which creates an empty database.

        Example:
        existing_database = [{'formula': 'H2O', 'smiles': 'O'}]
        db = RSMIDataImpute(existing_database)
        """
        if database is None:
            self.database = []
        elif isinstance(database, pd.DataFrame):
            self.database = database.to_dict('records')
        else:
            self.database = database

    def add_entry(self, formula, smiles):
        """
        Adds a new entry to the database.

        Parameters:
        formula (str): The formula of the compound.
        smiles (str): The SMILES string of the compound.

        Raises:
        ValueError: If the SMILES string is invalid.

        Example:
        try:
            db.add_entry('H2O', 'O')
        except ValueError as e:
            print(e)
        """
        # Check if SMILES string is valid
        if not self.is_valid_smiles(smiles):
            raise ValueError(f"'{smiles}' is not a valid SMILES string.")

        # Canonicalize the smiles string
        smiles = self.canonicalize_smiles(smiles)

        # Check if 'smiles' value already exists in the database
        if any(d['smiles'] == smiles for d in self.database):
            print(f"Entry with smiles '{smiles}' already exists in the database.")
            return

        # If not, add the new entry
        self.database.append({'formula': formula, 'smiles': smiles})
        print(f"Entry with formula '{formula}' and smiles '{smiles}' added to the database.")

    def add_entries(self, entries):
        """
        Adds multiple entries to the database.

        Parameters:
        entries (list): A list of dictionaries representing the entries to be added. Each dictionary should have two keys: 'formula' (str) and 'smiles' (str).

        Returns:
        list: A list of invalid entries.

        Example:
        entries = [{'formula': 'CO2', 'smiles': 'C=O'}, {'formula': 'Invalid', 'smiles': 'Invalid'}]
        invalid_entries = db.add_entries(entries)
        print(f"Invalid entries: {invalid_entries}")
        """
        invalid_entries = []

        for entry in entries:
            try:
                self.add_entry(entry['formula'], entry['smiles'])
            except ValueError:
                invalid_entries.append(entry)

        return invalid_entries

    @staticmethod
    def canonicalize_smiles(smiles):
        """
        Returns the canonical form of the SMILES string.

        Parameters:
        smiles (str): The SMILES string to be canonicalized.

        Returns:
        str: The canonical form of the SMILES string.

        Example:
        canonical_smiles = db.canonicalize_smiles('O')
        """
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToSmiles(mol, isomericSmiles=True)

    @staticmethod
    def is_valid_smiles(smiles):
        """
        Checks if the SMILES string is valid.

        Parameters:
        smiles (str): The SMILES string to be checked.

        Returns:
        bool: True if the SMILES string is valid, False otherwise.

        Example:
        is_valid = db.is_valid_smiles('O')
        """
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
