import logging
from rdkit.rdBase import BlockLogs
from typing import List, Dict
from rdkit import Chem
from joblib import Parallel, delayed


class InvalidSmilesException(Exception):
    """Custom exception for invalid SMILES strings."""

    pass


class CheckCarbonBalance:
    """
    A class to check the atom balance in chemical reactions using
    parallel processing.

    Class Variables
    ---------------
    rsmi_col : str
        Column name for the reaction SMILES string.
    symbol : str
        Symbol separating reactants and products in the SMILES string.
    atom_type : str
        Type of atom to check the balance for.
    smiles_cache : dict
        Cache for storing counts of previously processed SMILES strings.

    Methods
    -------
    count_atoms(smiles: str, atom_type: str) -> int:
        Count the number of specified atoms in a molecule represented by a
        SMILES string.
    process_reaction(reaction: Dict[str, str]) -> Dict[str, str]:
        Process a single reaction to check atom balance.
    check_atom_balance() -> List[Dict[str, str]]:
        Check and return the atom balance status for each reaction using
        parallel processing.
    """

    def __init__(
        self,
        reactions_data: List[Dict[str, str]],
        rsmi_col="reactions",
        symbol=">>",
        atom_type="C",
        n_jobs=4,
    ):
        self.reactions_data = reactions_data
        self.rsmi_col = rsmi_col
        self.symbol = symbol
        self.atom_type = atom_type
        self.n_jobs = n_jobs
        self.smiles_cache = {}

    @staticmethod
    def count_atoms(smiles: str, atom_type: str, smiles_cache: Dict[str, int]) -> int:
        if smiles in smiles_cache:
            return smiles_cache[smiles]

        try:
            mol = Chem.MolFromSmiles(smiles)
            count = (
                sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == atom_type)
                if mol
                else 0
            )
            smiles_cache[smiles] = count
            return count
        except Exception as e:
            raise InvalidSmilesException(f"Invalid SMILES string {smiles}: {e}")

    @staticmethod
    def process_reaction(
        reaction: Dict[str, str],
        rsmi_col: str,
        symbol: str,
        atom_type: str,
        smiles_cache: Dict[str, int],
    ) -> Dict[str, str]:
        block = BlockLogs()
        new_reaction = reaction.copy()
        try:
            reactants_smiles, products_smiles = new_reaction[rsmi_col].split(symbol)
            reactants_carbon = sum(
                CheckCarbonBalance.count_atoms(smiles, atom_type, smiles_cache)
                for smiles in reactants_smiles.split(".")
            )
            products_carbon = sum(
                CheckCarbonBalance.count_atoms(smiles, atom_type, smiles_cache)
                for smiles in products_smiles.split(".")
            )

            if reactants_carbon == products_carbon:
                new_reaction["carbon_balance_check"] = "balanced"
            elif reactants_carbon > products_carbon:
                new_reaction["carbon_balance_check"] = "products"
            else:
                new_reaction["carbon_balance_check"] = "reactants"
        except InvalidSmilesException as e:
            logging.error(e)
            new_reaction["carbon_balance_check"] = "error"
        except KeyError as e:
            logging.error(f"Key error in reaction data: {e}")
            new_reaction["carbon_balance_check"] = "error"
        except ValueError as e:
            logging.error(f"Value error in parsing SMILES: {e}")
            new_reaction["carbon_balance_check"] = "error"

        del block
        return new_reaction

    def check_carbon_balance(self) -> List[Dict[str, str]]:
        if not all(isinstance(reaction, dict) for reaction in self.reactions_data):
            raise ValueError("Each item in reactions_data should be a dictionary.")

        parallel_results = Parallel(n_jobs=self.n_jobs, verbose=0)(
            delayed(self.process_reaction)(
                reaction, self.rsmi_col, self.symbol, self.atom_type, self.smiles_cache
            )
            for reaction in self.reactions_data
        )
        return parallel_results
