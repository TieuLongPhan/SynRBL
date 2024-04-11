from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from joblib import Parallel, delayed
import pandas as pd
from typing import List
from synrbl.SynRuleImputer.rule_data_manager import RuleImputeManager


class AutomaticRulesExtraction:
    """
    A class for extracting automatic rules from chemical reactions.

    Attributes
    ----------
    rule_list : list
        A list of automatic rules extracted from reactions.
    existing_database : list
        A list of existing database entries.
    n_jobs : int, optional
        The number of jobs to run in parallel (default is -5).
    verbose : int, optional
        The verbosity level (default is 1).
    """

    def __init__(
        self, existing_database: list = None, n_jobs: int = 4, verbose: int = 1
    ) -> None:
        """
        Constructor for the class.

        Args:
            existing_database (list): A list representing an existing database.
                Default is an empty list.
            n_jobs (int): The number of jobs. Default is -5.
            verbose (int): The verbosity level. Default is 1.
        """
        self.rule_list = []
        self.existing_database = existing_database or []
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def smiles_to_molecular_formula(smiles: str) -> str:
        """
        Convert a SMILES string to a molecular formula.

        Args:
            smiles (str): The SMILES string of the molecule.

        Returns:
            str: The molecular formula of the molecule.
        """
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            return formula

    def add_new_entries(self, filtered_fragments: dict) -> None:
        """
        Add new entries to the existing database.

        Parameters
        ----------
        filtered_fragments : dict
            A dictionary containing 'smiles' and 'formula' information for new entries.
        """
        # Calculate molecular formulas in parallel
        molecular_formula = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self.smiles_to_molecular_formula)(smi)
            for smi in filtered_fragments["smiles"]
        )

        # Create a DataFrame with 'smiles' and 'formula' and convert to a list
        # of records
        self.new_smiles_dict = pd.DataFrame(
            {"formula": molecular_formula, "smiles": filtered_fragments["smiles"]}
        ).to_dict("records")

    def extract_rules(self) -> List:
        """
        Extract automatic rules from the database.

        Returns
        -------
        List
            A list of automatic rules extracted from the database.
        """
        db = RuleImputeManager(
            self.existing_database
        )  # Create a RuleImputeManager instance
        db.add_entries(self.new_smiles_dict)  # Add new entries to the database
        self.rule_list = (
            db.database
        )  # Set self.rule_list to the database attribute of db
        return self.rule_list  # Return self.rule_list
