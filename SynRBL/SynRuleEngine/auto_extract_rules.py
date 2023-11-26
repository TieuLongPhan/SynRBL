from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from joblib import Parallel, delayed
import pandas as pd
from SynRBL.SynRuleEngine.rule_data_manager import RuleImputeManager

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

    def __init__(self, existing_database=None, n_jobs=-5, verbose=1):
        self.rule_list = []
        self.existing_database = existing_database or []  # Initialize existing database
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def smiles_to_molecular_formula(smiles):
        """
        Convert a SMILES string to a molecular formula.

        Parameters
        ----------
        smiles : str
            The SMILES string of the molecule.

        Returns
        -------
        str
            The molecular formula of the molecule.
        """
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            formula = rdMolDescriptors.CalcMolFormula(molecule)
            return formula

    def add_new_entries(self, filtered_fragments):
        """
        Add new entries to the existing database.

        Parameters
        ----------
        filtered_fragments : dict
            A dictionary containing 'smiles' and 'formula' information for new entries.
        """
        # Calculate molecular formulas in parallel
        molecular_formula = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self.smiles_to_molecular_formula)(smi) for smi in filtered_fragments['smiles'])
        
        # Create a DataFrame with 'smiles' and 'formula' and convert to a list of records
        self.new_smiles_dict = pd.DataFrame({'formula': molecular_formula, 'smiles': filtered_fragments['smiles']}).to_dict('records')

    def extract_rules(self):
        """
        Extract automatic rules from the database.

        Returns
        -------
        list
            A list of automatic rules extracted from the database.
        """
        db = RuleImputeManager(self.existing_database)
        db.add_entries(self.new_smiles_dict)  # Add new entries to the database
        new_rule = db.database
        self.rule_list = new_rule
        return self.rule_list