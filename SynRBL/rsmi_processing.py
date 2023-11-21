from joblib import Parallel, delayed
from rdkit import Chem
import pandas as pd

def can_parse(rsmi, symbol='>>'):
    """
    Check if a RSMI string can be parsed into reactants and products.

    Parameters:
    rsmi (str): The RSMI string.

    Returns:
    bool: True if the RSMI string can be parsed, False otherwise.
    """

    react, prod = rsmi.split(symbol)
    return Chem.MolFromSmiles(prod) is not None and Chem.MolFromSmiles(react) is not None

class RSMIProcessing:
    """
    A class to process reaction SMILES (RSMI) data.
    """

    def __init__(self, rsmi=None, data=None, rsmi_col=None, symbol='>>', n_jobs=10, verbose=1, parallel=True, 
                 save_json=True, save_path_name='reaction.json.gz', orient='records', compression='gzip'):
        """
        Initialize the RSMIProcessing object.

        Parameters:
        - rsmi (str): The reaction SMILES string.
        - data (DataFrame): The DataFrame containing the RSMI data.
        - rsmi_col (str): The column name in the DataFrame that contains the RSMI data.
        - symbol (str): The symbol that separates reactants and products in the RSMI string.
        - n_jobs (int): The number of jobs to run in parallel.
        - verbose (int): The verbosity level.
        - save_json (bool): Whether to save the processed data to a JSON file.
        - save_path_name (str): The path and name of the JSON file to save.
        - orient (str): The orientation of the JSON file.
        - compression (str): The compression type for the JSON file.
        """

        self.rsmi = rsmi
        self.symbol = symbol
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.parallel = parallel
        self.data = data
        self.rsmi_col = rsmi_col
        self.save_json = save_json
        self.save_path_name = save_path_name
        self.orient = orient
        self.compression = compression


    def smi_splitter(self):
        """
        Split a RSMI string into reactants and products.

        Returns:
        tuple: The reactants and products, or a message if the RSMI string can't be parsed.
        """

        if can_parse(self.rsmi):
            return self.rsmi.split(self.symbol)
        else:
            return "Can't parse"

    def data_splitter(self):
        """
        Split the RSMI data into reactants and products.

        Returns:
        DataFrame: The processed data.
        """
        if self.parallel:
            parsable = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(delayed(can_parse)(rsmi) for rsmi in self.data[self.rsmi_col].values)
            self.data = self.data[parsable]
        else:
            self.data = self.data[self.data[self.rsmi_col].apply(can_parse)]
        split_smiles = self.data[self.rsmi_col].str.split(self.symbol, expand=True)
        self.data['reactants'] = split_smiles[0]
        self.data['products'] = split_smiles[1]

        if self.save_json:
            self.data.to_json(self.save_path_name, orient=self.orient, compression=self.compression)

        return self.data
