from joblib import Parallel, delayed
from rdkit import Chem
import pandas as pd
from SynRBL.rsmi_utils import save_database
def can_parse(rsmi, symbol='>>'):
    """
    Check if a RSMI string can be parsed into reactants and products.

    Parameters
    ----------
    rsmi : str
        The reaction smiles (RSMI) string to be checked for parsability.
    symbol : str, optional
        The symbol used to separate reactants and products in the RSMI string (default is '>>').

    Returns
    -------
    bool
        True if the RSMI string can be parsed into valid reactant and product SMILES, False otherwise.
    """

    # Split the RSMI string into reactants and products using the provided symbol
    react, prod = rsmi.split(symbol)
    # Check if both reactants and products can be converted into RDKit molecule objects
    return Chem.MolFromSmiles(prod) is not None and Chem.MolFromSmiles(react) is not None


class RSMIProcessing:
    """
    A class to process reaction SMILES (RSMI) data.

    Parameters
    ----------
    rsmi : str, optional
        The reaction SMILES string to be processed.
    data : DataFrame, optional
        The DataFrame containing the RSMI data.
    rsmi_col : str, optional
        The column name in the DataFrame that contains the RSMI data.
    symbol : str, optional
        The symbol that separates reactants and products in the RSMI string (default is '>>').
    n_jobs : int, optional
        The number of jobs to run in parallel (default is 10).
    verbose : int, optional
        The verbosity level (default is 1).
    parallel : bool, optional
        Whether to use parallel processing (default is True).
    save_json : bool, optional
        Whether to save the processed data to a JSON file (default is True).
    save_path_name : str, optional
        The path and name of the JSON file to save (default is 'reaction.json.gz').
    orient : str, optional
        The orientation of the JSON file (default is 'records').
    compression : str, optional
        The compression type for the JSON file (default is 'gzip').

    Example
    -------
    # Create an instance of the RSMIProcessing class with a DataFrame
    >>> data = pd.DataFrame({'rsmi': ['CCO>>CCOCC', 'CC>>C']})
    >>> processor = RSMIProcessing(data=data, rsmi_col='rsmi', parallel=False)
    # Split the RSMI data into reactants and products
    >>> processed_data = processor.data_splitter()
    # Display the processed data
    >>> print(processed_data)
    """

    def __init__(self, rsmi=None, data=None, rsmi_col=None, symbol='>>', n_jobs=10, verbose=1, parallel=True, 
                 save_json=True, save_path_name='reaction.json.gz', orient='records', compression='gzip'):
        self.rsmi = rsmi  # Reaction SMILES string
        self.symbol = symbol  # Separator symbol in RSMI
        self.n_jobs = n_jobs  # Number of parallel jobs
        self.verbose = verbose  # Verbosity level
        self.parallel = parallel  # Flag for parallel processing
        self.data = data  # DataFrame with RSMI data
        self.rsmi_col = rsmi_col  # Column name for RSMI in DataFrame
        self.save_json = save_json  # Flag to save processed data as JSON
        self.save_path_name = save_path_name  # Path and filename for saving JSON
        self.orient = orient  # JSON file orientation
        self.compression = compression  # JSON file compression type


    def smi_splitter(self):
        """
        Split a RSMI string into reactants and products.

        Returns
        -------
        tuple
            The reactants and products as separate strings, if the RSMI string can be parsed.
        str
            A message indicating the RSMI string can't be parsed, if applicable.
        """

        # Check if the RSMI string can be parsed
        if can_parse(self.rsmi, self.symbol):
            # Split the RSMI string into reactants and products
            return self.rsmi.split(self.symbol)
        else:
            # Return a message if the RSMI string cannot be parsed
            return "Can't parse"

    def data_splitter(self):
        """
        Split the RSMI data in the DataFrame into reactants and products.

        Returns
        -------
        DataFrame
            The processed DataFrame with separate columns for reactants and products.

        Example
        -------
        >>> data = pd.DataFrame({'rsmi': ['C>>CC', 'CC>>CCC']})
        >>> processor = RSMIProcessing(data=data, rsmi_col='rsmi', parallel=False)
        >>> processed_data = processor.data_splitter()
        >>> print(processed_data)
        """
        # Check if parallel processing is enabled
        if self.parallel:
            # Use joblib's Parallel to concurrently process each RSMI string in the DataFrame
            # 'can_parse' function is applied to each RSMI string to check if it's parsable
            parsable = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(can_parse)(rsmi) for rsmi in self.data[self.rsmi_col].values
            )
            # Filter the data to include only parsable RSMI strings
            self.data = self.data[parsable]
        else:
            # If parallel processing is not enabled, apply 'can_parse' function sequentially
            # to each RSMI string in the DataFrame and filter parsable ones
            self.data = self.data[self.data[self.rsmi_col].apply(can_parse)]

        # Split each RSMI string in the DataFrame into reactants and products
        # 'expand=True' splits the string into separate columns
        split_smiles = self.data[self.rsmi_col].str.split(self.symbol, expand=True)

        # Assign the first part of the split (reactants) to a new column in the DataFrame
        self.data['reactants'] = split_smiles[0]
        # Assign the second part of the split (products) to another new column
        self.data['products'] = split_smiles[1]

        

        # Check if there's a need to save the processed data to a JSON file
        if self.save_json:
            data=self.data.to_dict(orient=self.orient)
            save_database(data, self.save_path_name)
            # Save the DataFrame to a JSON file with specified format and compression
            #self.data.to_json(self.save_path_name, orient=self.orient, compression=self.compression)

        # Return the processed DataFrame with separate columns for reactants and products
        return self.data