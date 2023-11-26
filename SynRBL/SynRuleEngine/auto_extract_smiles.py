from rdkit import Chem
from rdkit.Chem import Descriptors
from joblib import Parallel, delayed
import pandas as pd

class AutomaticSmilesExtraction:
    """
    A class for automatically extracting information from reaction SMILES (RSMI) data.

    Parameters
    ----------
    reactions : list of dict
        A list of dictionaries containing reaction data. Each dictionary should
        contain 'reactants' and 'products' keys, where the values are SMILES strings.
    n_jobs : int, optional
        The number of jobs to run in parallel (default is -1 for maximum available).
    verbose : int, optional
        The verbosity level (default is 1).

    Attributes
    ----------
    smiles_list : list of str
        List of SMILES strings extracted from the reactions.
    mw : list of float
        List of molecular weights calculated from the SMILES strings.
    n_C : list of int
        List of counts of carbon atoms in the molecules represented by SMILES.

    Example
    -------
    # Create an instance of the AutomaticSmilesExtraction class with a list of reaction dictionaries
    >>> reactions = [
    ...     {'reactants': 'CCO', 'products': 'CCOCC'},
    ...     {'reactants': 'CC', 'products': 'C'}
    ... ]
    >>> extractor = AutomaticSmilesExtraction(reactions, n_jobs=-1, verbose=1)
    # Access extracted data
    >>> print(extractor.mw)
    [46.069, 74.123]
    >>> print(extractor.n_C)
    [2, 2]
    """

    def __init__(self, reactions, n_jobs=-1, verbose=1):
        self.reactions = reactions
        self.smiles_list = self.get_smiles(reactions)
        # Use n_jobs and verbose as arguments in Parallel
        self.mw = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self.calculate_mol_weight)(smi) for smi in self.smiles_list
        )
        self.n_C = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(self.count_carbon_atoms)(smi) for smi in self.smiles_list
        )

    def get_smiles(self, list_of_dicts):
        """
        Extract SMILES strings from a list of dictionaries containing reaction data.

        Parameters
        ----------
        list_of_dicts : list of dict
            A list of dictionaries containing reaction data. Each dictionary should
            contain 'reactants' and 'products' keys, where the values are SMILES strings.

        Returns
        -------
        list of str
            List of SMILES strings extracted from the reactions.
        """
        smiles_list = []
        for reaction in list_of_dicts:
            reactants = reaction.get('reactants', '')
            products = reaction.get('products', '')
            smiles_list.extend([reactants, products])
        return smiles_list

    @staticmethod
    def calculate_mol_weight(smiles):
        """
        Calculate the molecular weight of a molecule represented by a SMILES string.

        Parameters
        ----------
        smiles : str
            The SMILES string of the molecule.

        Returns
        -------
        float
            The molecular weight of the molecule, or None if the SMILES string is invalid.
        """
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            return Descriptors.MolWt(molecule)

    @staticmethod
    def count_carbon_atoms(smiles):
        """
        Count the number of carbon (C) atoms in a molecule represented by its SMILES string.

        Parameters
        ----------
        smiles : str
            The SMILES representation of the molecule.

        Returns
        -------
        int
            The number of carbon atoms in the molecule.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES string")
        num_carbon_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'C')

        return num_carbon_atoms

    @staticmethod
    def get_fragments(input_dict, mw=100, n_C=10, combination='union'):
        """
        Filter a dictionary of lists based on conditions specified by mw and n_C.

        Parameters
        ----------
        input_dict : dict
            A dictionary where each key corresponds to a list of values.
        mw : float, optional
            Maximum allowed molecular weight. Default is 100.
        n_C : int, optional
            Minimum required number of carbon atoms. Default is 10.
        combination : str, optional
            Filter combination ('union' or 'intersection'). Default is 'union'.

        Returns
        -------
        dict
            Filtered dictionary with the same keys and filtered lists of values.
        """
        filtered_dict = {key: [] for key in input_dict.keys()}

        for i in range(len(input_dict['smiles'])):
            include_fragment = False

            if combination == 'union':
                if input_dict['mw'][i] <= mw or input_dict['n_C'][i] <= n_C:
                    include_fragment = True

            if combination == 'intersection':
                if input_dict['mw'][i] <= mw and input_dict['n_C'][i] <= n_C:
                    include_fragment = True

            if include_fragment:
                for key in input_dict.keys():
                    filtered_dict[key].append(input_dict[key][i])

        return filtered_dict
