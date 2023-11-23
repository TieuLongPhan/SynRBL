from collections import defaultdict
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from joblib import Parallel, delayed

class RSMIDecomposer:
    
    def __init__(self, smiles=None, data=None, reactant_col='reactants', product_col='products', parallel=True, n_jobs=10, verbose=1):
        """
        Initialize the RSMIDecomposer object.

        Parameters:
        smiles (str): The SMILES string to decompose.
        data (list or DataFrame): The data containing the SMILES strings to decompose.
        reactant_col (str): The column or key name in the data that contains the reactant SMILES strings.
        product_col (str): The column or key name in the data that contains the product SMILES strings.
        parallel (bool): Whether to use parallel processing.
        n_jobs (int): The number of jobs to run in parallel.
        verbose (int): The verbosity level.
        """

        self.smiles = smiles
        self.data = data
        self.reactant_col = reactant_col
        self.product_col = product_col
        self.parallel = parallel
        self.n_jobs = n_jobs
        self.verbose = verbose

    def calculate_mol_weight(self, smiles):
        """
        Calculate the molecular weight of the molecule represented by the SMILES string.

        Parameters:
        smiles (str): The SMILES string to analyze.

        Returns:
        float: The molecular weight of the molecule.
        """

        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            return Descriptors.MolWt(molecule)

    def data_decomposer(self):
        """
        Decompose the SMILES strings in the data into atomic compositions.

        Returns:
        tuple: Two lists of dictionaries of atomic number and number of atoms for the reactants and products.
        """

        if isinstance(self.data, list):
            reactants = [item[self.reactant_col] for item in self.data]
            products = [item[self.product_col] for item in self.data]
        else:
            reactants = self.data[self.reactant_col]
            products = self.data[self.product_col]

        if self.parallel:
            react_dict = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(delayed(decompose)(smiles) for smiles in reactants)
            prods_dict = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(delayed(decompose)(smiles) for smiles in products)
        else:
            react_dict = [decompose(smiles) for smiles in reactants]
            prods_dict = [decompose(smiles) for smiles in products]

        return react_dict, prods_dict
    
def decompose(smiles):
    """
    Get the composition of an RDKit molecule:
    Atomic counts, including hydrogen atoms, and any charge.
    For example, fluoride ion (chemical formula F-, SMILES string [F-])
    returns {9: 1, 0: -1}.

    Parameters:
    smiles (str): The SMILES string to analyze.

    Returns:
    dict: A dictionary of atomic number and number of atoms.
    """

    # Check that there is a valid molecule
    molecule = Chem.MolFromSmiles(smiles)
    if molecule:
        # Add hydrogen atoms--RDKit excludes them by default
        molecule_with_Hs = Chem.AddHs(molecule)
        comp = defaultdict(lambda: 0)

        # Get atom counts
        for atom in molecule_with_Hs.GetAtoms():
            comp[atom.GetAtomicNum()] += 1

        # If charged, add charge as "atomic number" 0
        charge = Chem.GetFormalCharge(molecule_with_Hs)
        if charge != 0:
            comp[0] = charge
        else:
            comp[0] = 0
        return dict(sorted(comp.items()))
