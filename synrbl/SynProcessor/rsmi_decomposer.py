from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import Descriptors
from joblib import Parallel, delayed
import pandas as pd

from typing import Dict, List, Tuple, Union


class RSMIDecomposer:
    """
    A class for decomposing SMILES strings into atomic compositions and
    calculating molecular weights.

    Attributes
    ----------
    smiles : str, optional
        The SMILES string to decompose.
    data : list or DataFrame, optional
        The data containing the SMILES strings to decompose.
    reactant_col : str, optional
        The column or key name in the data for reactant SMILES strings.
    product_col : str, optional
        The column or key name in the data for product SMILES strings.
    parallel : bool, optional
        Whether to use parallel processing (default is True).
    n_jobs : int, optional
        The number of jobs to run in parallel (default is 10).
    verbose : int, optional
        The verbosity level (default is 1).
    """

    atomic_symbols = {
        0: "Q",
        1: "H",
        2: "He",
        3: "Li",
        4: "Be",
        5: "B",
        6: "C",
        7: "N",
        8: "O",
        9: "F",
        10: "Ne",
        11: "Na",
        12: "Mg",
        13: "Al",
        14: "Si",
        15: "P",
        16: "S",
        17: "Cl",
        18: "Ar",
        19: "K",
        20: "Ca",
        21: "Sc",
        22: "Ti",
        23: "V",
        24: "Cr",
        25: "Mn",
        26: "Fe",
        27: "Co",
        28: "Ni",
        29: "Cu",
        30: "Zn",
        31: "Ga",
        32: "Ge",
        33: "As",
        34: "Se",
        35: "Br",
        36: "Kr",
        37: "Rb",
        38: "Sr",
        39: "Y",
        40: "Zr",
        41: "Nb",
        42: "Mo",
        43: "Tc",
        44: "Ru",
        45: "Rh",
        46: "Pd",
        47: "Ag",
        48: "Cd",
        49: "In",
        50: "Sn",
        51: "Sb",
        52: "Te",
        53: "I",
        54: "Xe",
        55: "Cs",
        56: "Ba",
        57: "La",
        58: "Ce",
        59: "Pr",
        60: "Nd",
        61: "Pm",
        62: "Sm",
        63: "Eu",
        64: "Gd",
        65: "Tb",
        66: "Dy",
        67: "Ho",
        68: "Er",
        69: "Tm",
        70: "Yb",
        71: "Lu",
        72: "Hf",
        73: "Ta",
        74: "W",
        75: "Re",
        76: "Os",
        77: "Ir",
        78: "Pt",
        79: "Au",
        80: "Hg",
        81: "Tl",
        82: "Pb",
        83: "Bi",
        84: "Po",
        85: "At",
        86: "Rn",
        # This covers elements up to Radon (Rn). Extend further if needed.
    }

    def __init__(
        self,
        smiles: str = None,
        data: Union[List[str], pd.DataFrame] = None,
        reactant_col: str = "reactants",
        product_col: str = "products",
        parallel: bool = True,
        n_jobs: int = 4,
        verbose: int = 1,
    ) -> None:
        """
        Initialize the RSMIDecomposer object.

        Args:
            smiles (str): The SMILES string to decompose.
            data (list or DataFrame): The data containing the SMILES strings
                to decompose.
            reactant_col (str): The column or key name in the data for reactant
                SMILES strings.
            product_col (str): The column or key name in the data for product
                SMILES strings.
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

    def calculate_mol_weight(self, smiles: str) -> float:
        """
        Calculates the molecular weight of a molecule represented by a
        SMILES string.

        Args:
            smiles: The SMILES string of the molecule.

        Returns:
            The molecular weight of the molecule, or None if the SMILES string
            is invalid.

        Example:
        >>> decomposer = RSMIDecomposer()
        >>> decomposer.calculate_mol_weight('CCO')
        46.069
        """
        molecule = Chem.MolFromSmiles(smiles)
        if molecule:
            return Descriptors.MolWt(molecule)

    def data_decomposer(self) -> Tuple[List[Dict[str, int]], List[Dict[str, int]]]:
        """
        Decomposes SMILES strings in the data into atomic compositions for
        reactants and products.

        Returns:
        tuple of lists: Two lists containing dictionaries of atomic
            compositions for reactants and products.

        Example:
        >>> data = pd.DataFrame(
            {"reactants": ["CCO", "CC"], "products": ["C=O", "C=C"]}
        )
        >>> decomposer = RSMIDecomposer(data=data, parallel=False)
        >>> reactants, products = decomposer.data_decomposer()
        >>> reactants  # Outputs [{'C': 2, 'O': 1, 'H': 6}, {'C': 2, 'H': 6}]
        >>> products  # Outputs [{'C': 1, 'O': 1, 'H': 2}, {'C': 2, 'H': 4}]
        """
        # Extract reactants and products based on the provided data format
        if isinstance(self.data, list):
            reactants = [item[self.reactant_col] for item in self.data]
            products = [item[self.product_col] for item in self.data]
        else:  # Assuming data is a DataFrame
            reactants = self.data[self.reactant_col]
            products = self.data[self.product_col]

        # Decompose reactants and products using parallel processing if enabled
        if self.parallel:
            react_dict = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(RSMIDecomposer.decompose)(smiles) for smiles in reactants
            )
            prods_dict = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
                delayed(RSMIDecomposer.decompose)(smiles) for smiles in products
            )
        else:
            react_dict = [RSMIDecomposer.decompose(smiles) for smiles in reactants]
            prods_dict = [RSMIDecomposer.decompose(smiles) for smiles in products]

        return react_dict, prods_dict

    @staticmethod
    def decompose(smiles: str) -> Dict[str, int]:
        """
        Decomposes a SMILES string into a dictionary of atomic counts and charges.

        Parameters:
        smiles : str
            The SMILES string to decompose.

        Returns:
        dict
            A dictionary with atomic symbols as keys and counts as values.
            Charge is represented as 'Q'.

        Example:
        >>> RSMIDecomposer.decompose('CCO')
        {'C': 2, 'O': 1, 'H': 6, 'Q': 0}
        """
        molecule = Chem.MolFromSmiles(smiles)
        comp = defaultdict(int)
        if molecule:
            molecule_with_Hs = Chem.AddHs(molecule)

            for atom in molecule_with_Hs.GetAtoms():
                atomic_symbol = RSMIDecomposer.atomic_symbols.get(
                    atom.GetAtomicNum(), "Unknown"
                )
                comp[atomic_symbol] += 1

            charge = Chem.GetFormalCharge(molecule_with_Hs)
            if charge != 0:
                comp["Q"] = charge

        return dict(comp)
