from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdFMCS

class MCSMissingGraphAnalyzer:
    """A class for detecting missing graph in reactants and products using MCS and RDKit."""

    def __init__(self):
        """Initialize the MolecularOperations class."""
        pass
    
    @staticmethod
    def get_smiles(reaction_dict):
        """
        Extract reactant and product SMILES strings from a reaction dictionary.

        Parameters:
        - reaction_dict: dict
            A dictionary containing 'reactants' and 'products' as keys.

        Returns:
        - tuple
            A tuple containing reactant SMILES and product SMILES strings.
        """
        return reaction_dict['reactants'], reaction_dict['products']

    @staticmethod
    def convert_smiles_to_molecule(smiles):
        """
        Convert a SMILES string to an RDKit molecule object.

        Parameters:
        - smiles: str
            The SMILES string representing a molecule.

        Returns:
        - rdkit.Chem.Mol
            The RDKit molecule object.
        """
        return Chem.MolFromSmiles(smiles)

    @staticmethod
    def mol_to_smiles(mol):
        """
        Convert an RDKit molecule object to a SMILES string.

        Parameters:
        - mol: rdkit.Chem.Mol
            The RDKit molecule object.

        Returns:
        - str or None
            The SMILES string representation of the molecule, or None if the molecule is None.
        """
        return Chem.MolToSmiles(mol) if mol else None

    @staticmethod
    def mol_to_smarts(mol):
        """
        Convert an RDKit molecule object to a SMARTS string.

        Parameters:
        - mol: rdkit.Chem.Mol
            The RDKit molecule object.

        Returns:
        - str or None
            The SMARTS string representation of the molecule, or None if the molecule is None.
        """
        return Chem.MolToSmarts(mol) if mol else None

    @staticmethod
    def find_maximum_common_substructure(mol1, mol2, params=None):
        """
        Find the maximum common substructure (MCS) between two molecules.

        Parameters:
        - mol1, mol2: rdkit.Chem.Mol
            The RDKit molecule objects to compare.

        Returns:
        - rdkit.Chem.Mol or None
            The RDKit molecule object representing the MCS, or None if MCS search was canceled.
        """
        mcs_result = rdFMCS.FindMCS([mol1, mol2], params=params)
        if mcs_result.canceled:
            return None
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        return mcs_mol
    
    @staticmethod
    def IterativeMCSReactionPairs(reactant_mol_list, product_mol, params=None, sort='MCS', remove_substructure=False):
        """
        Find the MCS for each reactant fragment with the product, updating the product after each step.
        Reactants are processed based on the size of their MCS with the product at each iteration.

        Parameters:
        - reactant_mol_list: list of rdkit.Chem.Mol
            List of RDKit molecule objects for reactants.
        - product_mol: rdkit.Chem.Mol
            RDKit molecule object for the product.
        - sort (str): 
            Method of sorting reactants, either 'MCS' or 'Fragments'.
        - remove_substructure (bool): 
            If True, update the product by removing the MCS substructure.
         - params (rdkit.Chem.rdFMCS.MCSParameters): Parameters for RDKit's rdFMCS.

        Returns:
        - list of rdkit.Chem.Mol
            List of RDKit molecule objects representing the MCS for each reactant-product pair.
        - list of rdkit.Chem.Mol
            Sorted list of reactant molecule objects.
        """

        # Sort reactants based on the specified method
        if sort == 'MCS':
            mcs_results = [(reactant, rdFMCS.FindMCS([reactant, product_mol], params)) for reactant in reactant_mol_list]
            mcs_results = [(reactant, mcs_result) for reactant, mcs_result in mcs_results if not mcs_result.canceled]
            sorted_reactants = sorted(mcs_results, key=lambda x: x[1].numAtoms, reverse=True)
        elif sort == 'Fragments':
            sorted_reactants = sorted(reactant_mol_list, key=lambda x: x.GetNumAtoms(), reverse=True)
        else:
            raise ValueError("Invalid sort method. Choose 'MCS' or 'Fragments'.")

        mcs_list = []
        current_product = product_mol

        for reactant, _ in sorted_reactants:
            # Calculate the MCS with the current product
            mcs_result = rdFMCS.FindMCS([reactant, current_product], params)

            if not mcs_result.canceled:
                mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                mcs_list.append(mcs_mol)

                # Conditional substructure removal
                if remove_substructure:
                    current_product = Chem.DeleteSubstructs(Chem.RWMol(current_product), mcs_mol)
                    try:
                        Chem.SanitizeMol(current_product)
                    except:
                        pass

        return mcs_list, [reactant for reactant, _ in sorted_reactants]

    @staticmethod
    def fit(reaction_dict, RingMatchesRingOnly=True, CompleteRingsOnly=True, Timeout = 60,
            sort='MCS', remove_substructure=False, AtomCompare=False, BondTyper = False):
        """
        Process a reaction dictionary to find MCS, missing parts in reactants and products.

        Parameters:
        - reaction_dict: dict
            A dictionary containing 'reactants' and 'products' as keys.

        Returns:
        - tuple
            A tuple containing lists of MCS, missing parts in reactants, missing parts in products,
            reactant molecules, and product molecules.
        """
        
        # define parameters
        params = rdFMCS.MCSParameters()
        #params.AtomTyper = rdFMCS.AtomCompare.CompareElements
        #params.BondTyper = rdFMCS.BondCompare.CompareOrder
        params.Timeout = Timeout
        params.BondCompareParameters.RingMatchesRingOnly = RingMatchesRingOnly
        params.BondCompareParameters.CompleteRingsOnly = CompleteRingsOnly


        # Calculate the MCS for each reactant with the product 
        reactant_smiles, product_smiles = MCSMissingGraphAnalyzer.get_smiles(reaction_dict)
        reactant_mol_list = [MCSMissingGraphAnalyzer.convert_smiles_to_molecule(smiles) for smiles in reactant_smiles.split('.')]
        product_mol = MCSMissingGraphAnalyzer.convert_smiles_to_molecule(product_smiles)

        mcs_list, sorted_reactants = MCSMissingGraphAnalyzer.IterativeMCSReactionPairs(reactant_mol_list, product_mol,  params,
                                                                                       sort = sort, remove_substructure=remove_substructure)

        return mcs_list , sorted_reactants, product_mol