from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdFMCS
from rdkit.Chem import rdRascalMCES
from SynRBL.SynMCS.SubStructure.substructure_analyzer import SubstructureAnalyzer
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
    def find_maximum_common_substructure(mol1, mol2, params):
        """
        Find the maximum common substructure (MCS) between two molecules.

        Parameters:
        - mol1, mol2: rdkit.Chem.Mol
            The RDKit molecule objects to compare.

        Returns:
        - rdkit.Chem.Mol or None
            The RDKit molecule object representing the MCS, or None if MCS search was canceled.
        """
        if params == None:
            mcs_result = rdFMCS.FindMCS([mol1, mol2])
        else:
            mcs_result = rdFMCS.FindMCS([mol1, mol2], params)
        if mcs_result.canceled:
            return None
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        return mcs_mol
    
    def IterativeMCSReactionPairs(reactant_mol_list, product_mol, params=None, method = 'MCIS', sort='MCIS', remove_substructure=True):
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
        if sort == 'MCIS':
            if params == None:
                params = rdFMCS.MCSParameters()
            mcs_results = [(reactant, rdFMCS.FindMCS([reactant, product_mol], params)) for reactant in reactant_mol_list]
            mcs_results = [(reactant, mcs_result) for reactant, mcs_result in mcs_results if not mcs_result.canceled]
            sorted_reactants = sorted(mcs_results, key=lambda x: x[1].numAtoms, reverse=True)
        elif sort == 'MCES':
            if params == None:
                params = rdRascalMCES.RascalOptions()
            mcs_results = [(reactant, rdRascalMCES.FindMCES(reactant, product_mol, params)[0]) for reactant in reactant_mol_list]
            mcs_results = [(reactant, mcs_result) for reactant, mcs_result in mcs_results if hasattr(mcs_result, 'atomMatches')]
            sorted_reactants = sorted(mcs_results, key=lambda x: len(x[1].atomMatches()), reverse=True)
        elif sort == 'Fragments':
            sorted_reactants = sorted(reactant_mol_list, key=lambda x: x.GetNumAtoms(), reverse=True)
        else:
            raise ValueError("Invalid sort method. Choose 'MCS' or 'Fragments'.")


        mcs_list = []
        current_product = product_mol
        for reactant, _ in sorted_reactants:
            # Calculate the MCS with the current product
            if method == 'MCIS':
                mcs_result = rdFMCS.FindMCS([reactant, current_product], params)
            elif method == 'MCES':
                mcs_result = rdRascalMCES.FindMCES(reactant, current_product, params)[0]
            else:
                raise ValueError("Invalid method. Choose 'MCIS' or 'MCES'.")
            
            if not mcs_result.canceled if method == 'MCIS' else hasattr(mcs_result, 'atomMatches'):
                mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                mcs_list.append(mcs_mol)
                # Conditional substructure removal
                if remove_substructure:

                    # Identify the optimal substructure
                    analyzer = SubstructureAnalyzer()
                    optimal_substructure = analyzer.identify_optimal_substructure(parent_mol=current_product, child_mol=mcs_mol)
                    if optimal_substructure:
                        rw_mol = Chem.RWMol(current_product)
                        # Remove atoms in descending order of their indices
                        for atom_idx in sorted(optimal_substructure, reverse=True):
                            if atom_idx < rw_mol.GetNumAtoms():  # Check if the index is valid
                                rw_mol.RemoveAtom(atom_idx)
                            else:
                                pass
                        current_product = rw_mol.GetMol()

                try:
                    Chem.SanitizeMol(current_product)
                except:
                    pass

        return mcs_list, [reactant for reactant, _ in sorted_reactants]

    @staticmethod
    def fit(reaction_dict, RingMatchesRingOnly=True, CompleteRingsOnly=True, Timeout = 60, similarityThreshold = 0.5,
            sort='MCIS', method='MCIS', remove_substructure=True, ignore_atom_map=False, ignore_bond_order = False):
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

        if method == 'MCIS':
            params = rdFMCS.MCSParameters()
            params.Timeout = Timeout
            params.BondCompareParameters.RingMatchesRingOnly = RingMatchesRingOnly
            params.BondCompareParameters.CompleteRingsOnly = CompleteRingsOnly
            if ignore_bond_order:
                params.BondTyper = rdFMCS.BondCompare.CompareAny
            if ignore_atom_map:
                params.AtomTyper = rdFMCS.AtomCompare.CompareAny

        elif method == 'MCES':
            params = rdRascalMCES.RascalOptions()
            params.returnEmptyMCES = True
            params.singleLargestFrag = False
            params.timeout = Timeout
            params.similarityThreshold = similarityThreshold


        # Calculate the MCS for each reactant with the product 
        reactant_smiles, product_smiles = MCSMissingGraphAnalyzer.get_smiles(reaction_dict)
        reactant_mol_list = [MCSMissingGraphAnalyzer.convert_smiles_to_molecule(smiles) for smiles in reactant_smiles.split('.')]
        product_mol = MCSMissingGraphAnalyzer.convert_smiles_to_molecule(product_smiles)

        mcs_list, sorted_reactants = MCSMissingGraphAnalyzer.IterativeMCSReactionPairs(reactant_mol_list, product_mol,  params, 
                                                                                       method=method, sort = sort, remove_substructure=remove_substructure)

        return mcs_list , sorted_reactants, reactant_mol_list, product_mol