from SynRBL.SynChemistry.functional_group_checker import FunctionalGroupChecker
from rdkit import Chem

class PeroxidGroupImputer:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def impute_peroxide(reactant_smiles: str, product_smiles: str) -> tuple:
        """
        Impute a peroxide functional group in the reactant SMILES string and update the product SMILES accordingly.
        """
        updated = False
        reactant_components = reactant_smiles.split('.')
        for i, smiles in enumerate(reactant_components):
            if FunctionalGroupChecker.check_peroxide(smiles):
                reactant_components[i] = smiles.replace('OO', 'O.O')
                updated = True

        if updated:
            updated_reactant_smiles = '.'.join(reactant_components)
            updated_product_smiles = product_smiles + '.' + Chem.CanonSmiles(updated_reactant_smiles)
            return updated_reactant_smiles, updated_product_smiles
        else:
            return reactant_smiles, product_smiles
    
    @staticmethod
    def impute_peracid(reactant_smiles: str, product_smiles: str) -> tuple:
        """
        Impute a peracid functional group in the reactant SMILES string and update the product SMILES accordingly.
        """
        updated = False
        reactant_components = reactant_smiles.split('.')
        for i, smiles in enumerate(reactant_components):
            if FunctionalGroupChecker.check_peracid(smiles):
                reactant_components[i] = smiles.replace('O=C(OO)', 'O=C(O)')
                updated = True

        if updated:
            updated_reactant_smiles = '.'.join(reactant_components)
            updated_product_smiles = product_smiles + '.' + Chem.CanonSmiles(updated_reactant_smiles)
            return updated_reactant_smiles, updated_product_smiles
        else:
            return reactant_smiles, product_smiles