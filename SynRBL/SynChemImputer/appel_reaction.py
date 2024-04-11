from typing import Dict
from rdkit import Chem


class AppelReaction:
    TCM = "ClC(Cl)(Cl)Cl"
    TCM_product = "ClC(Cl)Cl"
    TBM = "BrC(Br)(Br)Br"
    TBM_product = "BrC(Br)Br"
    TPP = "c1ccccc1P(c2ccccc2)c3ccccc3"
    TPPO = "O=P(c1ccccc1)(c2ccccc2)c3ccccc3"

    def __init__(self):
        pass

    @staticmethod
    def check_alcohol_group(smiles: str) -> bool:
        """
        Check for the presence of an alcohol (hydroxyl, -OH) group in a molecule.

        Args:
        smiles (str): A SMILES string representing the molecule.

        Returns:
        bool: True if the alcohol group is present, False otherwise.
        """
        alcohol_pattern = Chem.MolFromSmarts("[OH]")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(alcohol_pattern) if mol else False

    @staticmethod
    def check_appel_reaction(reactants: str) -> bool:
        """
        Check if the Appel reaction conditions are met in the reactants.

        Args:
        reactants (str): SMILES string of the reactants.

        Returns:
        bool: True if Appel reaction conditions are met, False otherwise.
        """
        return any(
            reagent in reactants for reagent in [AppelReaction.TCM, AppelReaction.TBM]
        ) and AppelReaction.check_alcohol_group(reactants)

    @staticmethod
    def check_missing_reagent(reactants: str) -> bool:
        """
        Check if the key reagent (Triphenylphosphine) is missing in the reactants.

        Args:
        reactants (str): SMILES string of the reactants.

        Returns:
        bool: True if Triphenylphosphine is missing, False otherwise.
        """
        return AppelReaction.TPP not in reactants

    @staticmethod
    def check_missing_products(products: str) -> bool:
        """
        Check if the expected product (Triphenylphosphine oxide) is missing in
        the products.

        Args:
        products (str): SMILES string of the products.

        Returns:
        bool: True if Triphenylphosphine oxide is missing, False otherwise.
        """
        return AppelReaction.TPPO not in products

    def fit(
        self, reaction_dict: Dict[str, str], rmsi_col: str, symbol: str = ">>"
    ) -> Dict[str, str]:
        """
        Modify the reaction dictionary to include missing reactants or products
        for Appel reaction.

        Args:
        reaction_dict (Dict[str, str]): Dictionary containing reaction information.
        rmsi_col (str): Key for the reaction SMILES string in the dictionary.
        symbol (str): Symbol used to separate reactants and products. Default is '>>'.

        Returns:
        Dict[str, str]: Updated reaction dictionary with modified SMILES string.
        """
        reaction = reaction_dict[rmsi_col]
        reactants, products = reaction.split(symbol)

        if self.check_appel_reaction(reactants):
            if self.check_missing_reagent(reactants):
                reactants += "." + AppelReaction.TPP
            if self.check_missing_products(products):
                products += "." + AppelReaction.TPPO

            # Add TCM or TBM products if they are missing
            if (
                AppelReaction.TCM in reactants
                and AppelReaction.TCM_product not in products
            ):
                products += "." + AppelReaction.TCM_product
            elif (
                AppelReaction.TBM in reactants
                and AppelReaction.TBM_product not in products
            ):
                products += "." + AppelReaction.TBM_product

            reaction_dict[rmsi_col] = symbol.join([reactants, products])

        return reaction_dict
