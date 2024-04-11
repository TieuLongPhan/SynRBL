from rdkit import Chem


class PeroxidGroupImputer:
    """A class for imputing peroxide and peracid groups in chemical reactions."""

    @staticmethod
    def check_peroxide(smiles: str) -> bool:
        """Check for the presence of a peroxide substructure in a molecule."""
        try:
            peroxide_pattern = Chem.MolFromSmarts("OO")
            mol = Chem.MolFromSmiles(smiles)
            return (
                mol.HasSubstructMatch(peroxide_pattern)
                if mol and not PeroxidGroupImputer.check_peracid(smiles)
                else False
            )
        except Exception:
            return False

    @staticmethod
    def check_peracid(smiles: str) -> bool:
        """Check for the presence of a peracid substructure in a molecule."""
        try:
            peracid_pattern = Chem.MolFromSmarts("C(OO)=O")
            mol = Chem.MolFromSmiles(smiles)
            return mol.HasSubstructMatch(peracid_pattern) if mol else False
        except Exception:
            return False

    @staticmethod
    def impute_peroxide(reactant_smiles: str, product_smiles: str) -> tuple:
        """Impute a peroxide functional group in the reactant SMILES string and
        update the product SMILES accordingly."""
        updated = False
        reactant_components = reactant_smiles.split(".")
        for smiles in reactant_components:
            if PeroxidGroupImputer.check_peroxide(smiles):
                updated_smiles = smiles.replace("OO", "O.O")
                updated = True

        if updated:
            updated_product_smiles = (
                product_smiles + "." + Chem.CanonSmiles(updated_smiles)
            )
            return reactant_smiles, updated_product_smiles
        else:
            return reactant_smiles, product_smiles

    @staticmethod
    def impute_peracid(reactant_smiles: str, product_smiles: str) -> tuple:
        """Impute a peracid functional group in the reactant SMILES string and
        update the product SMILES accordingly."""
        updated = False
        reactant_components = reactant_smiles.split(".")
        for smiles in reactant_components:
            if PeroxidGroupImputer.check_peracid(smiles):
                updated_smiles = Chem.CanonSmiles(
                    smiles.replace("O=C(OO)", "O=C(O)").replace("C(=O)OO", "C(=O)O")
                )
                updated = True

        if updated and updated_smiles not in product_smiles.split("."):
            updated_product_smiles = (
                product_smiles + "." + Chem.CanonSmiles(updated_smiles)
            )
            return reactant_smiles, updated_product_smiles
        else:
            return reactant_smiles, product_smiles

    def fix(self, reactions_dict: dict, rsmi_col: str, symbol: str = ">>") -> dict:
        """Process and update reactions containing peroxide or peracid groups."""
        reactants, products = reactions_dict[rsmi_col].split(symbol)
        if PeroxidGroupImputer.check_peroxide(reactants):
            reactants, products = PeroxidGroupImputer.impute_peroxide(
                reactants, products
            )
        elif PeroxidGroupImputer.check_peracid(reactants):
            reactants, products = PeroxidGroupImputer.impute_peracid(
                reactants, products
            )
        reactions_dict[rsmi_col] = symbol.join([reactants, products])
        return reactions_dict
