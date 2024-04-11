from rdkit import Chem


class FunctionalGroupChecker:
    # 1. peroxid group
    @staticmethod
    def check_peroxide(smiles: str) -> bool:
        """
        Check for the presence of a peroxide substructure in a molecule.
        """
        peroxide_pattern = Chem.MolFromSmarts("OO")
        mol = Chem.MolFromSmiles(smiles)
        return (
            mol.HasSubstructMatch(peroxide_pattern)
            if mol and not FunctionalGroupChecker.check_peracid(smiles)
            else False
        )

    @staticmethod
    def check_peracid(smiles: str) -> bool:
        """
        Check for the presence of a peracid substructure in a molecule.
        """
        peracid_pattern = Chem.MolFromSmarts("C(OO)=O")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(peracid_pattern) if mol else False

    # 2. Alcohol group
    @staticmethod
    def check_alcohol(smiles: str) -> bool:
        """
        Check for the presence of an alcohol functional group in a molecule.
        """
        alcohol_pattern = Chem.MolFromSmarts("CO")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(alcohol_pattern) if mol else False

    @staticmethod
    def check_enol(smiles: str) -> bool:
        """
        Check for the presence of an enol functional group in a molecule.
        """
        enol_pattern = Chem.MolFromSmarts("C=C(O)")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(enol_pattern) if mol else False

    @staticmethod
    def check_phenol(smiles: str) -> bool:
        """
        Check for the presence of a phenol functional group in a molecule.
        """
        phenol_pattern = Chem.MolFromSmarts("[c]O")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(phenol_pattern) if mol else False

    @staticmethod
    def check_vicinal_diol(smiles: str) -> bool:
        """
        Check for the presence of a vicinal diol functional group in a molecule.
        """
        vicinal_diol_pattern = Chem.MolFromSmarts("OCO")
        mol = Chem.MolFromSmiles(smiles)
        return (
            mol.HasSubstructMatch(vicinal_diol_pattern)
            if mol
            and not FunctionalGroupChecker.check_hemiacetal(smiles)
            and not FunctionalGroupChecker.check_carbonate(smiles)
            and not FunctionalGroupChecker.check_carboxylic_acid(smiles)
            and not FunctionalGroupChecker.check_ester(smiles)
            else False
        )

    @staticmethod
    def check_gem_diol(smiles: str) -> bool:
        """
        Check for the presence of a gem diol functional group in a molecule.
        """
        gem_diol_pattern = Chem.MolFromSmarts("OCCO")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(gem_diol_pattern) if mol else False

    @staticmethod
    def check_ether(smiles: str) -> bool:
        """
        Check for the presence of an ether functional group in a molecule.
        """
        ether_pattern = Chem.MolFromSmarts("COC")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(ether_pattern) if mol else False

    # 3. Carbonyl group
    @staticmethod
    def check_aldehyde(smiles: str) -> bool:
        """
        Check for the presence of an aldehyde functional group in a molecule.
        """
        aldehyde_pattern = Chem.MolFromSmarts("[CX3H1](=O)[#6]")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(aldehyde_pattern) if mol else False

    @staticmethod
    def check_ketone(smiles: str) -> bool:
        """
        Check for the presence of a ketone functional group in a molecule.
        """
        ketone_pattern = Chem.MolFromSmarts("[#6][CX3](=O)[#6]")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(ketone_pattern) if mol else False

    @staticmethod
    def check_acetal(smiles: str) -> bool:
        """
        Check for the presence of an acetal functional group in a molecule.
        """
        acetal_pattern = Chem.MolFromSmarts("[CX4][OX2][CX4]")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(acetal_pattern) if mol else False

    @staticmethod
    def check_hemiacetal(smiles: str) -> bool:
        """
        Check for the presence of a hemiacetal functional group in a molecule.
        """
        hemiacetal_pattern = Chem.MolFromSmarts("COCO")
        mol = Chem.MolFromSmiles(smiles)
        return (
            mol.HasSubstructMatch(hemiacetal_pattern)
            if mol
            and not FunctionalGroupChecker.check_carbonate(smiles)
            and not FunctionalGroupChecker.check_carboxylic_acid(smiles)
            and not FunctionalGroupChecker.check_ester(smiles)
            else False
        )

    # 4. Carboxylic group
    @staticmethod
    def check_carboxylic_acid(smiles: str) -> bool:
        """
        Check for the presence of a carboxylic acid functional group in a molecule.
        """
        carboxylic_acid_pattern = Chem.MolFromSmarts("C(=O)O")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(carboxylic_acid_pattern) if mol else False

    @staticmethod
    def check_ester(smiles: str) -> bool:
        """
        Check for the presence of an ester functional group in a molecule.
        """
        ester_pattern = Chem.MolFromSmarts("C(=O)OC")
        mol = Chem.MolFromSmiles(smiles)
        return (
            mol.HasSubstructMatch(ester_pattern)
            if mol and not FunctionalGroupChecker.check_carbonate(smiles)
            else False
        )

    @staticmethod
    def check_amide(smiles: str) -> bool:
        """
        Check for the presence of an amide functional group in a molecule.
        """
        amide_pattern = Chem.MolFromSmarts("NC=O")
        mol = Chem.MolFromSmiles(smiles)
        return (
            mol.HasSubstructMatch(amide_pattern)
            if mol and not FunctionalGroupChecker.check_urea(smiles)
            else False
        )

    @staticmethod
    def check_cyanide(smiles: str) -> bool:
        """
        Check for the presence of a cyanide functional group in a molecule.
        """
        cyanide_pattern = Chem.MolFromSmarts("[C-]#[N+]")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(cyanide_pattern) if mol else False

    @staticmethod
    def check_urea(smiles: str) -> bool:
        """
        Check for the presence of a urea functional group in a molecule.
        """
        urea_pattern = Chem.MolFromSmarts("NC(=O)N")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(urea_pattern) if mol else False

    @staticmethod
    def check_carbonate(smiles: str) -> bool:
        """
        Check for the presence of a carbonate functional group in a molecule.
        """
        carbonate_pattern = Chem.MolFromSmarts("OC(=O)O")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(carbonate_pattern) if mol else False

    # 5. Amine group
    @staticmethod
    def check_amine(smiles: str) -> bool:
        """
        Check for the presence of an amine functional group in a molecule.
        """
        amine_pattern = Chem.MolFromSmarts("CN")
        mol = Chem.MolFromSmiles(smiles)
        return (
            mol.HasSubstructMatch(amine_pattern)
            if mol and not FunctionalGroupChecker.check_amide(smiles)
            else False
        )

    @staticmethod
    def check_nitro(smiles: str) -> bool:
        """
        Check for the presence of a nitro functional group in a molecule.
        """
        nitro_pattern = Chem.MolFromSmarts("[N+](=O)[O-]")
        mol = Chem.MolFromSmiles(smiles)
        return mol.HasSubstructMatch(nitro_pattern) if mol else False
