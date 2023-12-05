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
    def find_maximum_common_substructure(mol1, mol2, ringMatchesRingOnly=True):
        """
        Find the maximum common substructure (MCS) between two molecules.

        Parameters:
        - mol1, mol2: rdkit.Chem.Mol
            The RDKit molecule objects to compare.

        Returns:
        - rdkit.Chem.Mol or None
            The RDKit molecule object representing the MCS, or None if MCS search was canceled.
        """
        mcs_result = rdFMCS.FindMCS([mol1, mol2], ringMatchesRingOnly=ringMatchesRingOnly)
        if mcs_result.canceled:
            return None
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        return mcs_mol

    @staticmethod
    def find_mcs_for_each_fragment(reactant_mol_list, product_mol_list, ringMatchesRingOnly=True, **kwargs):
        """
        Find the MCS for each pair of reactant and product fragments.

        Parameters:
        - reactant_mol_list, product_mol_list: list of rdkit.Chem.Mol
            Lists of RDKit molecule objects for reactants and products.

        Returns:
        - list of rdkit.Chem.Mol
            List of RDKit molecule objects representing the MCS for each reactant-product pair.
        """
        mcs_list = []
        for reactant in reactant_mol_list:
            for product in product_mol_list:
                mcs_result = rdFMCS.FindMCS([reactant, product], ringMatchesRingOnly=ringMatchesRingOnly, **kwargs)
                if mcs_result.canceled:
                    continue
                mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                mcs_list.append(mcs_mol)
        return mcs_list

    @staticmethod
    def find_missing_parts(mol, mcs_list):
        """
        Find the missing parts of a molecule relative to a list of MCS.

        Parameters:
        - mol: rdkit.Chem.Mol
            The RDKit molecule object to analyze.
        - mcs_list: list of rdkit.Chem.Mol
            List of RDKit molecule objects representing MCS.

        Returns:
        - rdkit.Chem.Mol or None
            The RDKit molecule object representing missing parts, or None if no missing parts.
        """
        atoms_to_remove = set()
        for mcs_mol in mcs_list:
            match = mol.GetSubstructMatch(mcs_mol)
            atoms_to_remove.update(match)
        if len(atoms_to_remove) == mol.GetNumAtoms():
            return None
        missing_part = Chem.RWMol(mol)
        for idx in sorted(atoms_to_remove, reverse=True):
            missing_part.RemoveAtom(idx)
        Chem.SanitizeMol(missing_part)
        return missing_part if missing_part.GetNumAtoms() > 0 else None
    
    @staticmethod
    def add_hydrogens_to_radicals(mol):
        """
        Add hydrogen atoms to radical sites in a molecule.

        Parameters:
        - mol: rdkit.Chem.Mol
            RDKit molecule object.

        Returns:
        - rdkit.Chem.Mol
            The modified molecule with added hydrogens.
        """
        if mol:
            # Create a copy of the molecule
            mol_with_h = Chem.RWMol(mol)

            # Add explicit hydrogens (not necessary if they are already present in the input molecule)
            mol_with_h = rdmolops.AddHs(mol_with_h)

            # Find and process radical atoms
            for atom in mol_with_h.GetAtoms():
                num_radical_electrons = atom.GetNumRadicalElectrons()
                if num_radical_electrons > 0:
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + num_radical_electrons)
                    atom.SetNumRadicalElectrons(0)
            curate_mol = Chem.RemoveHs(mol_with_h)
            # Return the molecule with added hydrogens
            return curate_mol

    @staticmethod
    def fit(reaction_dict, curate_radicals=False, ringMatchesRingOnly=True,**kwargs):
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
        reactant_smiles, product_smiles = MCSMissingGraphAnalyzer.get_smiles(reaction_dict)
        reactant_mol_list = [MCSMissingGraphAnalyzer.convert_smiles_to_molecule(smiles) for smiles in reactant_smiles.split('.')]
        product_mol_list = [MCSMissingGraphAnalyzer.convert_smiles_to_molecule(smiles) for smiles in product_smiles.split('.')]
        mcs_list = MCSMissingGraphAnalyzer.find_mcs_for_each_fragment(reactant_mol_list, product_mol_list, ringMatchesRingOnly=ringMatchesRingOnly, **kwargs)
        missing_parts_reactant = [MCSMissingGraphAnalyzer.find_missing_parts(frag, mcs_list) for frag in reactant_mol_list]
        missing_part_product = [MCSMissingGraphAnalyzer.find_missing_parts(frag, mcs_list) for frag in product_mol_list]
        if curate_radicals:
            missing_parts_reactant = [MCSMissingGraphAnalyzer.add_hydrogens_to_radicals(frag) for frag in missing_parts_reactant]
            missing_part_product = [MCSMissingGraphAnalyzer.add_hydrogens_to_radicals(frag) for frag in missing_part_product]

        return mcs_list, missing_parts_reactant, missing_part_product, reactant_mol_list, product_mol_list
