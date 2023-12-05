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
    def IterativeMCSReactionPairs(reactant_mol_list, product_mol, params=None):
        """
        Find the MCS for each reactant fragment with the product, updating the product after each step.

        Parameters:
        - reactant_mol_list: list of rdkit.Chem.Mol
            List of RDKit molecule objects for reactants.
        - product_mol: rdkit.Chem.Mol
            RDKit molecule object for the product.

        Returns:
        - list of rdkit.Chem.Mol
            List of RDKit molecule objects representing the MCS for each reactant-product pair.
        """
        mcs_list = []
        current_product = product_mol

        # Sort reactant molecules based on the number of atoms (descending order)
        sorted_reactants = sorted(reactant_mol_list, key=lambda x: x.GetNumAtoms(), reverse=True)

        for reactant in sorted_reactants:
            mcs_result = rdFMCS.FindMCS([reactant, current_product], params)
            if mcs_result.canceled:
                continue

            mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
            mcs_list.append(mcs_mol)

            # Update the product by removing the MCS substructure
            current_product = Chem.DeleteSubstructs(Chem.RWMol(current_product), mcs_mol)
            current_product = Chem.RemoveHs(current_product)
            try:
                Chem.SanitizeMol(current_product)
            except:
                pass

        return mcs_list, sorted_reactants

    
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
    def fit(reaction_dict, params=None):
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
        product_mol = MCSMissingGraphAnalyzer.convert_smiles_to_molecule(product_smiles)

        mcs_list, sorted_reactants = MCSMissingGraphAnalyzer.IterativeMCSReactionPairs(reactant_mol_list, product_mol,  params)

        return mcs_list , sorted_reactants, product_mol