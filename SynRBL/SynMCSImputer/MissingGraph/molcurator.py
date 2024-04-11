from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import AllChem


class MoleculeCurator:
    """
    A class for curating molecules, which includes adding hydrogens to radicals,
    standardizing diazo charges, and manually kekulizing molecules.

    Methods:
    - add_hydrogens_to_radicals: Adds hydrogens to radical sites in a molecule.
    - standardize_diazo_charge: Converts a diazo compound with charged atoms to
        its neutral form.
    - manual_kekulize: Manually kekulizes a molecule to ensure valid structure.
    """

    @staticmethod
    def add_hydrogens_to_radicals(mol: Chem.Mol) -> Chem.Mol:
        """
        Add hydrogen atoms to radical sites in a molecule.

        Args:
        - mol (Chem.Mol): RDKit molecule object.

        Returns:
        - Chem.Mol: The modified molecule with added hydrogens.
        """
        mol_with_h = Chem.RWMol(mol)
        mol_with_h = rdmolops.AddHs(mol_with_h)

        for atom in mol_with_h.GetAtoms():
            num_radical_electrons = atom.GetNumRadicalElectrons()
            if num_radical_electrons > 0:
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + num_radical_electrons)
                atom.SetNumRadicalElectrons(0)

        curate_mol = Chem.RemoveHs(mol_with_h)
        return curate_mol

    @staticmethod
    def standardize_diazo_charge(mol: Chem.Mol) -> Chem.Mol:
        """
        Convert a diazo compound with charged atoms to its neutral form.

        Args:
        - mol (Chem.Mol): A RDKit molecule object representing a diazo compound.

        Returns:
        - Chem.Mol: The neutralized molecule, or the original molecule if the
            reaction doesn't occur.
        """
        neutral_mol = AllChem.ReactionFromSmarts(
            "[N-]=[NH2+]>>[N:1]#[N:2]"
        ).RunReactants((mol,))
        return neutral_mol[0][0] if neutral_mol else mol

    @staticmethod
    def manual_kekulize(smiles: str) -> Chem.Mol:
        """
        Manually kekulizes a molecule represented by a SMILES string.

        Args:
        - smiles (str): A SMILES string representing the molecule.

        Returns:
        - Chem.Mol: The kekulized molecule, or None if it's not possible.
        """

        # Function to add a hydrogen to a specific atom in a molecule
        def add_hydrogen(mol, atom_index):
            edited_mol = Chem.RWMol(mol)
            atom = edited_mol.GetAtomWithIdx(atom_index)
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
            return edited_mol.GetMol()

        # Process a single component of the molecule
        def process_component(component):
            for atom_index in range(component.GetNumAtoms()):
                try:
                    test_mol = add_hydrogen(component, atom_index)
                    rdmolops.SanitizeMol(test_mol)
                    return test_mol
                except Exception:
                    continue
            return None

        components_smiles = smiles.split(".")
        valid_components = []

        for comp_smiles in components_smiles:
            comp = Chem.MolFromSmiles(comp_smiles, sanitize=False)
            valid_mol = process_component(comp)
            if valid_mol:
                valid_components.append(valid_mol)

        if valid_components:
            combined_mol = valid_components[0]
            for i in range(1, len(valid_components)):
                combined_mol = Chem.CombineMols(combined_mol, valid_components[i])
            return combined_mol
        else:
            return None
