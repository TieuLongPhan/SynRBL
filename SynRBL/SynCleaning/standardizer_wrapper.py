from rdkit import Chem
from rdkit.Chem.MolStandardize import normalize, tautomer, charge
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem.MolStandardize import rdMolStandardize


def normalize_molecule(mol: Chem.Mol) -> Chem.Mol:
    """
    Normalize a molecule using RDKit's Normalizer.

    Args:
        mol (Chem.Mol): RDKit Mol object to be normalized.

    Returns:
        Chem.Mol: Normalized RDKit Mol object.
    """
    return normalize.Normalizer().normalize(mol)

def canonicalize_tautomer(mol: Chem.Mol) -> Chem.Mol:
    """
    Canonicalize the tautomer of a molecule using RDKit's TautomerCanonicalizer.

    Args:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with canonicalized tautomer.
    """
    return tautomer.TautomerCanonicalizer().canonicalize(mol)

def salts_remover(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove salt fragments from a molecule using RDKit's SaltRemover.

    Args:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with salts removed.
    """
    remover = SaltRemover()
    return remover.StripMol(mol)

def reionize_charges(mol: Chem.Mol) -> Chem.Mol:
    """
    Adjust molecule to its most likely ionic state using RDKit's Reionizer.

    Args:
    - mol: RDKit Mol object.

    Returns:
    - Mol object with reionized charges.
    """
    return charge.Reionizer().reionize(mol)

def uncharge_molecule(mol: Chem.Mol) -> Chem.Mol:
    """
    Neutralize a molecule by removing counter-ions using RDKit's Uncharger.

    Args:
        mol: RDKit Mol object.

    Returns:
        Neutralized Mol object.
    """
    uncharger = rdMolStandardize.Uncharger()
    return uncharger.uncharge(mol)

def assign_stereochemistry(mol: Chem.Mol, cleanIt: bool = True, force: bool = True) -> None:
    """
    Assigns stereochemistry to a molecule using RDKit's AssignStereochemistry.

    Args:
        mol: RDKit Mol object.
        cleanIt: Flag indicating whether to clean the molecule. Default is True.
        force: Flag indicating whether to force stereochemistry assignment. Default is True.

    Returns:
        None
    """
    Chem.AssignStereochemistry(mol, cleanIt=cleanIt, force=force)

def fragments_remover(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove small fragments from a molecule, keeping only the largest one.

    Args:
        mol (Chem.Mol): RDKit Mol object.

    Returns:
        Chem.Mol: Mol object with small fragments removed.
    """
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    return max(frags, default=None, key=lambda m: m.GetNumAtoms())

def remove_hydrogens_and_sanitize(mol: Chem.Mol) -> Chem.Mol:
    """
    Remove explicit hydrogens and sanitize a molecule.

    Args:
        mol (Chem.Mol): RDKit Mol object.

    Returns:
        Chem.Mol: Mol object with explicit hydrogens removed and sanitized.
    """
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    return mol


def gemdiol_standardize(mol: Chem.Mol) -> Chem.Mol:
    """
    Convert a geminal diol (two hydroxyl groups on the same carbon) to a carbonyl group in a molecule.
    This function finds the first occurrence of a geminal diol in the molecule and converts one of 
    the hydroxyl groups to a carbonyl group. The conversion is a simplification and does not necessarily
    represent a true chemical reaction pathway.

    Args:
        mol (Chem.Mol): RDKit Mol object.

    Returns:
        Chem.Mol: Mol object after conversion of the first geminal diol to a carbonyl group.
             Returns the original mol if no geminal diol is found or if the input is invalid.
    """
    

    # Define SMARTS pattern for geminal diol
    diol_pattern = Chem.MolFromSmarts('[OH][C][OH]')

    # Find the geminal diol substructure
    diol_indices = mol.GetSubstructMatches(diol_pattern)
    if not diol_indices:
        return mol  # Return the original SMILES if no geminal diol is found

    # Consider only the first geminal diol group found
    hydroxyl_1_idx, carbon_idx, hydroxyl_2_idx = diol_indices[0]

    # Create an editable molecule for modification
    emol = Chem.EditableMol(mol)

    # Remove one hydroxyl group completely
    emol.RemoveAtom(hydroxyl_2_idx)

    # Convert the remaining hydroxyl group to a carbonyl group
    emol.RemoveBond(carbon_idx, hydroxyl_1_idx)
    emol.AddBond(carbon_idx, hydroxyl_1_idx, order=Chem.rdchem.BondType.DOUBLE)

    # Get the modified molecule and sanitize it
    modified_mol: Mol = emol.GetMol()
    Chem.SanitizeMol(modified_mol)

    return modified_mol


