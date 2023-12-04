from rdkit import Chem
from rdkit.Chem.MolStandardize import normalize, tautomer, charge
from rdkit.Chem.SaltRemover import SaltRemover
from rdkit.Chem import Draw, MolStandardize, rdDepictor
from rdkit.Chem.MolStandardize import rdMolStandardize
def normalize_molecule(mol):
    """
    Normalize a molecule using RDKit's Normalizer.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object to be normalized.

    Returns:
    - Chem.Mol: Normalized RDKit Mol object.
    """
    return normalize.Normalizer().normalize(mol)

def canonicalize_tautomer(mol):
    """
    Canonicalize the tautomer of a molecule using RDKit's TautomerCanonicalizer.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with canonicalized tautomer.
    """
    return tautomer.TautomerCanonicalizer().canonicalize(mol)

def salts_remover(mol):
    """
    Remove salt fragments from a molecule using RDKit's SaltRemover.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with salts removed.
    """
    return SaltRemover().StripMol(mol)

def reionize_charges(mol):
    """
    Adjust molecule to its most likely ionic state using RDKit's Reionizer.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with reionized charges.
    """
    return charge.Reionizer().reionize(mol)

def uncharge_molecule(mol):
    """
    Neutralize a molecule by removing counter-ions using RDKit's Uncharger.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Neutralized Mol object.
    """
    return rdMolStandardize.Uncharger().uncharge(mol)

def assign_stereochemistry(mol, cleanIt=True, force=True):
    """
    Assign stereochemistry to a molecule using RDKit's AssignStereochemistry.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.
    - cleanIt (bool, optional): Clean the molecule. Default is True.
    - force (bool, optional): Force stereochemistry assignment. Default is True.

    Returns:
    - None
    """
    Chem.AssignStereochemistry(mol, cleanIt=cleanIt, force=force)

def fragmets_remover(mol):
    """
    Remove small fragments from a molecule, keeping only the largest one.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with small fragments removed.
    """
    return max(Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True), default=None, key=lambda m: m.GetNumAtoms())

def remove_hydrogens_and_sanitize(mol):
    """
    Remove explicit hydrogens and sanitize a molecule.

    Parameters:
    - mol (Chem.Mol): RDKit Mol object.

    Returns:
    - Chem.Mol: Mol object with explicit hydrogens removed and sanitized.
    """
    mol = Chem.RemoveHs(mol)
    Chem.SanitizeMol(mol)
    return mol
