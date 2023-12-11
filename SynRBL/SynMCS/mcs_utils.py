from rdkit import Chem
from joblib import Parallel, delayed
from typing import List

def convert_smiles_to_mols(
    smiles_list: List[str]
    ) -> List[Chem.rdchem.Mol]:
    """
    Convert a list of SMILES strings to a list of RDKit molecule objects.

    Args:
        smiles_list: A list containing SMILES strings.

    Returns:
        A list containing RDKit molecule objects.
    """
    mol_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol:
            try:
                Chem.SanitizeMol(mol)
            except:
                pass
            
        mol_list.append(mol)
    return mol_list

def smiles_to_mol_parallel(
    smiles_lists: List[List[str]], 
    n_jobs: int = 4
    ) -> List[List[Chem.Mol]]:
    """
    Convert a list of lists of SMILES strings to a list of lists of RDKit molecule objects using parallel processing.

    :param smiles_lists: List of lists containing SMILES strings.
    :param n_jobs: The number of jobs to run in parallel. -1 means using all processors.
    :return: A list of lists containing RDKit molecule objects.
    """
    mol_lists = Parallel(n_jobs=n_jobs)(delayed(Chem.MolFromSmiles)(smiles_list) for smiles_list in smiles_lists)
    return mol_lists
