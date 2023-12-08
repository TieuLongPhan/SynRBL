from rdkit import Chem
from joblib import Parallel, delayed

def convert_smiles_to_mols(smiles_list):
    """
    Convert a list of SMILES strings to a list of RDKit molecule objects.

    :param smiles_list: List containing SMILES strings.
    :return: A list containing RDKit molecule objects.
    """
    mol_list = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol:
            try:
                Chem.SanitizeMol(mol)
                #mol = Chem.RemoveHs(mol)
            except:
                pass
            
        mol_list.append(mol)
    return mol_list

def smiles_to_mol_parallel(smiles_lists, n_jobs=-1):
    """
    Convert a list of lists of SMILES strings to a list of lists of RDKit molecule objects using parallel processing.

    :param smiles_lists: List of lists containing SMILES strings.
    :param n_jobs: The number of jobs to run in parallel. -1 means using all processors.
    :return: A list of lists containing RDKit molecule objects.
    """
    mol_lists = Parallel(n_jobs=n_jobs)(delayed(convert_smiles_to_mols)(smiles_list) for smiles_list in smiles_lists)
    return mol_lists
