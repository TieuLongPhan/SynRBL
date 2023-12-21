from rdkit import Chem
from joblib import Parallel, delayed
from typing import List
from SynRBL.SynMCS.MissingGraph.find_missing_graphs import FindMissingGraphs
from SynRBL.SynMCS.MissingGraph.uncertainty_graph import GraphMissingUncertainty
from SynRBL.rsmi_utils import load_database, save_database
import pandas as pd


def find_graph_dict(msc_dict_path: str,  save_path: str, save: bool =True,
                    n_jobs: int=4, use_findMCS: bool=False):
    """
    Function to find missing graphs for a given MCS dictionary.
    """
    mcs_dict = load_database(msc_dict_path)

    msc_df = pd.DataFrame(mcs_dict)

    mcs_results = msc_df['mcs_results'].to_list()
    sorted_reactants = msc_df['sorted_reactants'].to_list()

    mcs_mol_list = smiles_to_mol_parallel(mcs_results, useSmiles=False)
    sorted_reactants_mol_list = smiles_to_mol_parallel(sorted_reactants, useSmiles=True)

    find_graph = FindMissingGraphs()
    missing_results = find_graph.find_single_graph_parallel(mcs_mol_list, sorted_reactants_mol_list, n_jobs=n_jobs, use_findMCS=use_findMCS)
    missing_results = GraphMissingUncertainty(missing_results, threshold=2).fit()
    uncertainty_data = len(pd.DataFrame(missing_results)) - pd.DataFrame(missing_results)['Certainty'].sum()
    print('Uncertainty Data:', uncertainty_data)
    if save:
        save_database(missing_results, save_path)
    
    return missing_results

def convert_smiles_to_mols(
    smiles_list: List[str],
    useSmiles: bool = True
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
        if useSmiles:
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
        else:
            mol = Chem.MolFromSmarts(smiles)
        mol_list.append(mol)
    return mol_list

def smiles_to_mol_parallel(
    smiles_lists: List[List[str]], 
    n_jobs: int = 4,
    useSmiles: bool = True
    ) -> List[List[Chem.Mol]]:
    """
    Convert a list of lists of SMILES strings to a list of lists of RDKit molecule objects using parallel processing.

    :param smiles_lists: List of lists containing SMILES strings.
    :param n_jobs: The number of jobs to run in parallel. -1 means using all processors.
    :return: A list of lists containing RDKit molecule objects.
    """
    mol_lists = Parallel(n_jobs=n_jobs)(delayed(convert_smiles_to_mols)(smiles_list, useSmiles) for smiles_list in smiles_lists)
    return mol_lists
