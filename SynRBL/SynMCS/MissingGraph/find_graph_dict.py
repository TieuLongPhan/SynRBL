from rdkit import Chem
from joblib import Parallel, delayed
from typing import List
from SynRBL.SynMCS.MissingGraph.find_missing_graphs import FindMissingGraphs
from SynRBL.SynMCS.MissingGraph.uncertainty_graph import GraphMissingUncertainty
from SynRBL.rsmi_utils import load_database, save_database
import pandas as pd

def find_single_graph(mcs_mol_list, sorted_reactants_mol_list, use_findMCS=True):
    """
    Find missing parts, boundary atoms, and nearest neighbors for a list of reactant molecules
    using a corresponding list of MCS (Maximum Common Substructure) molecules.

    Parameters:
    - mcs_mol_list (list of rdkit.Chem.Mol): List of RDKit molecule objects representing the MCS,
    corresponding to each molecule in sorted_reactants_mol_list.
    - sorted_reactants_mol_list (list of rdkit.Chem.Mol): The list of RDKit molecule objects to analyze.

    Returns:
    - Dictionary containing:
    - 'smiles' (list of list of str): SMILES representations of the missing parts for each molecule.
    - 'boundary_atoms_products' (list of list of dict): Lists of boundary atoms for each molecule.
    - 'nearest_neighbor_products' (list of list of dict): Lists of nearest neighbors for each molecule.
    - 'issue' (list): Any issues encountered during processing.
    """
    missing_results = {'smiles': [], 'boundary_atoms_products': [], 'nearest_neighbor_products': [], 'issue': []}
    for i in zip(sorted_reactants_mol_list, mcs_mol_list):
        try:
            mols, boundary_atoms_products, nearest_neighbor_products = FindMissingGraphs.find_missing_parts_pairs(i[0], i[1], use_findMCS=use_findMCS)
            missing_results['smiles'].append([Chem.MolToSmiles(mol) for mol in mols])
            missing_results['boundary_atoms_products'].append(boundary_atoms_products)
            missing_results['nearest_neighbor_products'].append(nearest_neighbor_products)
            missing_results['issue'].append([])
        except Exception as e:
            missing_results['smiles'].append([])
            missing_results['boundary_atoms_products'].append([])
            missing_results['nearest_neighbor_products'].append([])
            missing_results['issue'].append(str(e))
    return missing_results


def find_single_graph_parallel(mcs_mol_list, sorted_reactants_mol_list, n_jobs=-1, use_findMCS=True):
    """
    Find missing parts, boundary atoms, and nearest neighbors for a list of reactant molecules
    using a corresponding list of MCS (Maximum Common Substructure) molecules in parallel.

    Parameters:
    - mcs_mol_list (list of rdkit.Chem.Mol): List of RDKit molecule objects representing the MCS,
    corresponding to each molecule in sorted_reactants_mol_list.
    - sorted_reactants_mol_list (list of rdkit.Chem.Mol): The list of RDKit molecule objects to analyze.
    - n_jobs (int): The number of parallel jobs to run. Default is -1, which uses all available CPU cores.

    Returns:
    - List of dictionaries, where each dictionary contains:
    - 'smiles' (list of str): SMILES representations of the missing parts for each molecule.
    - 'boundary_atoms_products' (list of dict): Lists of boundary atoms for each molecule.
    - 'nearest_neighbor_products' (list of dict): Lists of nearest neighbors for each molecule.
    - 'issue' (str): Any issues encountered during processing.
    """
    def process_single_pair(reactant_mol, mcs_mol, use_findMCS=True):
        try:
            mols, boundary_atoms_products, nearest_neighbor_products = FindMissingGraphs.find_missing_parts_pairs(reactant_mol, mcs_mol, use_findMCS=use_findMCS)
            return {
                'smiles': [Chem.MolToSmiles(mol) if mol is not None else None for mol in mols],
                'boundary_atoms_products': boundary_atoms_products,
                'nearest_neighbor_products': nearest_neighbor_products,
                'issue': ''
            }
        except Exception as e:
            return {
                'smiles': [],
                'boundary_atoms_products': [],
                'nearest_neighbor_products': [],
                'issue': str(e)
            }

    results = Parallel(n_jobs=n_jobs)(delayed(process_single_pair)(reactant_mol, mcs_mol, use_findMCS=use_findMCS) for reactant_mol, mcs_mol in zip(sorted_reactants_mol_list, mcs_mol_list))
    return results

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

    #find_graph = FindMissingGraphs()
    missing_results = find_single_graph_parallel(mcs_mol_list, sorted_reactants_mol_list, n_jobs=n_jobs, use_findMCS=use_findMCS)
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
