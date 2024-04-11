import pandas as pd

from rdkit import Chem
from rdkit.rdBase import BlockLogs
from joblib import Parallel, delayed
from typing import List
from synrbl.SynMCSImputer.MissingGraph.find_missing_graphs import FindMissingGraphs
from synrbl.SynMCSImputer.MissingGraph.uncertainty_graph import GraphMissingUncertainty


def find_single_graph(mcs_mol_list, sorted_reactants_mol_list):
    """
    Find missing parts, boundary atoms, and nearest neighbors for a list of
    reactant molecules using a corresponding list of MCS (Maximum Common
    Substructure) molecules.

    Parameters:
    - mcs_mol_list (list of rdkit.Chem.Mol): List of RDKit molecule objects
        representing the MCS, corresponding to each molecule in
        sorted_reactants_mol_list.
    - sorted_reactants_mol_list (list of rdkit.Chem.Mol): The list of RDKit
        molecule objects to analyze.

    Returns:
    - Dictionary containing:
    - 'smiles' (list of list of str): SMILES representations of the missing
        parts for each molecule.
    - 'boundary_atoms_products' (list of list of dict): Lists of boundary atoms
        for each molecule.
    - 'nearest_neighbor_products' (list of list of dict): Lists of nearest
        neighbors for each molecule.
    - 'issue' (list): Any issues encountered during processing.
    """
    missing_results = {
        "smiles": [],
        "boundary_atoms_products": [],
        "nearest_neighbor_products": [],
        "issue": [],
    }
    for i in zip(sorted_reactants_mol_list, mcs_mol_list):
        try:
            (
                mols,
                boundary_atoms_products,
                nearest_neighbor_products,
            ) = FindMissingGraphs.find_missing_parts_pairs(i[0], i[1])
            missing_results["smiles"].append([Chem.MolToSmiles(mol) for mol in mols])
            missing_results["boundary_atoms_products"].append(boundary_atoms_products)
            missing_results["nearest_neighbor_products"].append(
                nearest_neighbor_products
            )
            missing_results["issue"].append([])
        except Exception as e:
            missing_results["smiles"].append([])
            missing_results["boundary_atoms_products"].append([])
            missing_results["nearest_neighbor_products"].append([])
            missing_results["issue"].append(
                "FindMissingGraphs.find_missing_parts() failed:" + str(e)
            )
    return missing_results


def find_single_graph_parallel(mcs_mol_list, sorted_reactants_mol_list, n_jobs=4):
    """
    Find missing parts, boundary atoms, and nearest neighbors for a list of
    reactant molecules using a corresponding list of MCS (Maximum Common
    Substructure) molecules in parallel.

    Parameters:
    - mcs_mol_list (list of rdkit.Chem.Mol): List of RDKit molecule objects
        representing the MCS, corresponding to each molecule in
        sorted_reactants_mol_list.
    - sorted_reactants_mol_list (list of rdkit.Chem.Mol): The list of RDKit
        molecule objects to analyze.
    - n_jobs (int): The number of parallel jobs to run. Default is -1, which
        uses all available CPU cores.

    Returns:
    - List of dictionaries, where each dictionary contains:
    - 'smiles' (list of str): SMILES representations of the missing parts for
        each molecule.
    - 'boundary_atoms_products' (list of dict): Lists of boundary atoms for
        each molecule.
    - 'nearest_neighbor_products' (list of dict): Lists of nearest neighbors
        for each molecule.
    - 'issue' (str): Any issues encountered during processing.
    """

    def process_single_pair(reactant_mol, mcs_mol):
        try:
            block = BlockLogs()
            (
                mols,
                boundary_atoms_products,
                nearest_neighbor_products,
            ) = FindMissingGraphs.find_missing_parts_pairs(reactant_mol, mcs_mol)
            del block
            return {
                "smiles": [
                    Chem.MolToSmiles(mol) if mol is not None else None for mol in mols
                ],
                "boundary_atoms_products": boundary_atoms_products,
                "nearest_neighbor_products": nearest_neighbor_products,
                "issue": "",
            }
        except Exception as e:
            return {
                "smiles": [],
                "boundary_atoms_products": [],
                "nearest_neighbor_products": [],
                "issue": str(e),
            }

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_single_pair)(reactant_mol, mcs_mol)
        for reactant_mol, mcs_mol in zip(sorted_reactants_mol_list, mcs_mol_list)
    )
    return results


def find_graph_dict(mcs_dict, n_jobs: int = 4):
    """
    Function to find missing graphs for a given MCS dictionary.
    """
    msc_df = pd.DataFrame(mcs_dict)

    mcs_results = msc_df["mcs_results"].to_list()
    sorted_reactants = msc_df["sorted_reactants"].to_list()

    mcs_mol_list = smiles_to_mol_parallel(mcs_results, useSmiles=False)
    sorted_reactants_mol_list = smiles_to_mol_parallel(sorted_reactants, useSmiles=True)

    missing_results = find_single_graph_parallel(
        mcs_mol_list, sorted_reactants_mol_list, n_jobs=n_jobs
    )

    missing_results = GraphMissingUncertainty(missing_results, threshold=2).fit()

    return missing_results


def convert_smiles_to_mols(
    smiles_list: List[str], useSmiles: bool = True
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
        try:
            if useSmiles:
                mol = Chem.MolFromSmiles(smiles, sanitize=True)
            else:
                mol = Chem.MolFromSmarts(smiles)
            mol_list.append(mol)
        except Exception:
            mol_list.append(None)
    return mol_list


def smiles_to_mol_parallel(
    smiles_lists: List[List[str]], n_jobs: int = 4, useSmiles: bool = True
) -> List[List[Chem.Mol]]:
    """
    Convert a list of lists of SMILES strings to a list of lists of RDKit
    molecule objects using parallel processing.

    :param smiles_lists: List of lists containing SMILES strings.
    :param n_jobs: The number of jobs to run in parallel. -1 means using
        all processors.
    :return: A list of lists containing RDKit molecule objects.
    """
    mol_lists = Parallel(n_jobs=n_jobs)(
        delayed(convert_smiles_to_mols)(smiles_list, useSmiles)
        for smiles_list in smiles_lists
    )
    return mol_lists
