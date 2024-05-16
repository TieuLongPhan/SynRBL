import pandas as pd

from rdkit import Chem
from joblib import Parallel, delayed
from typing import List
import multiprocessing
import multiprocessing.pool
from synrbl.SynMCSImputer.MissingGraph.find_missing_graphs import FindMissingGraphs
from synrbl.SynMCSImputer.MissingGraph.uncertainty_graph import GraphMissingUncertainty


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

    def process_single_pair(reactant_mol, mcs_mol, job_timeout=2):
        output = {
            "smiles": [],
            "boundary_atoms_products": [],
            "nearest_neighbor_products": [],
            "issue": "Find Missing Graph terminated by timeout",
        }
        try:
            pool = multiprocessing.pool.ThreadPool(1)
            async_result = pool.apply_async(
                FindMissingGraphs.find_missing_parts_pairs,
                (
                    reactant_mol,
                    mcs_mol,
                ),
            )
            result = async_result.get(job_timeout)
            pool.terminate()  # Terminate the pool to release resources
            output["smiles"] = [
                Chem.MolToSmiles(mol) if mol is not None else None for mol in result[0]
            ]
            output["boundary_atoms_products"] = result[1]
            output["nearest_neighbor_products"] = result[2]
        except multiprocessing.TimeoutError:
            output["issue"] = "Find Missing Graph terminated by timeout"
        except Exception as e:
            output["issue"] = "Find MCS failed with exception: {}".format(e)
        return output

    results = Parallel(n_jobs=n_jobs, verbose=0)(
        delayed(process_single_pair)(reactant_mol, mcs_mol)
        for reactant_mol, mcs_mol in zip(sorted_reactants_mol_list, mcs_mol_list)
    )
    return results


def find_graph_dict(mcs_dict, n_jobs: int = 4):
    """
    Function to find missing graphs for a given MCS dictionary.
    """
    if len(mcs_dict) == 0:
        return []

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
