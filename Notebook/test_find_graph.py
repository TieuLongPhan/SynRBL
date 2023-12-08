import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
from pathlib import Path
root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCS.mcs_utils import smiles_to_mol_parallel
from SynRBL.SynMCS.find_missing_graphs import FindMissingGraphs


def main(save: bool = False, save_path: str = f'{root_dir}/Data/MCS/final_graph.json.gz'):
    """
    Main function to process data and find missing graphs.

    Args:
        save (bool, optional): Flag to indicate if the results should be saved. Defaults to False.
        save_path (str, optional): File path to save the results. Defaults to f'{root_dir}/Data/MCS/final_graph.json.gz'.

    Returns:
        None
    """
    mcs_dict = load_database(f'{root_dir}/Data/MCS/Intersection_MCS.json.gz')
    msc_df = pd.DataFrame(mcs_dict)

    mcs_results = msc_df['mcs_results'].to_list()
    sorted_reactants = msc_df['sorted_reactants'].to_list()

    mcs_mol_list = smiles_to_mol_parallel(mcs_results)
    sorted_reactants_mol_list = smiles_to_mol_parallel(sorted_reactants)

    find_graph = FindMissingGraphs()
    missing_results = find_graph.find_single_graph_parallel(mcs_mol_list, sorted_reactants_mol_list, n_jobs=4, use_findMCS=True)

    if save:
        save_database(missing_results, save_path)

# Execute main function
if __name__ == "__main__":
    main(save=True)
