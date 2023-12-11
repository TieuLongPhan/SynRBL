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


def find_graph_dict(msc_dict_path: str,  save_path: str, save: bool =True,
                    n_jobs: int=4, use_findMCS: bool=True):
    """
    Function to find missing graphs for a given MCS dictionary.
    """
    mcs_dict = load_database(msc_dict_path)

    msc_df = pd.DataFrame(mcs_dict)

    mcs_results = msc_df['mcs_results'].to_list()
    sorted_reactants = msc_df['sorted_reactants'].to_list()

    mcs_mol_list = smiles_to_mol_parallel(mcs_results)
    sorted_reactants_mol_list = smiles_to_mol_parallel(sorted_reactants)

    find_graph = FindMissingGraphs()
    missing_results = find_graph.find_single_graph_parallel(mcs_mol_list, sorted_reactants_mol_list, n_jobs=n_jobs, use_findMCS=use_findMCS)
    missing_final = pd.DataFrame(missing_results)

    missing_final = missing_final.map(lambda x: np.nan if x == '' else x)
    non_pass_df = missing_final.iloc[missing_final.dropna(subset=['issue']).index,:].to_dict(orient='records')
    print('Bug:', len(non_pass_df))
    if save:
        save_database(missing_results, save_path)
    
    return missing_results

def main():

    missing_results_3_macth = find_graph_dict(msc_dict_path=f'{root_dir}/Data/MCS/Intersection_MCS_3+_matching_ensemble.json.gz',
                save_path=f'{root_dir}/Data/MCS/Final_Graph_macth_3+.json.gz')
    
    #missing_results_largest = find_graph_dict(msc_dict_path=f'{root_dir}/Data/MCS/Intersection_MCS_0_50_largest.json.gz',
    #            save_path=f'{root_dir}/Data/MCS/Final_Graph_macth_under2-.json.gz')
    
    

# Execute main function
if __name__ == "__main__":
    main()
