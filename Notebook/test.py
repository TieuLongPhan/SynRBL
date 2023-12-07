import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
from pathlib import Path
# Calculate the path to the root directory (two levels up)
root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))
from SynRBL.SynMCS.mcs_missing_graph_analyzer import MCSMissingGraphAnalyzer
from SynRBL.SynMCS.find_missing_graphs import find_missing_parts_pairs
from SynRBL.rsmi_utils import load_database, save_database
from rdkit import Chem

def single_mcs(data_dict, RingMatchesRingOnly=True, CompleteRingsOnly=True,
               sort='MCS', remove_substructure=True):
    """
    Performs mcs on a single reaction data entry and captures any issues encountered.

    Parameters:
    - data_dict: Dict containing reaction data.
    - params: Optional parameters for MCS analysis.

    Returns:
    - dict: A dictionary containing mcs results and any sorted reactants encountered.
    """

    mcs_results_dict = {'mcs_results': [], 'sorted_reactants': [], 'issue': []}

    try:
        analyzer = MCSMissingGraphAnalyzer()
        mcs_list, sorted_reactants, _ = analyzer.fit(data_dict,  RingMatchesRingOnly=RingMatchesRingOnly, CompleteRingsOnly=CompleteRingsOnly,
                                                       sort=sort, remove_substructure=remove_substructure)
        mcs_list_smiles = [Chem.MolToSmiles(mol) for mol in mcs_list]
        sorted_reactants_smiles = [Chem.MolToSmiles(mol) for mol in sorted_reactants]
        mcs_results_dict['mcs_results'] = mcs_list_smiles
        mcs_results_dict['sorted_reactants'] = sorted_reactants_smiles

    except Exception as e:
        mcs_results_dict['issue'] = data_dict

    return mcs_results_dict

def run_and_save_conditions(data, root_dir):
    conditions = [
        {'RingMatchesRingOnly': True, 'CompleteRingsOnly': True, 'remove_substructure': True},
        {'RingMatchesRingOnly': True, 'CompleteRingsOnly': True, 'remove_substructure': False},
        {'RingMatchesRingOnly': False, 'CompleteRingsOnly': False, 'remove_substructure': True},
        {'RingMatchesRingOnly': False, 'CompleteRingsOnly': False, 'remove_substructure': False},
    ]

    for idx, condition in enumerate(conditions, start=1):
        mcs_results = Parallel(n_jobs=-2)(delayed(single_mcs)(data_dict, **condition) for data_dict in data[20:21])
        save_database(mcs_results, pathname=root_dir / f'Data/Condition_{idx}.json.gz')

# Define the main function
def main():
    # Calculate the path to the root directory (two levels up)
    root_dir = Path(__file__).parents[1]
    sys.path.append(str(root_dir))

    # Load data
    data_path = root_dir / 'Data/Unsolved_reactions.json.gz'
    filtered_data = load_database(data_path)

    # Run and save conditions
    run_and_save_conditions(filtered_data, root_dir)

# Execute main function
if __name__ == "__main__":
    main()