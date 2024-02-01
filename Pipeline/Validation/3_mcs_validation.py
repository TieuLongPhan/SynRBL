import sys
from pathlib import Path
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
from SynRBL.SynMCSImputer.SubStructure.mcs_process import ensemble_mcs
from SynRBL.SynUtils.data_utils import load_database, save_database
from SynRBL.SynMCSImputer.SubStructure.extract_common_mcs import ExtractMCS
from SynRBL.SynMCSImputer.MissingGraph.find_graph_dict import find_graph_dict
from SynRBL.SynMCSImputer.MissingGraph.refinement_uncertainty import RefinementUncertainty

# Define the main function
def mcs(data_name = 'golden_dataset'):
    conditions = [
            {'RingMatchesRingOnly': True, 'CompleteRingsOnly': True, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': True},
            {'RingMatchesRingOnly': True, 'CompleteRingsOnly': True, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': False},
            {'RingMatchesRingOnly': False, 'CompleteRingsOnly': False, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': True},
            {'RingMatchesRingOnly': False, 'CompleteRingsOnly': False, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': False},
            {'method':'MCES', 'sort': 'MCES'},
        ]
    # Load data
    data_path = root_dir / f'Data/Validation_set/{data_name}/mcs_based_reactions.json.gz'
    filtered_data = load_database(data_path)

    # Calculate the path to the save directory
    save_dir = root_dir / f'Data/Validation_set/{data_name}/MCS'
    if not save_dir.exists():
        save_dir.mkdir(parents=True)

    # Run and save conditions
    ensemble_mcs(filtered_data, root_dir, save_dir, conditions, batch_size=5000, Timeout=90)




def merge_mcs(data_name = 'golden_dataset'):
    
    mcs_dir = root_dir / f'Data/Validation_set/{data_name}/MCS'
    condition_1 = load_database(f'{mcs_dir}/Condition_1.json.gz')
    condition_2 = load_database(f'{mcs_dir}/Condition_2.json.gz')
    condition_3 = load_database(f'{mcs_dir}/Condition_3.json.gz')
    condition_4 = load_database(f'{mcs_dir}/Condition_4.json.gz')
    condition_5 = load_database(f'{mcs_dir}/Condition_5.json.gz')

   
    # Select largest MCS, if equal, compare the 1st largest smart
    analysis = ExtractMCS()
    mcs_dict, threshold_index = analysis.extract_matching_conditions(0, 100, condition_1, condition_2, condition_3, condition_4, condition_5,
                                                                  extraction_method = 'largest_mcs', using_threshold=True)
    print(len(mcs_dict))
    save_database(mcs_dict, f'{mcs_dir}/MCS_Largest.json.gz')

    
def graph_find(data_name = 'golden_dataset'):
    data = load_database(f'{root_dir}/Data/Validation_set/{data_name}/mcs_based_reactions.json.gz')
    mcs_dir = root_dir / f'Data/Validation_set/{data_name}/MCS'
    mcs = load_database(mcs_dir/ 'MCS_Largest.json.gz')
    missing_results_largest  = find_graph_dict(msc_dict_path=mcs_dir/ 'MCS_Largest.json.gz', 
                                               save_path= mcs_dir / 'Final_Graph.json.gz')
    miss_id = [value['R-id'] for value in mcs]
    #print(miss_id)
    data_2 = [data[key] for key, value in enumerate(data) if value['R-id'] in miss_id]
    for key, _ in enumerate(missing_results_largest):
        missing_results_largest[key]['R-id'] = mcs[key]['R-id']
        missing_results_largest[key]['sorted_reactants'] = mcs[key]['sorted_reactants']
        missing_results_largest[key]['carbon_balance_check'] = mcs[key]['carbon_balance_check']
        missing_results_largest[key]['mcs_results'] = mcs[key]['mcs_results']
        missing_results_largest[key]['mcs_results'] = mcs[key]['mcs_results']
        missing_results_largest[key]['old_reaction'] = data_2[key]['reactions']


   
    save_database(missing_results_largest, mcs_dir / 'Final_Graph.json.gz')

    


# Execute main function
if __name__ == "__main__":
    #data_name = ['golden_dataset']
    #data_name = ['Jaworski']
    data_name = ['USPTO_unbalance_class']
    # data_name = ['Jaworski', 'USPTO_diff', 'USPTO_unbalance_class']
    #data_name = ['USPTO_diff']
    for i in data_name:
        mcs(i)
        merge_mcs(i)
        graph_find(i)