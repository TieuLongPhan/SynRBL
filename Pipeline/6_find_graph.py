import sys
from pathlib import Path
root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCS.MissingGraph.find_graph_dict import find_graph_dict




def main():
    # original_3 = load_database(f'{root_dir}/Data/MCS/Original_data_Intersection_MCS_3+_matching_ensemble.json.gz')
    # missing_results_3_match = find_graph_dict(msc_dict_path=f'{root_dir}/Data/MCS/Intersection_MCS_3+_matching_ensemble.json.gz',
    #            save_path=f'{root_dir}/Data/MCS/Final_Graph_macth_3+.json.gz')
    # #save_database(non_pass_df, root_dir / 'Data/MCS/Bug.json.gz')
    # for key, _ in enumerate(missing_results_3_match):
    #     missing_results_3_match[key]['R-id'] = original_3[key]['R-id']
    #     missing_results_3_match[key]['old_reaction'] = original_3[key]['reactions']
    # save_database(missing_results_3_match, root_dir / 'Data/MCS/Final_Graph_macth_3+.json.gz')
    
    # original_2 = load_database(f'{root_dir}/Data/MCS/Original_data_Intersection_MCS_0_50_largest.json.gz')
    # missing_results_largest  = find_graph_dict(msc_dict_path=f'{root_dir}/Data/MCS/Intersection_MCS_0_50_largest.json.gz',
    #             save_path=f'{root_dir}/Data/MCS/Final_Graph_macth_under2-.json.gz')
    # for key, _ in enumerate(missing_results_largest):
    #     missing_results_largest[key]['R-id'] = original_2[key]['R-id']
    #     missing_results_largest[key]['old_reaction'] = original_2[key]['reactions']
    # save_database(missing_results_largest, root_dir / 'Data/MCS/Final_Graph_macth_under2-.json.gz')

    data = load_database(f'{root_dir}/Data/Unsolved_reactions.json.gz')
    mcs = load_database(f'{root_dir}/Data/MCS/largest_mcs.json.gz')
    # missing_results_largest  = find_graph_dict(msc_dict_path=f'{root_dir}/Data/MCS/largest_mcs.json.gz', save_path=f'{root_dir}/Data/MCS/Final_first_Graph_macth_largest.json.gz')
    # for key, _ in enumerate(missing_results_largest):
    #     missing_results_largest[key]['R-id'] = data[key]['R-id']
    #     missing_results_largest[key]['old_reaction'] = data[key]['reactions']
    # save_database(missing_results_largest, root_dir / 'Data/MCS/Final_first_Graph_macth_largest.json.gz')


    for i in range(1,6):
        missing_results_largest  = find_graph_dict(msc_dict_path=f'{root_dir}/Data/MCS/Condition_{i}.json.gz', 
                                                   save_path=root_dir / f'Data/MCS/Final_graph_macth_condition_{i}.json.gz')
        for key, _ in enumerate(missing_results_largest):
            missing_results_largest[key]['R-id'] = data[key]['R-id']
            missing_results_largest[key]['old_reaction'] = data[key]['reactions']
        save_database(missing_results_largest, root_dir / f'Data/MCS/Final_graph_macth_condition_{i}.json.gz')

    
    

# Execute main function
if __name__ == "__main__":
    main()
