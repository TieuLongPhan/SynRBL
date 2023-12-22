import sys
from pathlib import Path
root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCS.MissingGraph.find_graph_dict import find_graph_dict




def main():


    data = load_database(f'{root_dir}/Data/Unsolved_reactions.json.gz')
    missing_results_largest  = find_graph_dict(msc_dict_path=f'{root_dir}/Data/MCS/MCS_Largest.json.gz', save_path=f'{root_dir}/Data/MCS/Final_Graph.json.gz')
    for key, _ in enumerate(missing_results_largest):
        missing_results_largest[key]['R-id'] = data[key]['R-id']
        missing_results_largest[key]['old_reaction'] = data[key]['reactions']
    save_database(missing_results_largest, root_dir / 'Data/MCS/Final_Graph.json.gz')


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
