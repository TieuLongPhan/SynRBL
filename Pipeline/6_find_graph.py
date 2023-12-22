import sys
from pathlib import Path
root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCS.MissingGraph.find_graph_dict import find_graph_dict
from SynRBL.SynMCS.MissingGraph.refinement_uncertainty import RefinementUncertainty



def graph_find():


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

def graph_refinment():
    finalgraph = load_database(root_dir / 'Data/MCS/Final_Graph.json.gz')
    finalgraph_certainty = [i for i in finalgraph if i['Certainty'] == True]
    finalgraph_uncertainty = [i for i in finalgraph if i['Certainty'] == False]

    graph_condition_1 = load_database(root_dir / f'Data/MCS/Final_graph_macth_condition_1.json.gz')
    graph_condition_2 = load_database(root_dir / f'Data/MCS/Final_graph_macth_condition_2.json.gz')
    graph_condition_3 = load_database(root_dir / f'Data/MCS/Final_graph_macth_condition_3.json.gz')
    graph_condition_4 = load_database(root_dir / f'Data/MCS/Final_graph_macth_condition_4.json.gz')
    #graph_condition_5 = load_database(root_dir / f'Data/MCS/Final_graph_macth_condition_5.json.gz')

    graph_cond_list = [finalgraph, graph_condition_1, graph_condition_2, graph_condition_3, graph_condition_4]

    refine = RefinementUncertainty(finalgraph_uncertainty, graph_cond_list)
    new_graph = refine.fit(intersection_num=3)
    new_certain = [new_graph[i] for i in range(len(new_graph)) if new_graph[i]['Certainty'] == True]
    new_certain_id = [new_certain[i]['R-id'] for i in range(len(new_certain))]

    finalgraph_certainty.extend(new_certain)

    remove_list = [x for x in finalgraph_uncertainty if x['R-id'] in new_certain_id]
    for i in remove_list:
        finalgraph_uncertainty.remove(i) 

    save_database(finalgraph_certainty, root_dir / 'Data/MCS/Final_Graph_certainty.json.gz')
    save_database(finalgraph_uncertainty, root_dir / 'Data/MCS/Final_Graph_uncertainty.json.gz')

    
    
    

# Execute main function
if __name__ == "__main__":
    #graph_find()
    graph_refinment()
