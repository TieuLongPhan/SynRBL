import pandas as pd
import numpy as np
import sys
from pathlib import Path
root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))

from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCS.SubStructure.extract_common_mcs import ExtractMCS


def main():
    original_data = load_database(f'{root_dir}/Data/Unsolved_reactions.json.gz')
    condition_1 = load_database(f'{root_dir}/Data/MCS/Condition_1.json.gz')
    condition_2 = load_database(f'{root_dir}/Data/MCS/Condition_2.json.gz')
    condition_3 = load_database(f'{root_dir}/Data/MCS/Condition_3.json.gz')
    condition_4 = load_database(f'{root_dir}/Data/MCS/Condition_4.json.gz')
    condition_5 = load_database(f'{root_dir}/Data/MCS/Condition_5.json.gz')

    # # ensemble case
    # analysis = ExtractMCS()
    # mcs_dict_75_100, threshold_index = analysis.extract_matching_conditions(70, 100, condition_1, condition_2, condition_3, condition_4, condition_5,
    #                                                              extraction_method = 'ensemble', using_threshold=True)
    
    # save_database(mcs_dict_75_100, f'{root_dir}/Data/MCS/Intersection_MCS_3+_matching_ensemble.json.gz')
    # data_solve = [d for d, b in zip(original_data, threshold_index) if b]
    # save_database(data_solve, f'{root_dir}/Data/MCS/Original_data_Intersection_MCS_3+_matching_ensemble.json.gz')

    # # Largest case for matching under 50%
    # analysis = ExtractMCS()
    # mcs_dict, threshold_index = analysis.extract_matching_conditions(0, 69, condition_1, condition_2, condition_3, condition_4, condition_5,
    #                                                                 extraction_method = 'largest_mcs', using_threshold=True)

    # save_database(mcs_dict, f'{root_dir}/Data/MCS/Intersection_MCS_0_50_largest.json.gz')
    # data_solve = [d for d, b in zip(original_data, threshold_index) if b]
    # save_database(data_solve, f'{root_dir}/Data/MCS/Original_data_Intersection_MCS_0_50_largest.json.gz')
    analysis = ExtractMCS()
    mcs_dict, threshold_index = analysis.extract_matching_conditions(0, 100, condition_1, condition_2, condition_3, condition_4, condition_5,
                                                                  extraction_method = 'largest_mcs', using_threshold=True)
    save_database(mcs_dict, f'{root_dir}/Data/MCS/largest_mcs.json.gz')
    #print(mcs_dict)
    #save_database(mcs_dict, f'{root_dir}/Data/MCS/test_largest_mcs.json.gz')


    


if __name__ == "__main__":
    main()
