import pandas as pd
import numpy as np
import sys
from pathlib import Path
root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))

from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCS.SubStructure.extract_common_mcs import ExtractMCS


def main():
    #original_data = load_database(f'{root_dir}/Data/Unsolved_reactions.json.gz')
    condition_1 = load_database(f'{root_dir}/Data/MCS/Condition_1.json.gz')
    condition_2 = load_database(f'{root_dir}/Data/MCS/Condition_2.json.gz')
    condition_3 = load_database(f'{root_dir}/Data/MCS/Condition_3.json.gz')
    condition_4 = load_database(f'{root_dir}/Data/MCS/Condition_4.json.gz')
    condition_5 = load_database(f'{root_dir}/Data/MCS/Condition_5.json.gz')

   
    # Select largest MCS, if equal, compare the 1st largest smart
   
    analysis = ExtractMCS()
    mcs_dict, threshold_index = analysis.extract_matching_conditions(0, 100, condition_1, condition_2, condition_3, condition_4, condition_5,
                                                                  extraction_method = 'largest_mcs', using_threshold=True)
    save_database(mcs_dict, f'{root_dir}/Data/MCS/MCS_Largest.json.gz')

    


if __name__ == "__main__":
    main()
