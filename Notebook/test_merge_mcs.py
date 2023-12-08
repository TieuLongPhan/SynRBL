import pandas as pd
import numpy as np
import sys
from pathlib import Path
root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))

from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCS.extract_common_mcs import ExtractMCS


def main(percentage=100):
    original_data = load_database(f'{root_dir}/Data/Unsolved_reactions.json.gz')
    condition_1 = load_database(f'{root_dir}/Data/MCS/Condition_1.json.gz')
    condition_2 = load_database(f'{root_dir}/Data/MCS/Condition_2.json.gz')
    condition_3 = load_database(f'{root_dir}/Data/MCS/Condition_3.json.gz')
    condition_4 = load_database(f'{root_dir}/Data/MCS/Condition_4.json.gz')

    analysis = ExtractMCS()
    mcs_dict, threshold_index = analysis.extract_common_mcs(percentage, condition_1, condition_2, condition_3, condition_4)
    data_solve = [d for d, b in zip(original_data, threshold_index) if b]

    save_database(mcs_dict, f'{root_dir}/Data/MCS/Intersection_MCS.json.gz')
    save_database(data_solve, f'{root_dir}/Data/MCS/MCS_Solved_reactions.json.gz')

if __name__ == "__main__":
    main(percentage=100)
