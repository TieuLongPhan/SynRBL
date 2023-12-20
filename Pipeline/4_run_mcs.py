import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
from pathlib import Path
root_dir = Path(__file__).parents[1]
sys.path.append(str(root_dir))
from SynRBL.SynMCS.SubStructure.mcs_process import ensemble_mcs
from SynRBL.rsmi_utils import load_database, save_database
from rdkit import Chem
import logging


# Define the main function
def main():
    # Calculate the path to the root directory (two levels up)
    root_dir = Path(__file__).parents[1]
    sys.path.append(str(root_dir))
    conditions = [
            {'RingMatchesRingOnly': True, 'CompleteRingsOnly': True, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': True},
            {'RingMatchesRingOnly': True, 'CompleteRingsOnly': True, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': False},
            {'RingMatchesRingOnly': False, 'CompleteRingsOnly': False, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': True},
            {'RingMatchesRingOnly': False, 'CompleteRingsOnly': False, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': False},
            {'method':'MCES', 'sort': 'MCES'},
        ]
    # Load data
    data_path = root_dir / 'Data/Unsolved_reactions.json.gz'
    filtered_data = load_database(data_path)


    # Run and save conditions
    ensemble_mcs(filtered_data, root_dir, conditions, batch_size=4000, Timeout=60)

# Execute main function
if __name__ == "__main__":
    main()