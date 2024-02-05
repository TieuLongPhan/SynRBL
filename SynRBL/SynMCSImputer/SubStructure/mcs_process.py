from SynRBL.SynMCSImputer.SubStructure.mcs_graph_detector import MCSMissingGraphAnalyzer
from rdkit import Chem
from joblib import Parallel, delayed
from SynRBL.rsmi_utils import save_database, load_database

import os
import logging


def single_mcs(data_dict, RingMatchesRingOnly=True, CompleteRingsOnly=True, Timeout=60,
               sort='MCES', method='MCES', similarityThreshold=0.5, 
               remove_substructure=True, ignore_bond_order = True):
    """
    Performs MCS on a single reaction data entry and captures any issues encountered.

    Parameters:
    - data_dict: Dict containing reaction data.
    - params: Optional parameters for MCS analysis.

    Returns:
    - dict: A dictionary containing MCS results and any sorted reactants encountered.
    """

    mcs_results_dict = {'R-id': data_dict['R-id'], 'mcs_results': [], 'sorted_reactants': [], 'issue': [], 'carbon_balance_check': data_dict['carbon_balance_check'], }

    try:
        analyzer = MCSMissingGraphAnalyzer()
        mcs_list, sorted_reactants, reactant_mol_list, _ = analyzer.fit(data_dict, RingMatchesRingOnly=RingMatchesRingOnly,
                                                     CompleteRingsOnly=CompleteRingsOnly, sort=sort, method=method,
                                                     remove_substructure=remove_substructure, Timeout=Timeout,
                                                     similarityThreshold=similarityThreshold, ignore_bond_order=ignore_bond_order)
        mcs_list_smiles = [Chem.MolToSmarts(mol) for mol in mcs_list]
        sorted_reactants_smiles = [Chem.MolToSmiles(mol) for mol in sorted_reactants]
        mcs_results_dict['mcs_results'] = mcs_list_smiles
        mcs_results_dict['sorted_reactants'] = sorted_reactants_smiles

        if len(reactant_mol_list) != len(sorted_reactants):
            mcs_results_dict['issue'] = 'MCS_Uncertainty'

    except Exception as e:
        mcs_results_dict['issue'] = data_dict
        logging.error(f"Error in single_mcs: {str(e)}")
    return mcs_results_dict


def ensemble_mcs(data, root_dir, save_dir, conditions, batch_size=100, Timeout=60):
    # Configure logging
    logging.basicConfig(filename=os.path.join(save_dir, f'process.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if conditions is None:
        conditions = [
            {'RingMatchesRingOnly': True, 'CompleteRingsOnly': True, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': True},
            {'RingMatchesRingOnly': True, 'CompleteRingsOnly': True, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': False},
            {'RingMatchesRingOnly': False, 'CompleteRingsOnly': False, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': True},
            {'RingMatchesRingOnly': False, 'CompleteRingsOnly': False, 'method':'MCIS', 'sort': 'MCIS', 'ignore_bond_order': False},
            {'method':'MCES', 'sort': 'MCES'},
        ]
    for idx, condition in enumerate(conditions, start=1):
        all_results = []  # Accumulate results for each condition

        # Process data in batches
        for start in range(0, len(data), batch_size):
            end = start + batch_size
            batch_results = Parallel(n_jobs=-2)(delayed(single_mcs)(data_dict, **condition, Timeout=Timeout) for data_dict in data[start:end])
            all_results.extend(batch_results)  # Combine batch results

            # Calculate progress percentages
            batch_progress = (end / len(data)) * 100
            data_progress = (start / len(data)) * 100
            logging.info(f"Condition {idx} | Batch Progress: {batch_progress:.2f}% | Data Progress: {data_progress:.2f}%")

        # Save all results for the current condition into a single file
        save_database(all_results, pathname=os.path.join(save_dir, f'Condition_{idx}.json.gz'))
        logging.info(f"Condition {idx}: Finished")

    # After processing all conditions
    logging.info("All conditions have been processed.")
