from SynRBL.SynMCSImputer.SubStructure.mcs_graph_detector import MCSMissingGraphAnalyzer
from rdkit import Chem
from joblib import Parallel, delayed
from SynRBL.rsmi_utils import save_database, load_database

import os
import copy


def single_mcs(
    data_dict,
    RingMatchesRingOnly=True,
    CompleteRingsOnly=True,
    Timeout=60,
    sort="MCES",
    method="MCES",
    similarityThreshold=0.5,
    remove_substructure=True,
    ignore_bond_order=True,
):
    """
    Performs MCS on a single reaction data entry and captures any issues encountered.

    Parameters:
    - data_dict: Dict containing reaction data.
    - params: Optional parameters for MCS analysis.

    Returns:
    - dict: A dictionary containing MCS results and any sorted reactants encountered.
    """
    mcs_results_dict = copy.deepcopy(data_dict)
    mcs_results_dict["mcs_results"] = []
    mcs_results_dict["sorted_reactants"] = []
    mcs_results_dict["issue"] = []

    try:
        analyzer = MCSMissingGraphAnalyzer()
        mcs_list, sorted_reactants, reactant_mol_list, _ = analyzer.fit(
            data_dict,
            RingMatchesRingOnly=RingMatchesRingOnly,
            CompleteRingsOnly=CompleteRingsOnly,
            sort=sort,
            method=method,
            remove_substructure=remove_substructure,
            Timeout=Timeout,
            similarityThreshold=similarityThreshold,
            ignore_bond_order=ignore_bond_order,
        )
        mcs_list_smiles = [Chem.MolToSmarts(mol) for mol in mcs_list]
        sorted_reactants_smiles = [Chem.MolToSmiles(mol) for mol in sorted_reactants]
        mcs_results_dict["mcs_results"] = mcs_list_smiles
        mcs_results_dict["sorted_reactants"] = sorted_reactants_smiles

        if len(reactant_mol_list) != len(sorted_reactants):
            mcs_results_dict["issue"] = "MCS_Uncertainty"

    except Exception:
        mcs_results_dict["issue"] = "Single MCS identification failed."
    return mcs_results_dict


def ensemble_mcs(data, conditions, batch_size=100, Timeout=60):
    condition_results = []
    for condition in conditions:
        all_results = []  # Accumulate results for each condition

        # Process data in batches
        for start in range(0, len(data), batch_size):
            end = start + batch_size
            batch_results = Parallel(n_jobs=-2, verbose=0)(
                delayed(single_mcs)(data_dict, **condition, Timeout=Timeout)
                for data_dict in data[start:end]
            )
            all_results.extend(batch_results)  # Combine batch results
        condition_results.append(all_results)
    return condition_results

