import datetime
import time
import logging

import rdkit.Chem.rdmolfiles as rdmolfiles

from synrbl.SynMCSImputer.SubStructure.mcs_graph_detector import MCSMissingGraphAnalyzer
from joblib import Parallel, delayed

from rdkit.rdBase import BlockLogs

logger = logging.getLogger("synrbl")


def single_mcs(
    data_dict,
    id_col="id",
    issue_col="issue",
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
    block_logs = BlockLogs()
    mcs_data = {
        id_col: data_dict[id_col],
        "mcs_results": [],
        "sorted_reactants": [],
        issue_col: "",
    }

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

        if len(reactant_mol_list) != len(sorted_reactants):
            mcs_data["issue"] = "Uncertian MCS."
        else:
            mcs_data["mcs_results"] = [rdmolfiles.MolToSmarts(mol) for mol in mcs_list]
            mcs_data["sorted_reactants"] = [
                rdmolfiles.MolToSmiles(mol) for mol in sorted_reactants
            ]
    except Exception as e:
        mcs_data[issue_col] = "MCS identification failed. {}".format(str(e))

    del block_logs
    return mcs_data


def ensemble_mcs(
    data, conditions, id_col="id", issue_col="issue", n_jobs=-1, Timeout=60
):
    condition_results = []
    start_time = time.time()
    last_tsmp = start_time
    for i, condition in enumerate(conditions):
        all_results = []  # Accumulate results for each condition

        p_generator = Parallel(n_jobs=n_jobs, verbose=0, return_as="generator")(
            delayed(single_mcs)(
                data_dict,
                id_col=id_col,
                issue_col=issue_col,
                **condition,
                Timeout=Timeout,
            )
            for data_dict in data
        )
        for result in p_generator:
            all_results.append(result)  # Combine batch results
            prg = (i * len(data) + len(all_results)) / (len(conditions) * len(data))
            t = time.time()
            if t - last_tsmp > 10:
                eta = (t - start_time) * (1 / prg - 1)
                logger.info(
                    "MCS Progress {:.2%} ETA {}".format(
                        prg, datetime.timedelta(seconds=int(eta))
                    )
                )
                last_tsmp = t
        condition_results.append(all_results)
    return condition_results
