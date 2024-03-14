import os
import json
import copy
import pickle
import argparse
import hashlib
import logging
import pandas as pd
import numpy as np
import rdkit

from typing import List, Union

from SynRBL.SynRuleImputer import SyntheticRuleImputer
from SynRBL.SynRuleImputer.synthetic_rule_constraint import RuleConstraint
from SynRBL.SynProcessor import (
    RSMIProcessing,
    RSMIDecomposer,
    RSMIComparator,
    BothSideReact,
    CheckCarbonBalance,
)
from SynRBL.rsmi_utils import (
    save_database,
    load_database,
    filter_data,
    extract_results_by_key,
)
from SynRBL.SynMCSImputer.SubStructure.mcs_process import ensemble_mcs
from SynRBL.SynUtils.data_utils import load_database, save_database
from SynRBL.SynMCSImputer.SubStructure.extract_common_mcs import ExtractMCS
from SynRBL.SynMCSImputer.MissingGraph.find_graph_dict import find_graph_dict
from SynRBL.SynAnalysis.analysis_utils import (
    calculate_chemical_properties,
    count_boundary_atoms_products_and_calculate_changes,
)

logger = logging.getLogger(__name__)


class MCS:
    def __init__(self, id_col, solved_col = "solved", mcs_data_col="mcs"):
        self.id_col = id_col
        self.solved_col = solved_col
        self.mcs_data_col = mcs_data_col

        self.conditions = [
            {
                "RingMatchesRingOnly": True,
                "CompleteRingsOnly": True,
                "method": "MCIS",
                "sort": "MCIS",
                "ignore_bond_order": True,
            },
            {
                "RingMatchesRingOnly": True,
                "CompleteRingsOnly": True,
                "method": "MCIS",
                "sort": "MCIS",
                "ignore_bond_order": False,
            },
            {
                "RingMatchesRingOnly": False,
                "CompleteRingsOnly": False,
                "method": "MCIS",
                "sort": "MCIS",
                "ignore_bond_order": True,
            },
            {
                "RingMatchesRingOnly": False,
                "CompleteRingsOnly": False,
                "method": "MCIS",
                "sort": "MCIS",
                "ignore_bond_order": False,
            },
            {"method": "MCES", "sort": "MCES"},
        ]

    def find(self, reactions):
        mcs_reactions = [
            reactions[key]
            for key, value in enumerate(reactions)
            if value["carbon_balance_check"] != "balanced"
            or (
                value["carbon_balance_check"] == "balanced"
                and not value[self.solved_col]
            )
        ]
        logger.info(
            "Find maximum-common-substructure for {} reactions.".format(
                len(mcs_reactions)
            )
        )

        condition_results = ensemble_mcs(
            mcs_reactions, self.conditions, batch_size=5000, Timeout=90
        )

        analysis = ExtractMCS()
        mcs_dict, _ = analysis.extract_matching_conditions(
            0,
            100,
            *condition_results,
            extraction_method="largest_mcs",
            using_threshold=True,
        )

        missing_results_largest = find_graph_dict(mcs_dict)

        assert len(mcs_dict) == len(missing_results_largest)
        for i, r in enumerate(missing_results_largest):
            _id = int(mcs_reactions[i][self.id_col])
            r['sorted_reactants'] = mcs_dict[i]['sorted_reactants']
            r['mcs_results'] = mcs_dict[i]['mcs_results']
            reactions[_id][self.mcs_data_col] = r

        return reactions
