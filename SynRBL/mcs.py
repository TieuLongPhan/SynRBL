import logging

from SynRBL.SynMCSImputer.SubStructure.mcs_process import ensemble_mcs
from SynRBL.SynMCSImputer.SubStructure.extract_common_mcs import ExtractMCS
from SynRBL.SynMCSImputer.MissingGraph.find_graph_dict import find_graph_dict

logger = logging.getLogger(__name__)


class MCS:
    def __init__(self, id_col, solved_col="solved", mcs_data_col="mcs", n_jobs=-1):
        self.id_col = id_col
        self.solved_col = solved_col
        self.mcs_data_col = mcs_data_col
        self.n_jobs = n_jobs

        self.conditions = [
            {
                "RingMatchesRingOnly": True,
                "CompleteRingsOnly": True,
                "method": "MCIS",
                "sort": "MCIS",
                "ignore_bond_order": True,
            },
            #{
            #    "RingMatchesRingOnly": True,
            #    "CompleteRingsOnly": True,
            #    "method": "MCIS",
            #    "sort": "MCIS",
            #    "ignore_bond_order": False,
            #},
            {
                "RingMatchesRingOnly": False,
                "CompleteRingsOnly": False,
                "method": "MCIS",
                "sort": "MCIS",
                "ignore_bond_order": True,
            },
            #{
            #    "RingMatchesRingOnly": False,
            #    "CompleteRingsOnly": False,
            #    "method": "MCIS",
            #    "sort": "MCIS",
            #    "ignore_bond_order": False,
            #},
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
        if len(mcs_reactions) == 0:
            return reactions
        logger.info(
            "Find maximum-common-substructure for {} reactions.".format(
                len(mcs_reactions)
            )
        )

        condition_results = ensemble_mcs(
            mcs_reactions, self.conditions, n_jobs=self.n_jobs, Timeout=30
        )

        analysis = ExtractMCS()
        mcs_dict, _ = analysis.extract_matching_conditions(
            0,
            100,
            *condition_results,
            extraction_method="largest_mcs",
            using_threshold=True,
        )
        if len(mcs_dict) == 0:
            return reactions

        missing_results_largest = find_graph_dict(mcs_dict)

        assert len(mcs_dict) == len(missing_results_largest)
        for i, r in enumerate(missing_results_largest):
            _id = int(mcs_reactions[i][self.id_col])
            r["sorted_reactants"] = mcs_dict[i]["sorted_reactants"]
            r["mcs_results"] = mcs_dict[i]["mcs_results"]
            reactions[_id][self.mcs_data_col] = r

        return reactions
