import logging

from synrbl.SynMCSImputer.SubStructure.mcs_process import ensemble_mcs
from synrbl.SynMCSImputer.SubStructure.extract_common_mcs import ExtractMCS
from synrbl.SynMCSImputer.MissingGraph.find_graph_dict import find_graph_dict

logger = logging.getLogger(__name__)


class MCSSearch:
    def __init__(
        self,
        id_col,
        solved_col="solved",
        mcs_data_col="mcs",
        issue_col="issue",
        n_jobs=-1,
    ):
        self.id_col = id_col
        self.solved_col = solved_col
        self.mcs_data_col = mcs_data_col
        self.issue_col = issue_col
        self.n_jobs = n_jobs

        self.conditions = [
            {
                "RingMatchesRingOnly": True,
                "CompleteRingsOnly": True,
                "method": "MCIS",
                "sort": "MCIS",
                "ignore_bond_order": True,
            },
            # {
            #     "RingMatchesRingOnly": True,
            #     "CompleteRingsOnly": True,
            #     "method": "MCIS",
            #     "sort": "MCIS",
            #     "ignore_bond_order": False,
            # },
            {
                "RingMatchesRingOnly": False,
                "CompleteRingsOnly": False,
                "method": "MCIS",
                "sort": "MCIS",
                "ignore_bond_order": True,
            },
            # {
            #     "RingMatchesRingOnly": False,
            #     "CompleteRingsOnly": False,
            #     "method": "MCIS",
            #     "sort": "MCIS",
            #     "ignore_bond_order": False,
            # },
            {"method": "MCES", "sort": "MCES"},
        ]

    def find(self, reactions):
        id2idx_map = {}
        mcs_reactions = []
        for idx, reaction in enumerate(reactions):
            if reaction[self.solved_col]:
                continue
            id2idx_map[reaction[self.id_col]] = idx
            reaction[self.mcs_data_col] = None
            reaction[self.issue_col] = "No MCS identified."
            mcs_reactions.append(reaction)

        if len(mcs_reactions) == 0:
            return reactions

        logger.info(
            "Find maximum-common-substructure for {} reactions.".format(
                len(mcs_reactions)
            )
        )

        condition_results = ensemble_mcs(
            mcs_reactions,
            self.conditions,
            id_col=self.id_col,
            issue_col=self.issue_col,
            n_jobs=self.n_jobs,
        )

        largest_conditions = ExtractMCS.get_largest_condition(*condition_results)

        mcs_results = find_graph_dict(largest_conditions, n_jobs=self.n_jobs)

        assert len(largest_conditions) == len(mcs_results)
        for largest_condition, mcs_result in zip(largest_conditions, mcs_results):
            _id = largest_condition[self.id_col]
            _idx = id2idx_map[_id]
            for k, v in largest_condition.items():
                mcs_result[k] = v
            reactions[_idx][self.mcs_data_col] = mcs_result
            reactions[_idx][self.issue_col] = mcs_result[self.issue_col]

        return reactions
