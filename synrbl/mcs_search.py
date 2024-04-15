import logging

from synrbl.SynMCSImputer.SubStructure.mcs_process import ensemble_mcs
from synrbl.SynMCSImputer.SubStructure.extract_common_mcs import ExtractMCS
from synrbl.SynMCSImputer.MissingGraph.find_graph_dict import find_graph_dict

logger = logging.getLogger(__name__)


class MCSSearch:
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
        mcs_keys = [
            key
            for key, value in enumerate(reactions)
            if value["carbon_balance_check"] != "balanced"
            or (
                value["carbon_balance_check"] == "balanced"
                and not value[self.solved_col]
            )
        ]
        mcs_reactions = [reactions[k] for k in mcs_keys]

        if len(mcs_reactions) == 0:
            return reactions

        logger.info(
            "Find maximum-common-substructure for {} reactions.".format(
                len(mcs_reactions)
            )
        )

        condition_results = ensemble_mcs(
            mcs_reactions, self.conditions, n_jobs=self.n_jobs, Timeout=60
        )

        assert len(self.conditions) == len(condition_results)
        assert len(mcs_keys) == len(condition_results[0])

        for i, k in enumerate(mcs_keys):
            max_atom_cnt = 0
            max_cond = None
            for cond_col in condition_results:
                mcs_atom_cnt = sum(
                    ExtractMCS.get_num_atoms(mcs) for mcs in cond_col[i]["mcs_results"]
                )
                if mcs_atom_cnt > max_atom_cnt:
                    max_atom_cnt = mcs_atom_cnt
                    max_cond = cond_col[i]
            assert max_cond is not None
            reactions[k][self.mcs_data_col] = find_graph_dict([max_cond])[0]
            reactions[k][self.mcs_data_col]["sorted_reactants"] = max_cond[
                "sorted_reactants"
            ]
            reactions[k][self.mcs_data_col]["mcs_results"] = max_cond["mcs_results"]

        return reactions
