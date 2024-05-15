import copy
import logging
import pandas as pd

from synrbl.preprocess import preprocess
from synrbl.postprocess import Validator
from synrbl.rule_based import RuleBasedMethod
from synrbl.mcs_search import MCSSearch
from synrbl.SynMCSImputer.mcs_based_method import MCSBasedMethod
from synrbl.SynChemImputer.post_process import PostProcess
from synrbl.SynChemImputer.molecule_standardizer import MoleculeStandardizer
from synrbl.confidence_prediction import ConfidencePredictor

logger = logging.getLogger("synrbl")


class Balancer:
    def __init__(
        self, id_col="id", reaction_col="reaction", confidence_threshold=0, n_jobs=-1
    ):
        self.__reaction_col = reaction_col
        self.__id_col = id_col
        self.__solved_col = "solved"
        self.__solved_by_col = "solved_by"
        self.__mcs_data_col = "mcs"
        self.__input_col = "input_reaction"
        self.__confidence_col = "confidence"
        self.__unbalance_col = "unbalance_col"
        self.__carbon_balance_col = "carbon_balance_check"
        self.__rules_col = "rules"
        self.__issue_col = "issue"
        self.columns = [
            self.__input_col,
            reaction_col,
            self.__solved_col,
            self.__solved_by_col,
            self.__confidence_col,
            self.__rules_col,
            self.__issue_col,
        ]

        self.confidence_threshold = confidence_threshold
        self.input_validator = Validator(
            reaction_col,
            "input-balanced",
            n_jobs=n_jobs,
            solved_col=self.__solved_col,
            solved_method_col=self.__solved_by_col,
            unbalance_col=self.__unbalance_col,
            carbon_balance_col=self.__carbon_balance_col,
            issue_col=self.__issue_col,
        )
        self.rb_validator = Validator(
            reaction_col,
            "rule-based",
            check_carbon_balance=False,
            n_jobs=n_jobs,
            solved_col=self.__solved_col,
            solved_method_col=self.__solved_by_col,
            unbalance_col=self.__unbalance_col,
            carbon_balance_col=self.__carbon_balance_col,
            issue_col=self.__issue_col,
        )
        self.mcs_validator = Validator(
            reaction_col,
            "mcs-based",
            n_jobs=n_jobs,
            solved_col=self.__solved_col,
            solved_method_col=self.__solved_by_col,
            unbalance_col=self.__unbalance_col,
            carbon_balance_col=self.__carbon_balance_col,
            issue_col=self.__issue_col,
        )

        self.rb_method = RuleBasedMethod(
            id_col, reaction_col, reaction_col, n_jobs=n_jobs
        )
        self.mcs_search = MCSSearch(
            id_col,
            solved_col=self.__solved_col,
            mcs_data_col=self.__mcs_data_col,
            issue_col=self.__issue_col,
            n_jobs=n_jobs,
        )
        self.mcs_method = MCSBasedMethod(
            reaction_col,
            reaction_col,
            mcs_data_col=self.__mcs_data_col,
            issue_col=self.__issue_col,
            rules_col=self.__rules_col,
            smiles_standardizer=[MoleculeStandardizer()],
        )
        self.post_processor = PostProcess(
            id_col=id_col, reaction_col=reaction_col, n_jobs=n_jobs, verbose=0
        )
        self.conf_predictor = ConfidencePredictor(
            reaction_col=reaction_col,
            solved_by_method="mcs-based",
            input_reaction_col=self.__input_col,
            confidence_col=self.__confidence_col,
            solved_col=self.__solved_col,
            solved_by_col=self.__solved_by_col,
            issue_col=self.__issue_col,
            mcs_col=self.__mcs_data_col,
        )

    def __post_process(self, reactions):
        key_index_map = {item[self.__id_col]: idx for idx, item in enumerate(reactions)}
        pp_data = [
            r
            for r in reactions
            if self.__solved_by_col in r.keys()
            and r[self.__solved_by_col] != "input-balanced"
        ]
        pp_results = self.post_processor.fit(pp_data)
        for pp_result in pp_results:
            if (
                pp_result["label"] != "unspecified"
                and "curated_reaction" in pp_result.keys()
            ):
                idx = key_index_map[pp_result[self.__id_col]]
                reactions[idx][self.__reaction_col] = pp_result["curated_reaction"]

    def __run_pipeline(self, reactions, stats=None):
        if stats is not None:
            stats["reaction_cnt"] = len(reactions)
        reactions = preprocess(
            reactions,
            self.__reaction_col,
            self.__id_col,
            self.__solved_col,
            self.__input_col,
        )
        rxn_cnt = len(reactions)
        self.input_validator.check(reactions)

        logger.info("Run rule-based method.")
        self.rb_method.run(reactions, stats=stats)
        self.rb_validator.check(reactions, override_unsolved=True)

        self.mcs_search.find(reactions)

        logger.info("Impute missing compounds from MCS.")
        self.mcs_method.run(reactions, stats=stats)
        self.mcs_validator.check(reactions)  # update carbon balance
        logger.info(
            "Run rule-based method again to fix remaining non-carbon imbalance."
        )
        self.__post_process(reactions)
        self.rb_method.run(reactions)
        self.mcs_validator.check(reactions, override_unsolved=True)

        self.conf_predictor.predict(
            reactions, stats=stats, threshold=self.confidence_threshold
        )

        assert rxn_cnt == len(reactions)

        logger.info("DONE")

        return reactions

    def rebalance(self, reactions, output_dict=False, stats=None):
        if isinstance(reactions, str):
            reactions = [reactions]
        if not isinstance(reactions, list):
            raise ValueError("Expected a list of reactions.")
        if len(reactions) == 0:
            return []
        if isinstance(reactions[0], str):
            reactions = pd.DataFrame({self.__reaction_col: reactions})
        result = self.__run_pipeline(copy.deepcopy(reactions), stats)

        if output_dict:
            output = []
            for r in result:
                output.append({k: v for k, v in r.items() if k in self.columns})
            return output
        else:
            return [r[self.__reaction_col] for r in result]
