import copy
import logging
import pandas as pd

from synrbl.preprocess import preprocess
from synrbl.postprocess import Validator
from synrbl.rule_based import RuleBasedMethod
from synrbl.mcs import MCS
from synrbl.SynMCSImputer.model import MCSBasedMethod
from synrbl.confidence_prediction import ConfidencePredictor

logger = logging.getLogger("SynRBL")


class Balancer:
    def __init__(
        self, id_col="id", reaction_col="reaction", confidence_threshold=0, n_jobs=-1
    ):
        self.__reaction_col = reaction_col
        self.__id_col = id_col
        self.solved_col = "solved"
        self.mcs_data_col = "mcs"
        self.columns = [
            "input_reaction",
            reaction_col,
            "solved",
            "solved_by",
            "confidence",
            "rules",
        ]
        self.confidence_threshold = confidence_threshold
        self.input_validator = Validator(reaction_col, "input-balanced", n_jobs=n_jobs)
        self.rb_validator = Validator(
            reaction_col, "rule-based", check_carbon_balance=False, n_jobs=n_jobs
        )
        self.mcs_validator = Validator(reaction_col, "mcs-based", n_jobs=n_jobs)
        self.rb_method = RuleBasedMethod(
            id_col, reaction_col, reaction_col, n_jobs=n_jobs
        )
        self.mcs = MCS(id_col, mcs_data_col=self.mcs_data_col, n_jobs=n_jobs)
        self.mcs_method = MCSBasedMethod(
            reaction_col, reaction_col, mcs_data_col=self.mcs_data_col
        )
        self.conf_predictor = ConfidencePredictor(reaction_col=reaction_col)

    def __run_pipeline(self, reactions, stats=None):
        if stats is not None:
            stats["reaction_cnt"] = len(reactions)
        reactions = preprocess(
            reactions, self.__reaction_col, self.__id_col, self.solved_col
        )
        rxn_cnt = len(reactions)
        self.input_validator.check(reactions)

        logger.info("Run rule-based method.")
        self.rb_method.run(reactions, stats=stats)
        self.rb_validator.check(reactions)

        self.mcs.find(reactions)

        logger.info("Impute missing compounds from MCS.")
        self.mcs_method.run(reactions, stats=stats)
        self.mcs_validator.check(reactions)  # update carbon balance
        logger.info(
            "Run rule-based method again to fix remaining non-carbon imbalance."
        )
        self.rb_method.run(reactions)
        self.mcs_validator.check(reactions)

        self.conf_predictor.predict(
            reactions, stats=stats, threshold=self.confidence_threshold
        )

        assert rxn_cnt == len(reactions)

        logger.info("DONE")

        return reactions

    def rebalance(self, reactions, output_dict=False, stats=None):
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
