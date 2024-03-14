import copy
import logging
import pandas as pd

from SynRBL.preprocess import preprocess
from SynRBL.postprocess import Validator
from SynRBL.rule_based import RuleBasedMethod
from SynRBL.mcs import MCS
from SynRBL.SynMCSImputer.model import MCSBasedMethod
from SynRBL.confidence_prediction import ConfidencePredictor

logger = logging.getLogger("SynRBL")


class SynRBL:
    def __init__(self, id_col="id", reaction_col="reaction"):
        self.reaction_col = reaction_col
        self.id_col = id_col
        self.solved_col = "solved"
        self.mcs_data_col = "mcs"
        self.columns = [
            "input_reaction",
            "reaction",
            "solved",
            "solved_by",
            "confidence",
            "rules",
        ]

    def __run_pipeline(self, reactions):
        r_col = self.reaction_col
        input_validator = Validator(r_col, "input-balanced")
        rb_validator = Validator(r_col, "rule-based", check_carbon_balance=False)
        mcs_validator = Validator(r_col, "mcs-based")

        reactions = preprocess(reactions, r_col, self.id_col, self.solved_col)
        l = len(reactions)
        input_validator.check(reactions)

        logger.info("Run rule-based method.")
        rb_method = RuleBasedMethod(
            self.id_col, r_col, r_col, "./Data/Rules/rules_manager.json.gz"
        )
        rb_method.run(reactions)
        rb_validator.check(reactions)

        logger.info("Find maximum common substructure.")
        mcs = MCS(self.id_col, mcs_data_col=self.mcs_data_col)
        mcs.find(reactions)

        logger.info("Impute missing compounds from MCS.")
        mcs_method = MCSBasedMethod(r_col, r_col, mcs_data_col=self.mcs_data_col)
        mcs_method.run(reactions)
        mcs_validator.check(reactions)  # update carbon balance
        logger.info(
            "Run rule-based method again to fix remaining non-carbon imbalance."
        )
        rb_method.run(reactions)
        mcs_validator.check(reactions)

        conf_predictor = ConfidencePredictor()
        conf_predictor.predict(reactions)

        assert l == len(reactions)

        logger.info("DONE")

        return reactions

    def rebalance(self, reactions, output_dict=False):
        if not isinstance(reactions, list):
            raise ValueError("Expected a list of reactions.")
        if len(reactions) == 0:
            return []
        if isinstance(reactions[0], str):
            reactions = pd.DataFrame({self.reaction_col: reactions})
        result = self.__run_pipeline(copy.deepcopy(reactions))

        if output_dict:
            output = []
            for r in result:
                output.append({k: v for k, v in r.items() if k in self.columns})
            return output
        else:
            return [r[self.reaction_col] for r in result]
