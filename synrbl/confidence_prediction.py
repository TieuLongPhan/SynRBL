import joblib
import pandas as pd
import numpy as np
import importlib.resources
import synrbl.SynAnalysis

from synrbl.SynAnalysis.analysis_utils import (
    calculate_chemical_properties,
    count_boundary_atoms_products_and_calculate_changes,
)
from synrbl.SynUtils.common import update_reactants_and_products


class ConfidencePredictor:
    def __init__(
        self,
        reaction_col="reaction",
        input_reaction_col="input_reaction",
        confidence_col="confidence",
        solved_col="solved",
        solved_by_col="solved_by",
        solved_by_method="mcs-based",
        issue_col="issue",
        mcs_col="mcs",
    ):
        self.model = joblib.load(
            importlib.resources.files(synrbl.SynAnalysis)
            .joinpath("scoring_function.dump")
            .open("rb")
        )
        self.reaction_col = reaction_col
        self.input_reaction_col = input_reaction_col
        self.confidence_col = confidence_col
        self.solved_col = solved_col
        self.solved_by_col = solved_by_col
        self.solved_by_method = solved_by_method
        self.issue_col = issue_col
        self.mcs_col = mcs_col

    def predict(self, reactions, stats=None, threshold=0):
        reactions = [
            r
            for r in reactions
            if self.solved_by_col in r.keys()
            and r[self.solved_by_col] == self.solved_by_method
        ]
        conf_success = 0
        if len(reactions) > 0:
            _reactions = count_boundary_atoms_products_and_calculate_changes(
                reactions, self.reaction_col, self.mcs_col
            )
            update_reactants_and_products(_reactions, self.input_reaction_col)
            _reactions = calculate_chemical_properties(_reactions)

            df = pd.DataFrame(_reactions)

            X_pred = df[
                [
                    "carbon_difference",
                    "fragment_count",
                    "total_carbons",
                    "total_bonds",
                    "total_rings",
                    "num_boundary",
                    "ring_change_merge",
                    "bond_change_merge",
                ]
            ]

            confidence = np.round(self.model.predict_proba(X_pred)[:, 1], 3)
            assert len(reactions) == len(confidence)
            for r, c in zip(reactions, confidence):
                r[self.confidence_col] = c.item()
                if c >= threshold:
                    conf_success += 1
                else:
                    assert (
                        r[self.issue_col] == ""
                    ), "Issue column has value for a solved reaction?"
                    r[self.solved_col] = False
                    r[self.issue_col] = (
                        "Confidence is below the threshold of {:.2%}.".format(threshold)
                    )
        if stats is not None:
            stats["confident_cnt"] = conf_success
        return reactions
