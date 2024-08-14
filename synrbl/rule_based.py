import json
import logging
import pandas as pd
import importlib.resources


import synrbl.SynRuleImputer
from synrbl.SynRuleImputer import SyntheticRuleImputer
from synrbl.SynRuleImputer.synthetic_rule_constraint import RuleConstraint
from synrbl.SynProcessor import (
    RSMIDecomposer,
    RSMIComparator,
    BothSideReact,
)
from synrbl.rsmi_utils import (
    load_database,
    filter_data,
    extract_results_by_key,
)

from synrbl.SynUtils.common import update_reactants_and_products

logger = logging.getLogger(__name__)


class RuleBasedMethod:
    def __init__(self, id_col, reaction_col, output_col, rules_path=None, n_jobs=1):
        self.id_col = id_col
        self.reaction_col = reaction_col
        if rules_path is None:
            with (
                importlib.resources.files(synrbl.SynRuleImputer)
                .joinpath("rules_manager.json.gz")
                .open("r") as f
            ):
                self.rules = json.load(f)
        else:
            self.rules = load_database(rules_path)
        self.n_jobs = n_jobs
        self.output_col = output_col

    def run(self, reactions, stats=None):
        update_reactants_and_products(reactions, self.reaction_col)
        decompose = RSMIDecomposer(
            smiles=None,  # type: ignore
            data=reactions,  # type: ignore
            reactant_col="reactants",
            product_col="products",
            parallel=True,
            n_jobs=self.n_jobs,
            verbose=0,
        )
        react_dict, product_dict = decompose.data_decomposer()

        comp = RSMIComparator(
            reactants=react_dict,  # type: ignore
            products=product_dict,  # type: ignore
            n_jobs=self.n_jobs,
            verbose=0,
        )
        unbalance, diff_formula = comp.run_parallel(
            reactants=react_dict, products=product_dict
        )

        both_side = BothSideReact(react_dict, product_dict, unbalance, diff_formula)
        diff_formula, unbalance = both_side.fit(n_jobs=self.n_jobs)

        # Handle oxidation case
        for index, formula in enumerate(diff_formula):
            if unbalance[index] == "Both":
                # Extract oxygen ratio if present, or skip to next iteration if absent.
                water_ratio = formula.get("O")
                if water_ratio is None:
                    continue

                hydrogen_change = 2 * water_ratio
                additional_oxygen = ".O" * water_ratio

                # Update the reactions list with additional oxygen on both specified keys.
                reactions[index]["products"] += additional_oxygen
                reactions[index][self.reaction_col] += additional_oxygen

                # Remove the 'O' key as it's processed now.
                del formula["O"]

                formula["H"] = formula.get("H", 0) - hydrogen_change

                # Check the new hydrogen count to update balance status.
                if formula["H"] >= 0:
                    unbalance[index] = "Products"
                else:
                    formula["H"] = -formula["H"]
                    unbalance[index] = "Reactants"

        _reactions = pd.concat(
            [
                pd.DataFrame(reactions),
                pd.DataFrame([unbalance]).T.rename(columns={0: "Unbalance"}),
                pd.DataFrame([diff_formula]).T.rename(columns={0: "Diff_formula"}),
            ],
            axis=1,
        ).to_dict(orient="records")

        cbalanced_reactions = [
            _reactions[key]
            for key, value in enumerate(_reactions)
            if value["carbon_balance_check"] == "balanced"
        ]
        cunbalanced_reactions = [
            _reactions[key]
            for key, value in enumerate(_reactions)
            if value["carbon_balance_check"] != "balanced"
        ]
        rule_based_reactions = filter_data(
            cbalanced_reactions,
            unbalance_values=["Reactants", "Products"],
            formula_key="Diff_formula",
            element_key=None,
            min_count=0,
            max_count=0,
        )

        both_side_cbalanced_reactions = filter_data(
            cbalanced_reactions,
            unbalance_values=["Both"],
            formula_key="Diff_formula",
            element_key=None,
            min_count=0,
            max_count=0,
        )

        imp = SyntheticRuleImputer(
            rule_dict=self.rules, select="all", ranking="ion_priority"
        )
        balanced_cnt = (
            len(cbalanced_reactions)
            - len(rule_based_reactions)
            - len(both_side_cbalanced_reactions)
        )
        if stats is not None:
            stats["balanced_cnt"] = balanced_cnt
            stats["rb_applied"] = len(rule_based_reactions)
        logger.info("Input data contains {} balanced reactions.".format(balanced_cnt))
        logger.info(
            "Run rule-based method on {} reactions.".format(len(rule_based_reactions))
        )
        expected_result = imp.parallel_impute(rule_based_reactions, n_jobs=self.n_jobs)

        solve, unsolve = extract_results_by_key(expected_result)
        unsolve = cunbalanced_reactions + both_side_cbalanced_reactions + unsolve

        constrain = RuleConstraint(
            solve,
            ban_atoms=[
                "[O].[O]",
                "F-F",
                "Cl-Cl",
                "Br-Br",
                "I-I",
                "Cl-Br",
                "Cl-I",
                "Br-I",
            ],
        )
        certain_reactions, uncertain_reactions = constrain.fit()
        if stats is not None:
            stats["rb_solved"] = len(certain_reactions)

        for r in certain_reactions:
            _id = int(r[self.id_col])
            reactions[_id][self.output_col] = r["new_reaction"]  # type: ignore
        return reactions
