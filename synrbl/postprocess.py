from synrbl.SynProcessor import RSMIComparator, RSMIDecomposer, CheckCarbonBalance
from synrbl.SynUtils.common import update_reactants_and_products


class Validator:
    def __init__(
        self,
        reaction_col,
        method,
        solved_col="solved",
        solved_method_col="solved_by",
        unbalance_col="unbalance",
        carbon_balance_col="carbon_balance_check",
        issue_col="issue",
        check_carbon_balance=True,
        n_jobs=1,
    ):
        self.reaction_col = reaction_col
        self.method = method
        self.solved_col = solved_col
        self.solved_method_col = solved_method_col
        self.unbalance_col = unbalance_col
        self.check_carbon_balance = check_carbon_balance
        self.carbon_balance_col = carbon_balance_col
        self.issue_col = issue_col
        self.n_jobs = n_jobs

    def check(self, reactions, override_unsolved=False, override_issue_msg=None):
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
        unbalance, _ = comp.run_parallel(reactants=react_dict, products=product_dict)

        if self.check_carbon_balance:
            check = CheckCarbonBalance(
                reactions,
                rsmi_col=self.reaction_col,
                symbol=">>",
                atom_type="C",
                n_jobs=self.n_jobs,
            )
            for i, r in enumerate(check.check_carbon_balance()):
                reactions[i][self.carbon_balance_col] = r["carbon_balance_check"]

        assert len(reactions) == len(unbalance)
        for reaction, b in zip(reactions, unbalance):
            reaction[self.unbalance_col] = b
            if (
                b == "Balance"
                and reaction[self.carbon_balance_col] == "balanced"
                and not reaction[self.solved_col]
            ):
                reaction[self.solved_col] = True
                reaction[self.solved_method_col] = self.method
            if override_unsolved and not reaction[self.solved_col]:
                reaction[self.reaction_col] = reaction["input_reaction"]
                if override_issue_msg is not None and reaction[self.issue_col] == "":
                    reaction[self.issue_col] = override_issue_msg
        return reactions
