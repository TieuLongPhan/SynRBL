import re
import copy
from typing import List, Dict, Any
from joblib import Parallel, delayed
from synrbl.SynChemImputer.reduction_template import ReductionTemplate
import rdkit.RDLogger as RDLogger

RDLogger.DisableLog("rdApp.*")


class CurateReduction:

    @staticmethod
    def check_for_isolated_hydrogen(smiles: str) -> bool:

        pattern = r"\[H\](?![^[]*\])"
        return bool(re.search(pattern, smiles))

    @staticmethod
    def curate(
        reaction_dict: Dict[str, Any],
        reaction_column: str = "reactions",
        compound_template: Dict[str, Any] = None,
        all_templates: Dict = None,
        return_all: bool = False,
    ) -> Dict[str, Any]:

        new_reaction_dict = copy.deepcopy(reaction_dict)
        reactions = reaction_dict.get(reaction_column, [])
        # print(reactions)
        if not reactions:
            return reaction_dict  # Early return if no reactions are found

        # Process the first reaction for simplification
        curate_reaction = ReductionTemplate.reduction_template(
            reactions, compound_template, all_templates, return_all
        )
        new_reaction_dict["curated_reaction"] = curate_reaction
        new_reaction_dict["radical"] = CurateReduction.check_for_isolated_hydrogen(
            curate_reaction[0] if curate_reaction else ""
        )

        return new_reaction_dict

    @classmethod
    def parallel_curate(
        cls,
        reaction_list: List[Dict[str, Any]],
        reaction_column: str = "reactions",
        compound_template: Dict[str, Any] = None,
        all_templates: Dict = None,
        return_all: bool = False,
        n_jobs: int = 4,
        verbose: int = 1,
    ) -> List[Dict[str, Any]]:

        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(cls.curate)(
                reaction, reaction_column, compound_template, all_templates, return_all
            )
            for reaction in reaction_list
        )
        return results
