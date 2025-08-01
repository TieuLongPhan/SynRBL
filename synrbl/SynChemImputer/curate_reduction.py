from typing import Dict, List, Tuple, Optional
from synrbl.SynUtils.chem_utils import (
    find_functional_reactivity,
    check_for_isolated_atom,
    count_radical_atoms,
)
from joblib import Parallel, delayed
import synrbl.SynChemImputer
from synkit.IO.data_io import load_database
import rdkit.RDLogger as RDLogger
import importlib.resources

RDLogger.DisableLog("rdApp.*")


compounds_template = load_database(
    importlib.resources.files(synrbl.SynChemImputer).joinpath("compounds_template.json")
)

reaction_templates = load_database(
    importlib.resources.files(synrbl.SynChemImputer).joinpath("reaction_template.json")
)


class CurationReduction:
    def __init__(
        self,
        compounds_template: Dict = compounds_template,
        reaction_templates: Dict = reaction_templates,
    ):
        self.compounds_template = compounds_template
        self.reaction_templates = reaction_templates

    @staticmethod
    def find_reduction_pattern(reaction_smiles: str) -> List[str]:
        """
        Determines the predominant oxidation pattern from a SMILES string
                    by identifying functional group changes.

        Parameters:
        reaction_smiles (str): SMILES string representing the chemical reaction.

        Returns:
        List[str]: A list containing the identified oxidation pattern
                    described as 'reactant>>product'.
        """
        reactant_fg, _ = find_functional_reactivity(reaction_smiles)
        if reactant_fg:
            return reactant_fg
        return []

    @staticmethod
    def process_reduct_template(
        reaction_smiles: str,
        compounds_template: Dict,
        reaction_templates: Dict,
        neutralize: bool = False,
        use_default: bool = False,
    ) -> Tuple[List[str], List[Optional[bool]]]:
        """
        Processes an oxidation template based on the given SMILES string of the reaction.

        Parameters:
        reaction_smiles (str): The SMILES string representing the reaction.
        compounds_template (Dict): A dictionary containing compounds templates.
        reaction_templates (Dict): A dictionary containing reaction templates.

        Returns:
        Tuple[str, Optional[bool]]: A tuple containing the modified SMILES string and
                                    a boolean indicating if the process was stoichiometric
        """
        reaction_list = []
        stoichiometry_list = []
        if use_default:
            compounds_template = compounds_template["reduction_default"]
        else:
            compounds_template = compounds_template["reduction"]
        reaction_templates = reaction_templates["reduction"]
        try:
            cp_temp = CurationReduction.find_reduction_pattern(reaction_smiles)[0]
            temps = compounds_template.get(cp_temp, compounds_template["other"])

            if len(temps) == 0:
                return [reaction_smiles], [None]
        except IndexError:
            # print("No reduction pattern found.")
            temps = compounds_template["other"]
            # return [reaction_smiles], [None]
        reactant, product = reaction_smiles.split(">>")
        h_count = count_radical_atoms(reactant, 1)
        if h_count % 2 != 0:
            return [reaction_smiles], [None]
        reactant = [x for x in reactant.split(".") if x != "[H]"]
        product = product.split(".")

        for temp in temps:
            reactant_copy = reactant.copy()
            product_copy = product.copy()
            hh_count = h_count // 2
            template_type = "neutral" if neutralize else "ion"
            for _ in range(hh_count):
                reactant_copy.extend(
                    reaction_templates[temp][template_type]["reactants"]
                )
                product_copy.extend(reaction_templates[temp][template_type]["products"])
                stoichiometry_list.append(
                    reaction_templates[temp][template_type]["stoichiometric"]
                )
            updated_reactants = ".".join(reactant_copy)
            updated_products = ".".join(product_copy)
            curated_reaction = f"{updated_reactants}>>{updated_products}"
            # print(curated_reaction)
            reaction_list.append(curated_reaction)
        return reaction_list, stoichiometry_list

    @staticmethod
    def process_dict(
        reaction_dict: Dict,
        reaction_columns: str = "reactions",
        compounds_template: Dict = None,
        reaction_templates: Dict = None,
        return_all: bool = False,
        neutralize: bool = False,
        use_default: bool = False,
    ) -> Dict:
        """
        Processes a single reaction dictionary and updates it with the results of
                oxidation reaction curation.

        Parameters:
        reaction_dict (Dict): The dictionary containing the reaction data.
        reaction_columns (str): The key where the reaction SMILES string is stored.
                                Defaults to 'reactions'.
        compounds_template (Dict, optional): A dictionary of compounds templates
        reaction_templates (Dict, optional): A dictionary of reaction templates
        return_all (bool): A flag to determine if all results
                or only the first result should be returned.

        Returns:
        Dict: The updated dictionary with additional fields for curated reaction,
                    stoichiometry, and radical presence.
        """
        reaction = reaction_dict[reaction_columns]
        new_reaction, stoichiometry = CurationReduction.process_reduct_template(
            reaction, compounds_template, reaction_templates, neutralize, use_default
        )
        if len(new_reaction) == 0 or len(stoichiometry) == 0:
            return reaction_dict
        reaction_dict["curated_reaction"] = (
            new_reaction if return_all else new_reaction[0]
        )
        reaction_dict["stoichiometric"] = (
            stoichiometry if return_all else stoichiometry[0]
        )
        reaction_dict["radical"] = check_for_isolated_atom(new_reaction[0], "H")
        return reaction_dict

    def parallel_curate(
        self,
        data: List[Dict],
        reaction_col="reactions",
        n_jobs: int = 4,
        verbose: int = 1,
        return_all: bool = False,
        neutralize: bool = False,
        use_default: bool = False,
    ) -> List[Dict]:
        """
        Curates a list of oxidation reaction dictionaries in parallel.

        Parameters:
        data (List[Dict]): A list of dictionaries
        n_jobs (int): The number of parallel jobs to run. Defaults to 4.
        verbose (int): The verbosity level for parallel processing. Defaults to 1.
        return_all (bool): A flag to determine if all results
                        or only the first result should be returned for each reaction.

        Returns:
        List[Dict]: A list of the curated dictionaries, each updated
                    with additional fields for curated reaction,
                    stoichiometry, and radical presence.
        """
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(CurationReduction.process_dict)(
                reaction,
                reaction_col,
                self.compounds_template,
                self.reaction_templates,
                return_all=return_all,
                neutralize=neutralize,
                use_default=use_default,
            )
            for reaction in data
        )
        return results
