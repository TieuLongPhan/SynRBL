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


class CurationOxidation:
    def __init__(
        self,
        compounds_template: Dict = compounds_template,
        reaction_templates: Dict = reaction_templates,
    ):
        self.compounds_template = compounds_template
        self.reaction_templates = reaction_templates

    @staticmethod
    def find_oxidation_pattern(reaction_smiles: str) -> List[str]:
        """
        Determines the predominant oxidation pattern from a SMILES string
            by identifying functional group changes.

        Parameters:
        reaction_smiles (str): SMILES string representing the chemical reaction.

        Returns:
        List[str]: A list containing the identified oxidation pattern
                    described as 'reactant>>product'.
        """
        reactant_fg, product_fg = find_functional_reactivity(reaction_smiles)
        if reactant_fg and product_fg:
            return [f"{reactant_fg[0]}>>{product_fg[0]}"]
        return []

    @staticmethod
    def process_ox_template(
        reaction_smiles: str, compounds_template: Dict, reaction_templates: Dict
    ) -> Tuple[List[str], List[Optional[bool]]]:
        """
        Processes an oxidation template based on the given SMILES string of the reaction.

        Parameters:
        reaction_smiles (str): The reaction SMILES string
        compounds_template (Dict): A dictionary containing compounds templates
        reaction_templates (Dict): A dictionary containing reaction templates

        Returns:
        Tuple[str, Optional[bool]]: A tuple containing the modified SMILES string
            and a boolean indicating if the process was stoichiometric.
        """
        reaction_list = []
        stoichiometry_list = []
        try:
            cp_temp = CurationOxidation.find_oxidation_pattern(reaction_smiles)[0]
            temps = compounds_template["oxidation"].get(
                cp_temp, compounds_template["oxidation"]["other"]
            )
        except IndexError:
            # print("No oxidation pattern found.")
            return [reaction_smiles], [None]

        reactant, product = reaction_smiles.split(">>")
        o_count = count_radical_atoms(reactant, 8)
        reactant = [x for x in reactant.split(".") if x != "[O]"]
        product = product.split(".")
        if len(temps) == 0:
            return [reaction_smiles], [None]

        for temp in temps:
            if cp_temp in [
                "primary_alcohol>>aldehyde",
                "secondary_alcohol>>ketone",
                "aldehyde>>carboxylic_acid",
            ]:
                for _ in range(o_count):
                    reactant.extend(reaction_templates["oxidation"][temp]["reactants"])
                    product.extend(reaction_templates["oxidation"][temp]["products"])
                    stoichiometry = reaction_templates["oxidation"][temp][
                        "stoichiometric"
                    ]
            elif cp_temp == "primary_alcohol>>carboxylic_acid":
                reactant.extend(reaction_templates["oxidation"][temp]["reactants"])
                product.extend(reaction_templates["oxidation"][temp]["products"])
                stoichiometry = reaction_templates["oxidation"][temp]["stoichiometric"]
            else:
                return [reaction_smiles], [None]

            reactant = ".".join(reactant)
            product = ".".join(product)
            rsmi = f"{reactant}>>{product}"
            reaction_list.append(rsmi)
            stoichiometry_list.append(stoichiometry)

        return reaction_list, stoichiometry_list

    @staticmethod
    def process_dict(
        reaction_dict: Dict,
        reaction_columns: str = "reactions",
        compounds_template: Dict = None,
        reaction_templates: Dict = None,
        return_all: bool = False,
    ) -> Dict:
        """
        Processes a single reaction dictionary and updates it with the results of
                    oxidation reaction curation.

        Parameters:
        reaction_dict (Dict): The dictionary containing the reaction data.
        reaction_columns (str): The key where the reaction SMILES string is stored.
                                Defaults to 'reactions'.
        compounds_template (Dict, optional): A dictionary of compounds templates.
        reaction_templates (Dict, optional): A dictionary of reaction templates.
        return_all (bool): A flag to determine if all results or only the first result
                            should be returned.

        Returns:
        Dict: The updated dictionary with additional fields for curated reaction,
                stoichiometry, and radical presence.
        """
        reaction = reaction_dict[reaction_columns]
        new_reaction, stoichiometry = CurationOxidation.process_ox_template(
            reaction, compounds_template, reaction_templates
        )
        if len(new_reaction) == 0 or len(stoichiometry) == 0:
            return reaction_dict
        reaction_dict["curated_reaction"] = (
            new_reaction if return_all else new_reaction[0]
        )
        reaction_dict["stoichiometric"] = (
            stoichiometry if return_all else stoichiometry[0]
        )
        reaction_dict["radical"] = check_for_isolated_atom(new_reaction[0], "O")
        return reaction_dict

    def parallel_curate(
        self,
        data: List[Dict],
        reaction_col="reactions",
        n_jobs: int = 4,
        verbose: int = 1,
        return_all: bool = False,
    ) -> List[Dict]:
        """
        Curates a list of reaction dictionaries in parallel, updating each
                    with oxidation reaction curation results.

        Parameters:
        data (List[Dict]): A list of dictionaries, each containing
                            reaction data to be curated.
        n_jobs (int): The number of parallel jobs to run. Defaults to 4.
        verbose (int): The verbosity level for parallel processing. Defaults to 1.
        return_all (bool): A flag to determine if all results or only the first result
                            should be returned for each reaction.

        Returns:
        List[Dict]: A list of the curated dictionaries, each updated with
                    additional fields for curated reaction,
                    stoichiometry, and radical presence.
        """
        results = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(CurationOxidation.process_dict)(
                reaction,
                reaction_col,
                self.compounds_template,
                self.reaction_templates,
                return_all=return_all,
            )
            for reaction in data
        )
        return results
