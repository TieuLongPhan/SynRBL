from rdkit import Chem
from typing import Dict, List
from synrbl.SynChemImputer.curate_oxidation import CurationOxidation
from synrbl.SynChemImputer.curate_reduction import CurationReduction
from joblib import Parallel, delayed


class PostProcess:

    def __init__(self, data: List[dict]):
        self.data = data

    @staticmethod
    def label_reactions(
        reaction_dict: Dict, id_column: str = "R-id", reaction_column: str = "reactions"
    ) -> Dict:
        """
        Labels chemical reactions based on their reactants, indicating whether they
        are oxidation or reduction reactions, and canonicalizes the SMILES strings.

        Parameters:
        - reaction_list (List[Dict]): A list of dictionaries, each representing a reaction
          with keys 'R-id' and 'new_reaction'.

        Returns:
        - List[Dict]: A list of dictionaries, each augmented with a 'label', 'reactants',
          and 'products' keys, where 'reactants' and 'products' are canonicalized SMILES.
        """

        label = "unspecified"
        r_id = reaction_dict.get("R-id", "N/A")
        new_reaction = reaction_dict.get(reaction_column, "")

        try:
            reactants, products = new_reaction.split(">>", 1)
        except ValueError:
            reactants, products = "", ""

        labeling_criteria = {
            ".[O]": "Oxidation",
            ".[H]": "Reduction",
        }

        for marker, reaction_label in labeling_criteria.items():
            if marker in reactants:
                label = reaction_label
                break

        reactants_smiles = Chem.CanonSmiles(reactants) if reactants else ""
        products_smiles = Chem.CanonSmiles(products) if products else ""

        new_dict = {
            id_column: r_id,
            reaction_column: new_reaction,
            "label": label,
            "reactants": reactants_smiles,
            "products": products_smiles,
        }

        return new_dict

    def fit(self, n_jobs=4, verbose=1):
        label_data = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(PostProcess.label_reactions)(d) for d in self.data
        )

        reduction_data = [
            value for value in label_data if value["label"] == "Reduction"
        ]
        oxidation_data = [
            value for value in label_data if value["label"] == "Oxidation"
        ]

        curate_reduction = CurationReduction()
        curate_oxidation = CurationOxidation()
        result_reduction = curate_reduction.parallel_curate(
            reduction_data, n_jobs=n_jobs, verbose=verbose
        )
        result_oxidation = curate_oxidation.parallel_curate(
            oxidation_data, n_jobs=n_jobs, verbose=verbose
        )
        return result_reduction, result_oxidation