from rdkit import Chem
from typing import Dict, List
from synrbl.SynChemImputer.curate_oxidation import CurationOxidation
from synrbl.SynChemImputer.curate_reduction import CurationReduction
from joblib import Parallel, delayed


class PostProcess:
    def __init__(
        self,
        id_col="R-id",
        reaction_col="reactions",
        n_jobs: int = 4,
        verbose: int = 1,
        use_default: bool = False,
    ):
        self.id_col = id_col
        self.reaction_col = reaction_col
        self.n_jobs = n_jobs
        self.verbose = verbose
        self.use_default = use_default

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
        r_id = reaction_dict[id_column]
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
        try:
            reactants_smiles = Chem.CanonSmiles(reactants)
            products_smiles = Chem.CanonSmiles(products)
        except Exception as e:
            print(f"An error occurred in smiles parsing in PostProcess: {e}")
            reactants_smiles = reactants
            products_smiles = products

        new_dict = {
            id_column: r_id,
            reaction_column: new_reaction,
            "label": label,
            "reactants": reactants_smiles,
            "products": products_smiles,
        }

        return new_dict

    def fit(self, data) -> List[Dict]:
        """
        Label reactions and curate data by reaction type.

        Parameters:
        - n_jobs (int): Number of CPUs to use for parallel processing.
        - verbose (int): Level of verbosity.

        Returns:
        - List[Dict]: List of dictionaries, each representing a reaction and with
          keys 'R-id', 'new_reaction', 'label', 'reactants', and 'products'.
        """
        label_data = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(PostProcess.label_reactions)(d, self.id_col, self.reaction_col)
            for d in data
        )

        reduction_data = [
            value for value in label_data if value["label"] == "Reduction"
        ]
        oxidation_data = [
            value for value in label_data if value["label"] == "Oxidation"
        ]
        other_data = [value for value in label_data if value["label"] == "unspecified"]

        curate_reduction = CurationReduction()
        curate_oxidation = CurationOxidation()
        result_reduction = curate_reduction.parallel_curate(
            reduction_data,
            self.reaction_col,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_all=False,
            use_default=self.use_default,
        )
        result_oxidation = curate_oxidation.parallel_curate(
            oxidation_data,
            self.reaction_col,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_all=False,
        )
        other_data.extend(result_reduction)
        other_data.extend(result_oxidation)
        return other_data
