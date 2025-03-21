import pandas as pd

from synkit.IO.data_io import load_database
from synrbl.SynAnalysis.analysis_utils import (
    remove_atom_mapping_from_reaction_smiles,
    calculate_chemical_properties,
    count_boundary_atoms_products_and_calculate_changes,
)
from IPython.display import clear_output


class AnalysisProcess:
    def __init__(self, list_data, pipeline_path, data_path):
        self.list_data = list_data
        self.pipeline_path = pipeline_path
        self.data_path = data_path

    def process_and_combine_datasets(self, remove_undetected=True):
        data_all = pd.DataFrame()

        for data_name in self.list_data:
            data_csv_path = (
                f"{self.pipeline_path}/Validation/Analysis/SynRBL - {data_name}.csv"
            )
            data = pd.read_csv(data_csv_path).drop(["Note"], axis=1)
            data.loc[data["Result"] == "CONSIDER", "Result"] = False
            data.loc[data["Result"] == "FALSE", "Result"] = False
            data.loc[data["Result"] == "TRUE", "Result"] = True

            merge_data_path = (
                f"{self.data_path}/Validation_set/{data_name}/MCS/MCS_Impute.json.gz"
            )
            mcs_data_path = (
                f"{self.data_path}/Validation_set/{data_name}"
                + "/mcs_based_reactions.json.gz"
            )

            merge_data = load_database(merge_data_path)
            merge_data = count_boundary_atoms_products_and_calculate_changes(merge_data)
            mcs_data = load_database(mcs_data_path)
            id = [value["R-id"] for value in merge_data]
            mcs_data = [value for value in mcs_data if value["R-id"] in id]
            mcs_data = calculate_chemical_properties(mcs_data)
            clear_output(wait=False)

            combined_data = pd.concat(
                [
                    pd.DataFrame(mcs_data)[
                        [
                            "R-id",
                            "reactions",
                            "carbon_difference",
                            "fragment_count",
                            "total_carbons",
                            "total_bonds",
                            "total_rings",
                        ]
                    ],
                    data,
                    pd.DataFrame(merge_data)[
                        [
                            "mcs_carbon_balanced",
                            "num_boundary",
                            "ring_change_merge",
                            "bond_change_merge",
                        ]
                    ],
                ],
                axis=1,
            )
            combined_data.loc[
                (combined_data["mcs_carbon_balanced"] is False)
                & (combined_data["Result"] is True),
                "Result",
            ] = False
            if remove_undetected:
                combined_data = combined_data[
                    combined_data["mcs_carbon_balanced"] is True
                ]

            data_all = pd.concat([data_all, combined_data], axis=0)
        data_all = data_all.reset_index(drop=True)
        unnamed_columns = [col for col in data_all.columns if "Unnamed" in col]
        data_all = data_all.drop(unnamed_columns, axis=1)

        return data_all

    def bin_value(self, value, bin_size=10):
        binned_value = (value // bin_size) * bin_size
        return binned_value

    def standardize_columns(self, data):
        data.loc[data["carbon_difference"] > 9, "carbon_difference"] = ">9"
        data.loc[data["fragment_count"] > 8, "fragment_count"] = ">8"
        data.loc[data["Bond Changes"] > 5, "Bond Changes"] = ">5"
        data.loc[data["bond_change_merge"] > 3, "bond_change_merge"] = ">3"
        data.loc[data["num_boundary"] > 3, "num_boundary"] = ">3"
        data["reactions"] = data["reactions"].apply(
            lambda x: remove_atom_mapping_from_reaction_smiles(x)
        )
        data = data.drop_duplicates(subset=["reactions"])
        data["Result"] = data["Result"].astype("bool")
        return data
