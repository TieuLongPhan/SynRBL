import logging

from collections import Counter
from rdkit import Chem
from joblib import Parallel, delayed
from typing import List, Dict

logger = logging.getLogger("synrbl")


class ExtractMCS:
    """
    A class to extract and analyze the most common Maximum Common Substructure
    (MCS) from a list of MCS results.
    Provides functionality to determine the most common elements, the top n
    common elements, and to calculate the corrected individual overlap
    percentage for multiple conditions. Additionally, extracts the MCS results
    that meet a specified threshold for commonality.
    """

    def __init__(self):
        pass

    @staticmethod
    def get_num_atoms(smiles):
        """
        Calculate the number of atoms in a molecule represented by a SMILES
        string.

        Args:
        smiles (str): A string representing a molecule in SMILES format.

        Returns:
        int: The number of atoms in the molecule. Returns 0 if the SMILES
            string is invalid or an error occurs.
        """
        try:
            molecule = Chem.MolFromSmarts(smiles)
            if molecule is not None:
                return molecule.GetNumAtoms()
            else:
                return 0
        except Exception:
            return 0

    @staticmethod
    def calculate_total_number_atoms_mcs_parallel(condition, n_jobs=4) -> list[int]:
        """
        Calculate the total number of atoms in the MCS results for each
        dictionary in a condition using parallel processing.

        Args:
        condition (list): A list of dictionaries, each containing
            'mcs_results', which are lists of SMILES strings.
        n_jobs (int, optional): The number of jobs to run in parallel.
            Defaults to 4.

        Returns:
        list: A list containing the total number of atoms in the MCS results
            for each dictionary in the condition.
        """

        def calculate_atoms_for_dict(d):
            return sum(ExtractMCS.get_num_atoms(mcs) for mcs in d["mcs_results"])

        total_number_atoms = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(calculate_atoms_for_dict)(d) for d in condition
        )
        return total_number_atoms  # type: ignore

    def get_popular_elements_from_list(self, elements_list):
        """
        Get the most popular elements in a flat list.

        :param elements_list: A list containing elements.
        :return: A list of elements that appear with the highest frequency in
            the provided list.
        """
        element_count = Counter(elements_list)
        max_frequency = max(element_count.values(), default=0)
        return [
            element
            for element, count in element_count.items()
            if count == max_frequency
        ]

    def get_top_n_common_elements(self, elements_list, top_n=2):
        """
        Get the top n most common elements from a list of sets.

        :param elements_list: A list of sets, each containing elements.
        :param top_n: The number of top elements to retrieve.
        :return: A list of the top n most common elements across all sets.
        """
        flattened_elements = [
            element for element_set in elements_list for element in element_set
        ]
        return [
            element for element, _ in Counter(flattened_elements).most_common(top_n)
        ]

    def calculate_corrected_individual_overlap_percentage(self, *conditions):
        """
        Calculate the corrected individual overlap percentage across multiple
        conditions.

        :param conditions: A variable number of conditions, each a list of
            dictionaries containing 'mcs_results'.
        :return: A list of overlap percentages for each index across the
            conditions.
        :raises ValueError: If all conditions do not have the same number of
            cases.
        """
        if not all(len(condition) == len(conditions[0]) for condition in conditions):
            raise ValueError("All conditions must have the same number of cases")

        num_conditions = len(conditions)
        list_overlap_percentages = []
        reference_results_list = []

        for idx in range(len(conditions[0])):
            list_length = [
                len(condition[idx]["mcs_results"]) for condition in conditions
            ]
            len_popular = self.get_popular_elements_from_list(list_length)[0]
            current_results = [
                set(condition[idx]["mcs_results"]) for condition in conditions
            ]
            reference_results = self.get_top_n_common_elements(
                current_results, top_n=len_popular
            )
            reference_results_list.append(reference_results)
            overlap_count = sum(
                sorted(reference_results) == sorted(set(condition[idx]["mcs_results"]))
                for condition in conditions
            )
            overlap_percentage = (overlap_count / num_conditions) * 100
            list_overlap_percentages.append(overlap_percentage)

        return list_overlap_percentages, reference_results_list

    def extract_common_mcs_index(self, lower_threshold, upper_threshold, *conditions):
        """
        Extract MCS results that meet a specified threshold for commonality.

        :param threshold: The percentage threshold for commonality.
        :param conditions: A variable number of conditions, each a list of
            dictionaries containing 'mcs_results'.
        :return: A list of dictionaries representing the MCS results that meet
            the specified threshold.
        """
        (
            overlap_percentages,
            reference_results_list,
        ) = self.calculate_corrected_individual_overlap_percentage(*conditions)
        threshold_index = [
            lower_threshold <= i <= upper_threshold for i in overlap_percentages
        ]
        # mcs_common = [d for d, b in zip(conditions[0], threshold_index) if b]
        return threshold_index, reference_results_list

    @staticmethod
    def get_largest_condition(*conditions: List[List[Dict]]) -> List[Dict]:
        """
        Compare the total number of atoms across different conditions and find
        the condition with the largest MCS for each index.
        In case of a tie, compares the total number of atoms in the first
        SMILES/SMARTS for those conditions.

        Args:
        - total_atoms_conditions (list): A list of lists, where each sublist
            contains the total number of atoms for each MCS result in
            a condition.

        Returns:
            list[dict]: A list of conditions. Each row contains the condition
                with the largest maximum-common-substructure or None if no
                suitable result was found.
        """

        total_atoms_conditions = [
            ExtractMCS.calculate_total_number_atoms_mcs_parallel(condition, n_jobs=4)
            for condition in conditions
        ]

        result = []
        min_length = min(len(total) for total in total_atoms_conditions)

        for idx in range(min_length):
            tied_conditions = []
            max_atoms = 0
            max_condition_idx = -1
            max_condition = None

            # First pass: Find conditions with the largest total atom counts
            for condition_idx, total in enumerate(total_atoms_conditions):
                if idx < len(total):
                    if total[idx] > max_atoms:
                        max_atoms = total[idx]
                        tied_conditions = [(condition_idx, total[idx])]
                    elif total[idx] == max_atoms:
                        tied_conditions.append((condition_idx, total[idx]))

            # Second pass: In case of a tie, compare the first SMILES/SMARTS
            if len(tied_conditions) > 1:
                max_first_smarts_atoms = 0
                winning_condition_idx = -1
                for condition_idx, _ in tied_conditions:
                    first_smarts = (
                        conditions[condition_idx][idx]["mcs_results"][0]
                        if conditions[condition_idx][idx]["mcs_results"]
                        else ""
                    )
                    first_smarts_atoms = ExtractMCS.get_num_atoms(first_smarts)
                    if first_smarts_atoms > max_first_smarts_atoms:
                        max_first_smarts_atoms = first_smarts_atoms
                        winning_condition_idx = condition_idx

                # Finalize the winning condition
                if winning_condition_idx != -1:
                    max_condition_idx = winning_condition_idx
                    max_condition = conditions[max_condition_idx][idx]
            else:
                max_condition_idx = tied_conditions[0][0] if tied_conditions else -1
                max_condition = (
                    conditions[max_condition_idx][idx]
                    if max_condition_idx != -1
                    else None
                )

            if max_condition is not None:
                result.append(max_condition)

        return result
