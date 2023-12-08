from collections import Counter
import pandas as pd

class ExtractMCS:
    """
    A class to extract and analyze the most common Maximum Common Substructure (MCS) from a list of MCS results.
    Provides functionality to determine the most common elements, the top n common elements, and to calculate
    the corrected individual overlap percentage for multiple conditions. Additionally, extracts the MCS results
    that meet a specified threshold for commonality.
    """

    def get_popular_elements_from_list(self, elements_list):
        """
        Get the most popular elements in a flat list.

        :param elements_list: A list containing elements.
        :return: A list of elements that appear with the highest frequency in the provided list.
        """
        element_count = Counter(elements_list)
        max_frequency = max(element_count.values(), default=0)
        return [element for element, count in element_count.items() if count == max_frequency]

    def get_top_n_common_elements(self, elements_list, top_n=2):
        """
        Get the top n most common elements from a list of sets.

        :param elements_list: A list of sets, each containing elements.
        :param top_n: The number of top elements to retrieve.
        :return: A list of the top n most common elements across all sets.
        """
        flattened_elements = [element for element_set in elements_list for element in element_set]
        return [element for element, _ in Counter(flattened_elements).most_common(top_n)]

    def calculate_corrected_individual_overlap_percentage(self, *conditions):
        """
        Calculate the corrected individual overlap percentage across multiple conditions.

        :param conditions: A variable number of conditions, each a list of dictionaries containing 'mcs_results'.
        :return: A list of overlap percentages for each index across the conditions.
        :raises ValueError: If all conditions do not have the same number of cases.
        """
        if not all(len(condition) == len(conditions[0]) for condition in conditions):
            raise ValueError("All conditions must have the same number of cases")

        num_conditions = len(conditions)
        list_overlap_percentages = []

        for idx in range(len(conditions[0])):
            list_length = [len(condition[idx]['mcs_results']) for condition in conditions]
            len_popular = self.get_popular_elements_from_list(list_length)[0]
            current_results = [set(condition[idx]['mcs_results']) for condition in conditions]
            reference_results = self.get_top_n_common_elements(current_results, top_n=len_popular)

            overlap_count = sum(sorted(reference_results) == sorted(set(condition[idx]['mcs_results'])) for condition in conditions)
            overlap_percentage = (overlap_count / num_conditions) * 100
            list_overlap_percentages.append(overlap_percentage)

        return list_overlap_percentages

    def extract_common_mcs(self, threshold, *conditions):
        """
        Extract MCS results that meet a specified threshold for commonality.

        :param threshold: The percentage threshold for commonality.
        :param conditions: A variable number of conditions, each a list of dictionaries containing 'mcs_results'.
        :return: A list of dictionaries representing the MCS results that meet the specified threshold.
        """
        overlap_percentages = self.calculate_corrected_individual_overlap_percentage(*conditions)
        threshold_index = [i >= threshold for i in overlap_percentages]
        mcs_common = [d for d, b in zip(conditions[0], threshold_index) if b]
        return mcs_common, threshold_index

