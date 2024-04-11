from typing import List, Tuple, Optional
from collections import Counter


class RefinementUncertainty:
    def __init__(self, finalgraph_uncertainty, graph_conditions: List[dict]):
        """
        Initialize the RefinementUncertainty class with the graph conditions.

        Parameters:
        finalgraph_uncertainty (List[dict]): A list of dictionaries
            representing missing graph data.
        graph_conditions (List[dict]): A list of dictionaries representing
            graph conditions.
        """
        self.finalgraph_uncertainty = finalgraph_uncertainty
        self.id = [i["R-id"] for i in finalgraph_uncertainty]
        self.graph_conditions = graph_conditions

    @staticmethod
    def get_smiles_graph(final_graph: List[dict], id: str) -> List[str]:
        """
        Retrieve the SMILES graph for a given final graph and ID.

        Parameters:
        final_graph (List[dict]): A list of dictionaries representing missing
            graph data.
        id (str): The ID to search for in the graph.

        Returns:
        List[str]: The SMILES graph for the given ID.
        """
        return [item["smiles"] for item in final_graph if item["R-id"] == id][0]

    @staticmethod
    def intersection_of_lists_with_count(
        lists: List[List[str]], intersection_num: int = 2
    ) -> Tuple[Optional[List[str]], Optional[int]]:
        """
        Find lists that are present at least a specified number of times in
        the provided list of lists and their first index.

        Parameters:
        lists (List[List[str]]): A list containing multiple lists of elements.
        intersection_num (int): The minimum number of times a list must appear
            to be considered.

        Returns:
        Tuple[Optional[List[str]], Optional[int]]: The first list that meets
            the intersection criteria and its first index, or None if no such
            list exists.
        """
        # Convert inner lists to tuples for hashing and count occurrences
        tuple_lists = [tuple(lst) for lst in lists]
        counts = Counter(tuple_lists)

        # Find the first list that meets the intersection criteria
        for index, lst in enumerate(tuple_lists):
            if counts[lst] >= intersection_num:
                return list(lst), index  # Convert tuple back to list for the result

        return None, None

    def fit(self, intersection_num: int = 2) -> List[dict]:
        """
        Generate a new graph for uncertain IDs based on the first intersection
        condition.

        Parameters:
        id_uncertainty (List[str]): A list of uncertain IDs.
        intersection_num (int): The minimum number of lists in which an element
            must appear to be considered.

        Returns:
        List[dict]: A new graph list for uncertain IDs.
        """
        new_graph_uncertain = []
        for id in self.id:
            list_cond = [
                self.get_smiles_graph(cond, id) for cond in self.graph_conditions
            ]
            intersection, first_key_index = self.intersection_of_lists_with_count(
                list_cond, intersection_num
            )
            if intersection and first_key_index is not None:
                match_cond = self.graph_conditions[first_key_index]
                new_graph = [value for value in match_cond if value["R-id"] == id]
                new_graph_uncertain.extend(new_graph)
            else:
                new_graph = [
                    value
                    for value in self.finalgraph_uncertainty
                    if value["R-id"] == id
                ]
                new_graph_uncertain.extend(new_graph)

        return new_graph_uncertain
