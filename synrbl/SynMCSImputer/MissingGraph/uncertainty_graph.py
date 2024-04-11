from typing import List, Dict


class GraphMissingUncertainty:
    """
    A class to handle uncertainty in graph data. It identifies uncertain
    elements in the graphs based on boundary conditions and fragment count.

    Attributes:
        missing_graph_list (List[Dict]): A list of graph data in dictionary
            format.
        threshold (int): A threshold value to determine uncertainty based on
            the number of fragments in a SMILES string.
    """

    def __init__(self, missing_graph_list: List[Dict], threshold: int = 2) -> None:
        """
        Initializes the GraphMissingUncertainty class with missing graph data
        and a threshold.

        Args:
            missing_graph_list (List[Dict]): The list of missing graph data.
            threshold (int, optional): The threshold for determining
                uncertainty based on fragments. Defaults to 2.
        """
        self.missing_graph_list = missing_graph_list
        self.threshold = threshold

    @staticmethod
    def check_boundary(data_list: List[Dict]) -> List[int]:
        """
        Identifies the indices of entries in the data list where
        'boundary_atoms_products' is empty.

        Args:
            data_list (List[Dict]): A list of dictionaries, each containing a
                key 'boundary_atoms_products'.

        Returns:
            List[int]: A list of indices where 'boundary_atoms_products' is
                empty.
        """
        without_boundary_key = []

        for key, item in enumerate(data_list):
            if all(
                element is None for element in item["boundary_atoms_products"]
            ):  # Checks if 'boundary_atoms_products' is empty
                without_boundary_key.append(key)

        return without_boundary_key

    @staticmethod
    def check_fragments(data_list: List[Dict], threshold: int = 2) -> List[int]:
        """
        Identifies the indices of entries in the data list where the number of
        SMILES string fragments meets or exceeds a specified threshold.

        Args:
            data_list (List[Dict]): A list of dictionaries, each containing a
                'smiles' key.
            threshold (int): The minimum number of fragments in a SMILES string
                to be considered uncertain.

        Returns:
            List[int]: A list of indices where the number of SMILES string
                fragments meets or exceeds the threshold.
        """
        graph_uncertain_key = []

        for key, entry in enumerate(data_list):
            for i in entry["smiles"]:
                if i is not None and len(i.split(".")) >= threshold:
                    graph_uncertain_key.append(key)

        return graph_uncertain_key

    def fit(self) -> List[Dict]:
        """
        Processes the missing graph data to update their 'Certainty' status
        based on boundary and fragment checks.

        Returns:
            List[Dict]: The updated list of missing graph data with 'Certainty'
                status marked.
        """
        uncertain_key = []
        without_boundary_key = self.check_boundary(self.missing_graph_list)
        graph_uncertain_key = self.check_fragments(
            self.missing_graph_list, threshold=self.threshold
        )

        uncertain_key.extend(without_boundary_key)
        uncertain_key.extend(graph_uncertain_key)

        for key, missing_graph in enumerate(self.missing_graph_list):
            missing_graph["Certainty"] = key not in uncertain_key

        return self.missing_graph_list
