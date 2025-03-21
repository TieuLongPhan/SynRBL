from synrbl.SynUtils.data_utils import find_shortest_sublists
from synrbl.SynUtils.chem_utils import calculate_net_charge
from typing import List, Dict, Any
from typing import Union


class SyntheticRuleMatcher:
    """
    A class to match rules based on given chemical data using a depth-first
    search algorithm. Implements backtracking to revert to previous states if a
    dead end is reached. Allows selection between finding the 'best' single
    solution or 'all' possible solutions.

    Example:
    rule_dict = [{'smiles': 'C1=CC=CC=C1', 'Composition': {'C': 6, 'H': 6}},
                 {'smiles': 'CCO', 'Composition': {'C': 2, 'H': 6, 'O': 1}}]
    data_dict = {'C': 6, 'H': 6,}
    matcher = RSMIRuleMatcher(rule_dict, data_dict, select='best', ranking=False)
    best_solution = matcher.match()
    print(best_solution)
    # Output: [{'smiles': 'C1=CC=CC=C1', 'Ratio': 1}]

    Attributes:
        rule_dict (list): A list of dictionaries representing chemical rules,
            each containing 'smiles' and 'Composition'.
        data_dict (dict): A dictionary representing available chemical data,
            with element symbols as keys and counts as values.
        select (str): Selection mode, either 'best' for the best solution or
            'all' for all solutions.
        ranking (str or bool): Ranking mode for solutions, options are
            'longest', 'least', 'greatliest', or False.

    Methods:
        match(): Find matching solutions based on the specified selection and
            ranking mode.
    """

    def __init__(
        self,
        rule_dict: List[Dict[str, Any]],
        data_dict: Dict[str, int],
        select: str = "best",
        ranking: bool = False,
    ) -> None:
        """
        Initialize the class.

        Args:
            rule_dict: A list of dictionaries representing rules.
            data_dict: A dictionary with data.
            select: A string indicating the selection method.
            ranking: A boolean indicating whether ranking is enabled.
        """
        # Sort rules by composition length in descending order for efficient matching.
        self.rule_dict = sorted(
            rule_dict, key=lambda r: len(r["Composition"]), reverse=True
        )
        self.data_dict = data_dict
        self.select = select
        if self.select == "all":
            self.all_solutions = []
        self.ranking = ranking

        # Ensure 'Q' key exists in data_dict and remove elements with zero counts.
        if "Q" not in self.data_dict:
            self.data_dict["Q"] = 0
        self.data_dict = {k: v for k, v in self.data_dict.items() if v != 0 or k == "Q"}

    def match(self) -> List[List[Dict[str, Any]]]:
        """Find matching solutions based on the specified selection and ranking
        mode.

        Args:
            self (obj): The instance of the class.

        Returns:
            List[List[Dict[str, Any]]]: List of matching solutions, each
                represented as a list of dictionaries.
        """
        if self.select == "all":
            self.dfs(self.data_dict, [])
            self.all_solutions = self.remove_overlapping_solutions(self.all_solutions)
            return self.rank_solutions(self.all_solutions, self.ranking)
        else:
            solution = self.dfs(self.data_dict, [])
            return [solution] if solution is not None else []

    def dfs(self, data: dict, path: list) -> Union[dict, None]:
        """
        Depth-First Search (DFS) algorithm to find solutions by exploring
        possible paths.

        Args:
            data (dict): Current chemical data to match against.
            path (list): List of dictionaries representing the current
                path/solution.

        Returns:
            Union[dict, None]: A solution (path) if found, None otherwise.
        """
        if self.exit_strategy_solution(data):
            if self.select == "all":
                self.all_solutions.append(path)
                return None
            else:
                return path

        for rule in self.rule_dict:
            new_data, new_path = self.apply_rule(data, path, rule)
            if new_data is not None:
                result = self.dfs(new_data, new_path)
                if result is not None:
                    if self.select != "all":
                        return result
        return None

    def apply_rule(self, data: dict, path: list, rule: dict) -> tuple[dict, list]:
        """
        Apply a chemical rule to the current data and path.

        Args:
            data: Current chemical data to match against.
            path: List of dictionaries representing the current path/solution.
            rule: Chemical rule to apply, containing 'Composition' and 'smiles'.

        Returns:
            A tuple containing the updated chemical data and path if the rule
            can be applied, None otherwise.
        """
        if not self.can_match(rule["Composition"], data):
            return None, None

        ratio = abs(
            min(
                (data[k] // v if v != 0 else 0)
                for k, v in rule["Composition"].items()
                if k != "Q"
            )
        )
        new_data = data.copy()
        for k, v in rule["Composition"].items():
            if k in new_data:
                new_data[k] -= v * ratio
                if new_data[k] == 0 and k != "Q":
                    del new_data[k]

        new_path = path + [{"smiles": rule["smiles"], "Ratio": ratio}]
        return new_data, new_path

    def can_match(self, rule: dict, data: dict) -> bool:
        """
        Check if a chemical rule can be matched with the current data.

        Args:
            rule: Chemical rule to apply, containing element symbols and counts.
            data: Current chemical data to match against.

        Returns:
            True if the rule can be matched, False otherwise.
        """
        # Iterate over each key-value pair in the rule dictionary
        # Check if the key is in the data dictionary and if the corresponding
        # value in the data dictionary is greater than or equal to the value in
        # the rule dictionary
        # Exclude the key 'Q' from the check
        # Return True if all the conditions are met, False otherwise
        return all(k in data and data[k] >= v for k, v in rule.items() if k != "Q")

    @staticmethod
    def rank_solutions(
        solutions: List[List[Dict[str, Any]]], ranking: Union[str, bool]
    ) -> List[List[Dict[str, Any]]]:
        """
        Rank a list of solutions based on the specified ranking mode.

        Args:
            solutions: List of solutions, each represented as a list of
                dictionaries.
            ranking: Ranking mode for solutions.

        Returns:
            List of ranked solutions.
        """
        if ranking == "longest":
            return sorted(solutions, key=lambda sol: -len(sol), reverse=True)
        elif ranking == "least":
            return sorted(solutions, key=lambda sol: sum(len(item) for item in sol))
        elif ranking == "greatest":
            return sorted(
                solutions, key=lambda sol: sum(len(item) for item in sol), reverse=True
            )
        elif ranking == "ion_priority":
            shortest_sublists = find_shortest_sublists(solutions)
            return sorted(shortest_sublists, key=calculate_net_charge, reverse=True)
        else:
            return solutions

    @staticmethod
    def exit_strategy_solution(data: dict) -> bool:
        """
        Check if the exit strategy condition for finding a solution is met.

        Args:
            data (dict): Current chemical data.

        Returns:
            bool: True if the exit strategy condition is met, False otherwise.
        """
        return len(data) == 1 and data.get("Q", 0) == 0

    @staticmethod
    def remove_overlapping_solutions(
        solutions: List[List[Dict[str, str]]],
    ) -> List[List[Dict[str, str]]]:
        """
        Remove overlapping solutions from the list of solutions.

        Args:
            solutions: List of solutions, each represented as a list of
                dictionaries. Each dictionary should have 'smiles' and 'Ratio'
                keys with string values.

        Returns:
            List of unique solutions with overlapping solutions removed.
        """
        unique_solutions = []
        seen = set()

        for solution in solutions:
            # Convert each solution to a set of tuples for comparison
            solution_set = frozenset(
                (item["smiles"], item["Ratio"]) for item in solution
            )
            if solution_set not in seen:
                seen.add(solution_set)
                unique_solutions.append(solution)

        return unique_solutions
