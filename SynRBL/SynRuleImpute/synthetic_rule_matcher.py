class SyntheticRuleMatcher:
    """
    A class to match rules based on given chemical data using a depth-first search algorithm.
    Implements backtracking to revert to previous states if a dead end is reached.
    Allows selection between finding the 'best' single solution or 'all' possible solutions.

    Example:
    rule_dict = [{'smiles': 'C1=CC=CC=C1', 'Composition': {'C': 6, 'H': 6}},
                 {'smiles': 'CCO', 'Composition': {'C': 2, 'H': 6, 'O': 1}}]
    data_dict = {'C': 6, 'H': 6,}
    matcher = RSMIRuleMatcher(rule_dict, data_dict, select='best', ranking=False)
    best_solution = matcher.match()
    print(best_solution)
    # Output: [{'smiles': 'C1=CC=CC=C1', 'Ratio': 1}]

    Attributes:
        rule_dict (list): A list of dictionaries representing chemical rules, each containing 'smiles' and 'Composition'.
        data_dict (dict): A dictionary representing available chemical data, with element symbols as keys and counts as values.
        select (str): Selection mode, either 'best' for the best solution or 'all' for all solutions.
        ranking (str or bool): Ranking mode for solutions, options are 'longest', 'least', 'greatliest', or False.

    Methods:
        match(): Find matching solutions based on the specified selection and ranking mode.
    """
    def __init__(self, rule_dict, data_dict, select='best', ranking=False):
        # Sort rules by composition length in descending order for efficient matching.
        self.rule_dict = sorted(rule_dict, key=lambda r: len(r['Composition']), reverse=True)
        self.data_dict = data_dict
        self.select = select
        if self.select == 'all':
            self.all_solutions = []

        self.ranking = ranking

        # Ensure 'Q' key exists in data_dict and remove elements with zero counts.
        if 'Q' not in self.data_dict:
            self.data_dict['Q'] = 0
        self.data_dict = {k: v for k, v in self.data_dict.items() if v != 0 or k == 'Q'}

    def match(self):
        """
        Find matching solutions based on the specified selection and ranking mode.

        Returns:
            list: List of matching solutions, each represented as a list of dictionaries.
        """
        if self.select == 'all':
            self.dfs(self.data_dict, [])
            self.all_solutions = self.remove_overlapping_solutions(self.all_solutions)
            return self.rank_solutions(self.all_solutions, self.ranking)
        else:
            solution = self.dfs(self.data_dict, [])
            return [solution] if solution is not None else []

    def dfs(self, data, path):
        """
        Depth-First Search (DFS) algorithm to find solutions by exploring possible paths.

        Args:
            data (dict): Current chemical data to match against.
            path (list): List of dictionaries representing the current path/solution.

        Returns:
            dict or None: A solution (path) if found, None otherwise.
        """
        if self.exit_strategy_solution(data):
            if self.select == 'all':
                self.all_solutions.append(path)
                return
            else:
                return path

        for rule in self.rule_dict:
            new_data, new_path = self.apply_rule(data, path, rule)
            if new_data is not None:
                result = self.dfs(new_data, new_path)
                if result is not None:
                    if self.select != 'all':
                        return result
        return None

    def apply_rule(self, data, path, rule):
        """
        Apply a chemical rule to the current data and path.

        Args:
            data (dict): Current chemical data to match against.
            path (list): List of dictionaries representing the current path/solution.
            rule (dict): Chemical rule to apply, containing 'Composition' and 'smiles'.

        Returns:
            dict or None: Updated chemical data and path if the rule can be applied, None otherwise.
        """
        if not self.can_match(rule['Composition'], data):
            return None, None

        ratio = abs(min((data[k] // v if v != 0 else 0) for k, v in rule['Composition'].items() if k != 'Q'))
        new_data = data.copy()
        for k, v in rule['Composition'].items():
            if k in new_data:
                new_data[k] -= v * ratio
                if new_data[k] == 0 and k != 'Q':
                    del new_data[k]

        new_path = path + [{'smiles': rule['smiles'], 'Ratio': ratio}]
        return new_data, new_path

    def can_match(self, rule, data):
        """
        Check if a chemical rule can be matched with the current data.

        Args:
            rule (dict): Chemical rule to apply, containing element symbols and counts.
            data (dict): Current chemical data to match against.

        Returns:
            bool: True if the rule can be matched, False otherwise.
        """
        return all(k in data and data[k] >= v for k, v in rule.items() if k != 'Q')

    @staticmethod
    def rank_solutions(solutions, ranking):
        """
        Rank a list of solutions based on the specified ranking mode.

        Args:
            solutions (list): List of solutions, each represented as a list of dictionaries.
            ranking (str or bool): Ranking mode for solutions.

        Returns:
            list: List of ranked solutions.
        """
        if ranking == 'longest':
            return sorted(solutions, key=lambda sol: -len(sol), reverse=True)
        elif ranking == 'least':
            return sorted(solutions, key=lambda sol: sum(len(item) for item in sol))
        elif ranking == 'greatliest':
            return sorted(solutions, key=lambda sol: sum(len(item) for item in sol), reverse=True)
        else:
            return solutions  # Default case, no sorting

    @staticmethod
    def exit_strategy_solution(data):
        """
        Check if the exit strategy condition for finding a solution is met.

        Args:
            data (dict): Current chemical data.

        Returns:
            bool: True if the exit strategy condition is met, False otherwise.
        """
        return len(data) == 1 and data.get('Q', 0) == 0

    @staticmethod
    def remove_overlapping_solutions(solutions):
        """
        Remove overlapping solutions from the list of solutions.

        Args:
            solutions (list): List of solutions, each represented as a list of dictionaries.

        Returns:
            list: List of unique solutions with overlapping solutions removed.
        """
        unique_solutions = []
        seen = set()

        for solution in solutions:
            # Convert each solution to a set of tuples for comparison
            solution_set = frozenset((item['smiles'], item['Ratio']) for item in solution)
            if solution_set not in seen:
                seen.add(solution_set)
                unique_solutions.append(solution)

        return unique_solutions