class RSMIRuleMatcher:
    """
    A class to match rules based on given data using a depth-first search algorithm.
    Implements backtracking to revert to previous states if a dead end is reached.
    """
    def __init__(self, rule_dict, data_dict):
        """
        Initializes the matcher with rules and data.
        """
        self.rule_dict = sorted(rule_dict, key=lambda r: len(r['Composition']), reverse=True)
        self.data_dict = data_dict

        if 'Q' not in self.data_dict:
            self.data_dict['Q'] = 0

        self.data_dict = {k: v for k, v in self.data_dict.items() if v != 0 or k == 'Q'}

    def match(self):
        """
        Starts the matching process using DFS.
        """
        return self.dfs(self.data_dict, [])

    def dfs(self, data, path):
        """
        Depth-First Search to find a matching path.
        """
        if self.exit_strategy_solution(data):
            return path

        for rule in self.rule_dict:
            new_data, new_path = self.apply_rule(data, path, rule)
            if new_data is not None:
                result = self.dfs(new_data, new_path)
                if result is not None:
                    return result
        return None

    def apply_rule(self, data, path, rule):
        """
        Applies a rule to the current data and updates the path.
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
        Checks if a rule can be applied to the current data.
        """
        return all(k in data and data[k] >= v for k, v in rule.items() if k != 'Q')

    @staticmethod
    def exit_strategy_solution(data):
        """
        Checks if the current data state represents a valid solution.
        """
        return len(data) == 1 and data.get('Q', 0) == 0
    

@classmethod

class RSMIRuleMatcher:
    """
    A class to match rules based on given data using a depth-first search algorithm.
    Implements backtracking to revert to previous states if a dead end is reached.
    Allows selection between finding the 'best' single solution or 'all' possible solutions.
    """
    def __init__(self, rule_dict, data_dict, select='best', ranking= False):
        self.rule_dict = sorted(rule_dict, key=lambda r: len(r['Composition']), reverse=True)
        self.data_dict = data_dict
        self.select = select
        if self.select == 'all':
            self.all_solutions = []

        self.ranking= ranking

        if 'Q' not in self.data_dict:
            self.data_dict['Q'] = 0
        self.data_dict = {k: v for k, v in self.data_dict.items() if v != 0 or k == 'Q'}

    def match(self):
        if self.select == 'all':
            self.dfs(self.data_dict, [])
            self.all_solutions = self.remove_overlapping_solutions(self.all_solutions)
            return self.rank_solutions(self.all_solutions, self.ranking)
        else:
            solution = self.dfs(self.data_dict, [])
            return [solution] if solution is not None else []

    def dfs(self, data, path):
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
        return all(k in data and data[k] >= v for k, v in rule.items() if k != 'Q')
    

    @staticmethod
    def rank_solutions(solutions, ranking):
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
        return len(data) == 1 and data.get('Q', 0) == 0
    
    @staticmethod
    def remove_overlapping_solutions(solutions):
        unique_solutions = []
        seen = set()

        for solution in solutions:
            # Convert each solution to a set of tuples for comparison
            solution_set = frozenset((item['smiles'], item['Ratio']) for item in solution)
            if solution_set not in seen:
                seen.add(solution_set)
                unique_solutions.append(solution)

        return unique_solutions

