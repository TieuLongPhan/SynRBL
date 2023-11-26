import copy
from rdkit import Chem
from SynRBL.SynRuleImpute.synthetic_rule_matcher import SyntheticRuleMatcher
import copy
from rdkit import Chem

class SyntheticRuleImputer(SyntheticRuleMatcher):
    """
    A class for imputing missing chemical data based on provided rules.

    Example:
    rule_dict = [{'smiles': 'C1=CC=CC=C1', 'Composition': {'C': 6, 'H': 6}},
                 {'smiles': 'CCO', 'Composition': {'C': 2, 'H': 6, 'O': 1}}]

    missing_data = [{'Diff_formula': {'C': 4, 'H': 4}, 'Unbalance': 'Products',
                     'reactants': 'C2H6', 'products': 'C2H6.C1=CC=CC=C1'},
                    {'Diff_formula': {'C': 3, 'H': 6}, 'Unbalance': 'Reactants',
                     'reactants': 'C2H6', 'products': 'C2H6.C1=CC=CC=C1'}]

    imputer = SyntheticRuleImputer(rule_dict, select='best', ranking='longest')
    imputed_data = imputer.impute(missing_data)

    print(imputed_data)
    # Output:
    # [{'Diff_formula': {'C': 4, 'H': 4}, 'Unbalance': 'Products',
    #   'reactants': 'C2H6', 'products': 'C2H6.C1=CC=CC=C1.CCO',
    #   'new_reaction': 'C2H6>>C2H6.C1=CC=CC=C1.CCO'},
    #  {'Diff_formula': {'C': 3, 'H': 6}, 'Unbalance': 'Reactants',
    #   'reactants': 'C2H6', 'products': 'C2H6.C1=CC=CC=C1',
    #   'new_reaction': 'C2H6>>C2H6.C1=CC=CC=C1'}]
    """

    def __init__(self, rule_dict, select='best', ranking='longest'):
        """
        Initialize the SyntheticRuleImputer.

        Args:
            rule_dict (list): A list of dictionaries representing chemical rules.
            select (str): Selection mode, either 'best' for the best solution or 'all' for all solutions.
            ranking (str): Ranking mode for solutions, options are 'longest', 'least', 'greatest'.
        """
        self.rule_dict = rule_dict
        self.select = select
        self.ranking = ranking

    def impute(self, missing_dict):
        """
        Impute missing chemical data based on the provided rules.

        Args:
            missing_dict (list): A list of dictionaries representing missing chemical data.

        Returns:
            list: A list of dictionaries with imputed data.
        """
        dict_impute = copy.deepcopy(missing_dict)

        for item in dict_impute:
            matcher = SyntheticRuleMatcher(self.rule_dict, item['Diff_formula'], select=self.select, ranking=self.ranking)
            solution = matcher.match()

            if solution:
                valid_smiles = self.get_and_validate_smiles(solution[0]) if len(solution[0]) > 1 else solution[0][0]['smiles']

                if valid_smiles:
                    key = 'products' if item['Unbalance'] == 'Products' else 'reactants'
                    item[key] += '.' + valid_smiles

                    # Construct the new_reaction key
                    item['new_reaction'] = item['reactants'] + '>>' + item['products']

        return dict_impute

    @staticmethod
    def get_and_validate_smiles(solution):
        """
        Concatenate smiles strings and validate the smiles string using RDKit.

        Args:
            solution (list): A list of dictionaries representing chemical solutions.

        Returns:
            str or None: A validated smiles string or None if validation fails.
        """
        # Concatenate smiles strings
        new_smiles = '.'.join(item['smiles'] for item in solution)

        # Validate the smiles string using RDKit
        if Chem.MolFromSmiles(new_smiles) is not None:
            return new_smiles
        return None

