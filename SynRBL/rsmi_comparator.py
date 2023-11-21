from collections import defaultdict
from joblib import Parallel, delayed

class RSMIComparator:
    """
    A class to compare two lists of dictionaries of reactants and products, 
    then return if the reaction is balanced or not,and return the difference in atmoic compositions.
    """

    def __init__(self, reactants, products, n_jobs=10, verbose=1):
        """
        Initialize the RSMIComparator object.

        Parameters:
        reactants (list): The list of dictionaries of reactants to compare.
        products (list): The list of dictionaries of products to compare.
        n_jobs (int): The number of jobs to run in parallel.
        verbose (int): The verbosity level.
        """

        self.reactants = reactants
        self.products = products
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def check_keys(dict1, dict2):
        """
        Check if all keys in dict2 are in dict1.

        Parameters:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

        Returns:
        bool: True if all keys in dict2 are in dict1, False otherwise.
        """

        return all(key in dict1 for key in dict2)

    @staticmethod
    def compare_dicts(reactant, product):
        """
        Compare two dictionaries.

        Parameters:
        reactant (dict): The dictionary of reactants.
        product (dict): The dictionary of products.

        Returns:
        str: A string indicating the comparison result.
        """

        if reactant.keys() != product.keys():
            if RSMIComparator.check_keys(reactant, product) and not RSMIComparator.check_keys(product, reactant):
                if all(reactant[key] >= product[key] for key in product.keys()):
                    return "Products"
                else:
                    return "Both"
            elif RSMIComparator.check_keys(product, reactant) and not RSMIComparator.check_keys(reactant, product):
                if all(reactant[key] <= product[key] for key in reactant.keys()):
                    return "Reactants"
                else:
                    return "Both"
            else:
                return "Both"
        else:
            equal = all(reactant[key] == product[key] for key in reactant.keys())
            greater = all(reactant[key] >= product[key] for key in reactant.keys())
            lower = all(reactant[key] <= product[key] for key in reactant.keys())
            if equal:
                return "Balance"
            elif greater:
                return "Products"
            elif lower:
                return "Reactants"
            else:
                return "Both"

    @staticmethod
    def diff_dicts(reactant, product):
        """
        Calculate the difference between two dictionaries.

        Parameters:
        reactant (dict): The dictionary of reactants.
        product (dict): The dictionary of products.

        Returns:
        dict: A dictionary of the absolute differences between the values of the same keys in the two dictionaries.
        """

        diff_dict = {}

        for key in reactant.keys():
            if key in product:
                diff_value = abs(reactant[key] - product[key])
                if diff_value != 0:
                    diff_dict[key] = diff_value
            else:
                if reactant[key] != 0:
                    diff_dict[key] = reactant[key]

        for key in product.keys():
            if key not in reactant and product[key] != 0:
                diff_dict[key] = product[key]

        return diff_dict

    def run_parallel(self):
        """
        Run the compare_dicts and diff_dicts methods in parallel.

        Returns:
        list: A list of tuples, each containing the results of the compare_dicts and diff_dicts methods for a pair of dictionaries.
        """

        # Initialize an empty list to store the results
        #results = []

        # Iterate over each pair of dictionaries in the reactants and products lists
        #for reactant, product in zip(self.reactants, self.products):
            # Append the results of the compare_dicts and diff_dicts methods to the list
            #results.append(Parallel(n_jobs=self.n_jobs, verbose=1)(delayed(func)(reactant, product) for func in [self.compare_dicts, self.diff_dicts]))

        unbalance = Parallel(n_jobs=-1, verbose=self.verbose)(delayed(self.compare_dicts)(self.reactants[i], self.products[i]) 
                                                   for i in range(len(self.reactants))) 
        diff_formula = Parallel(n_jobs=-1, verbose=self.verbose)(delayed(self.diff_dicts)(self.reactants[i], self.products[i]) 
                                                      for i in range(len(self.reactants)))   

        return unbalance, diff_formula
