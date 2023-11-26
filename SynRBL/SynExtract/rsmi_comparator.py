from collections import defaultdict
from joblib import Parallel, delayed

class RSMIComparator:
    """
    A class to compare two lists of dictionaries representing reactants and products. 
    It determines if the reaction is balanced and calculates the difference in atomic compositions.

    Parameters
    ----------
    reactants : list of dict
        List of dictionaries, each representing atomic composition of a reactant.
    products : list of dict
        List of dictionaries, each representing atomic composition of a product.
    n_jobs : int
        Number of jobs to run in parallel.
    verbose : int
        Verbosity level.
    
    Example
    -------
    # Example usage of RSMIComparator
    >>> reactants = [{'C': 1, 'H': 4}]
    >>> products = [{'C': 1, 'H': 6}]
    >>> comparator = RSMIComparator(reactants, products)
    >>> comparison_results, difference_results = comparator.run_parallel()
    >>> print(comparison_results, difference_results)
    """

    def __init__(self, reactants, products, n_jobs=10, verbose=1):
        """
        Initialize the RSMIComparator object.
        """
        self.reactants = reactants
        self.products = products
        self.n_jobs = n_jobs
        self.verbose = verbose

    @staticmethod
    def check_keys(dict1, dict2):
        """
        Check if all keys in the second dictionary (dict2) are present in the first dictionary (dict1).

        Parameters:
        dict1 (dict): The first dictionary to compare.
        dict2 (dict): The second dictionary to compare.

        Returns:
        bool: Returns True if all keys in dict2 are present in dict1, otherwise False.

        Example:
        >>> RSMIComparator.check_keys({'C': 2, 'H': 6}, {'C': 1})
        True
        """
        # Check if all keys in dict2 are present in dict1
        return all(key in dict1 for key in dict2)

    @staticmethod
    def compare_dicts(reactant, product):
        """
        Compare two dictionaries representing atomic compositions of reactants and products.

        Parameters:
        reactant (dict): Dictionary representing atomic composition of reactants.
        product (dict): Dictionary representing atomic composition of products.

        Returns:
        str: A string indicating whether the reaction is balanced, reactant-heavy, product-heavy, or both.

        Example:
        >>> RSMIComparator.compare_dicts({'C': 2, 'H': 6}, {'C': 2, 'H': 6})
        'Balance'
        """

        # Check if the keys in both dictionaries are the same
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
        Calculate the difference in atomic compositions between two dictionaries.

        Parameters:
        reactant (dict): Dictionary representing atomic composition of reactants.
        product (dict): Dictionary representing atomic composition of products.

        Returns:
        dict: Dictionary with absolute differences in atomic counts between reactants and products.

        Example:
        >>> RSMIComparator.diff_dicts({'C': 2, 'H': 6}, {'C': 2, 'H': 4})
        {'H': 2}
        """
        diff_dict = {}

        # Calculate differences for keys present in both dictionaries
        for key in reactant.keys():
            if key in product:
                diff_value = abs(reactant[key] - product[key])
                if diff_value != 0:
                    diff_dict[key] = diff_value
            else:
                # Add the count if the key is only in the reactant
                if reactant[key] != 0:
                    diff_dict[key] = reactant[key]

        # Add counts for keys that are only in the product
        for key in product.keys():
            if key not in reactant and product[key] != 0:
                diff_dict[key] = product[key]

        return diff_dict

    def run_parallel(self):
        """
        Run comparison and difference calculation in parallel for reactants and products.

        Returns
        -------
        tuple
            A tuple containing two lists: one for comparison results, and one for differences in compositions.
        """
        # Run comparisons and difference calculations in parallel using joblib
        comparison_results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self.compare_dicts)(reactant, product) for reactant, product in zip(self.reactants, self.products)
        )

        difference_results = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(self.diff_dicts)(reactant, product) for reactant, product in zip(self.reactants, self.products)
        )

        return comparison_results, difference_results
