from joblib import Parallel, delayed


class BothSideReact:
    """
    Class to process chemical reactions on both sides (reactants and products).

    Attributes:
    react_dict (list): A list of dictionaries representing reactants.
    product_dict (list): A list of dictionaries representing products.
    unbalance (list): A list indicating the balance status of each reaction.
    diff_formula (list): A list containing the differential formula for
        each reaction.
    """

    def __init__(self, react_dict, product_dict, unbalance, diff_formula):
        """
        Initializes the BothSideReact class with reaction dictionaries,
        unbalance, and differential formulas.

        Parameters:
        react_dict (list): List of dictionaries representing reactants.
        product_dict (list): List of dictionaries representing products.
        unbalance (list): List indicating the balance status of each reaction.
        diff_formula (list): List containing the differential formula for
            each reaction.
        """
        self.react_dict = react_dict
        self.product_dict = product_dict

        # Ensure 'Q' key is present in all reactant and product dictionaries
        for d in [self.react_dict, self.product_dict]:
            for value in d:
                if "Q" not in value:
                    value["Q"] = 0

        self.unbalance = unbalance
        self.diff_formula = diff_formula

    @staticmethod
    def enforce_product_side(react_dict, product_dict):
        """
        Enforces the product side of the reaction by calculating the difference
        between reactant and product counts.

        Parameters:
        react_dict (dict): Dictionary representing a single reactant.
        product_dict (dict): Dictionary representing a single product.

        Returns:
        dict: Dictionary representing the differential count between reactant
            and product.
        """
        diff_dict = {}
        # Calculate the difference between reactants and products
        for key, value in react_dict.items():
            diff_value = value - product_dict.get(key, 0)
            if diff_value != 0:
                diff_dict[key] = diff_value

        # Add keys only present in product_dict
        for key, value in product_dict.items():
            if key not in react_dict:
                diff_dict[key] = -value

        return diff_dict

    @staticmethod
    def filter_list_by_indices(data, indices):
        """
        Filters a list by specified indices.

        Parameters:
        data (list): The original list to filter.
        indices (list): List of indices to include in the filtered list.

        Returns:
        list: A filtered list containing elements at the specified indices.
        """
        # Use list comprehension for efficient filtering
        return [data[i] for i in indices if i < len(data)]

    @staticmethod
    def reverse_values_if_negative_except_Q(diff_dict):
        """
        Reverses the values in the dictionary if negative, except for the key 'Q'.

        Parameters:
        diff_dict (dict): The dictionary with potential negative values.

        Returns:
        tuple: A tuple containing the updated dictionary and a string
            indicating the balance status.
        """
        if len(diff_dict) == 2 and "Q" in diff_dict.keys():
            if any(value < 0 for key, value in diff_dict.items() if key != "Q"):
                # Reverse all values except for 'Q'
                return {key: -value for key, value in diff_dict.items()}, "Reactants"
            else:
                # Original dictionary if no negative values found
                return diff_dict, "Products"
        else:
            # Return as is if conditions not met
            return diff_dict, "Both"

    def fit(self, n_jobs=4):
        """
        Processes the reactions by balancing reactants and products and
        updating the unbalance status.

        Returns:
        tuple: A tuple containing the updated diff_formula and unbalance lists.
        """
        # Filter indices where balance status is 'Both'
        both_index = [i for i, val in enumerate(self.unbalance) if val == "Both"]
        react_dict_both = self.filter_list_by_indices(self.react_dict, both_index)
        product_dict_both = self.filter_list_by_indices(self.product_dict, both_index)

        # Process reactions in parallel for efficiency
        diff_dict = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(self.enforce_product_side)(react, prod)
            for react, prod in zip(react_dict_both, product_dict_both)
        )

        # Post-process to determine the new balance status
        diff_dict_both, unbalance_both = [], []
        for item in diff_dict:
            d, u = self.reverse_values_if_negative_except_Q(item)
            diff_dict_both.append(d)
            unbalance_both.append(u)

        # Update diff_formula and unbalance lists
        for index, diff_new, unbalance_new in zip(
            both_index, diff_dict_both, unbalance_both
        ):
            self.diff_formula[index] = diff_new
            self.unbalance[index] = unbalance_new

        return self.diff_formula, self.unbalance
