class RSMIComparator:
    # Class definition...

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
            # Additional logic for comparing dictionaries with different keys
            # ...
        else:
            # Logic for comparing dictionaries with the same keys
            # ...
        # Return comparison result

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
