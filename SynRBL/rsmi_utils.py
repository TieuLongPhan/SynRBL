import json
from typing import List, Dict

def save_database(database, pathname='./Data/database.json'):
    """
    Save a database (a list of dictionaries) to a JSON file.

    Args:
        database (list of dict): The database to be saved.
        pathname (str, optional): The path where the database will be saved. Defaults to './Data/database.json'.

    Raises:
        TypeError: If the database is not a list of dictionaries.
        IOError: If there is an error writing the file.
    """
    # Check if the database is a list of dictionaries
    if not all(isinstance(item, dict) for item in database):
        raise TypeError("Database should be a list of dictionaries.")

    try:
        # Open the file in write mode and save the database as JSON
        with open(pathname, 'w') as f:
            json.dump(database, f)
    except IOError as e:
        # If there is an error writing the file, raise an exception
        raise ValueError(f"Error writing to file {pathname}: {e}")

def load_database(pathname='./Data/database.json'):
    """
    Load a database (a list of dictionaries) from a JSON file.

    Args:
        pathname (str, optional): The path from where the database will be loaded. Defaults to './Data/database.json'.

    Returns:
        list of dict: The loaded database.

    Raises:
        IOError: If there is an error reading the file.
    """
    try:
        with open(pathname, 'r') as f:
            database = json.load(f)  # Load the JSON data from the file
        return database
    except IOError as e:
        raise ValueError(f"Error reading to file {pathname}: {e}")
    

def extract_atomic_elements(rules):
    """
    Extracts the set of all atomic elements from a list of rules.

    Args:
        rules (list of dict): A list of rules, where each rule is a dictionary
            representing a composition of atomic elements.

    Returns:
        set: A set of all atomic elements found in the rules.

    Example:
        ```python
        rules = [{"Composition": {"A": 1, "B": 1}}, {"Composition": {"C": 1, "D": 1, "E": 1}}]
        atomic_elements = extract_atomic_elements(rules)
        print(atomic_elements)  # Output: {'A', 'B', 'C', 'D', 'E'}
        ```
    """

    atomic_elements = set()

    # Iterate over the rules
    for rule in rules:

        # Extract the atomic elements from the current rule
        atomic_elements.update(rule["Composition"].keys())

    return atomic_elements




def _get_max_comp_len(database: List[Dict]) -> int:
    """
    Determines the maximum length of a composition in the database.

    Args:
        database (List[Dict]): A list of dictionaries representing the database.

    Returns:
        int: The maximum length of a composition in the database.

    Example:
    ```python
    database = [
        {'Composition': ['H2O', 'CO2']},
        {'Composition': ['NaCl', 'H2O']},
        {'Composition': ['C6H12O6', 'H2O']},
    ]

    max_comp_len = _get_max_comp_len(database)
    print(max_comp_len)  # Output: 3
    ```
    """

    max_comp_len = 0  # Initialize the maximum composition length

    # Iterate through each entry in the database
    for entry in database:
        # Check if the current entry's composition length exceeds the maximum
        if max_comp_len < len(entry['Composition']):
            # Update the maximum composition length if necessary
            max_comp_len = len(entry['Composition'])

    # Return the maximum composition length
    return max_comp_len


def build_lookups(atomic_elements: set, database: list) -> list:
    """
    Constructs lookup dictionaries for atomic elements in the database, removing empty lists from the lookup.

    Args:
        atomic_elements (set): A set of unique atomic elements.
        database (list): A list of database entries, where each entry is a dictionary
                        containing a "Composition" key.

    Returns:
        list: A list of lookup dictionaries, where each dictionary maps atomic elements to
                their corresponding database indices for molecules with different composition lengths.

    Example:
        ```python
        atomic_elements = {'H', 'C', 'O', 'N'}
        database = [
            {'Composition': {'H': 2, 'O': 1}},
            {'Composition': {'C': 1, 'H': 4}},
            {'Composition': {'N': 2, 'O': 3}},
            {'Composition': {'S': 1, 'O': 2}},
        ]

        lookup = build_lookups(atomic_elements, database)
        print(lookup)
        

        This example demonstrates how to use the `build_lookups` function to construct lookup dictionaries
        for the provided atomic elements and database. The resulting lookup dictionaries are optimized by
        removing empty lists, making them more compact and efficient.
    """

    max_comp_len = _get_max_comp_len(database)

    lookup = []

    # Iterate through composition lengths from 2 to max_comp_len
    for comp_len in range(2, max_comp_len+1):
        lookup_dict = {}

        for atom in atomic_elements:
            atom_list = []

            # Iterate through database entries
            for key, entry in enumerate(database):
                if atom in entry["Composition"].keys():
                    if len(entry["Composition"]) == comp_len:
                        atom_list.append(key)

            # Remove empty lists from the lookup dictionary
            if not atom_list:
                continue
            lookup_dict[atom] = atom_list

        lookup.append(lookup_dict)

    return lookup