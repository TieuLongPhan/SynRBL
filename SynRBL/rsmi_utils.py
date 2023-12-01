import json
from typing import List, Dict
from rdkit import Chem

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

def calculate_net_charge(sublist):
    """
    Calculate the net charge from a list of molecules represented as SMILES strings.

    Args:
        sublist (list): A list of dictionaries, each with a 'smiles' string and a 'Ratio'.

    Returns:
        int: Net charge of the sublist.
    """
    total_charge = 0
    for item in sublist:
        if 'smiles' in item and 'Ratio' in item:
            mol = Chem.MolFromSmiles(item['smiles'])
            if mol:
                charge = sum(abs(atom.GetFormalCharge()) for atom in mol.GetAtoms()) * item['Ratio']
                total_charge += charge
    return total_charge



def find_shortest_sublists(solution):
    """
    Find all sublists of dictionaries that have the shortest length.

    Args:
        solution (list of lists): A list containing lists of dictionaries.

    Returns:
        list: A list of all sublists with the shortest length.
    """
    if not solution:
        return []

    min_length = min(len(sublist) for sublist in solution)
    shortest_sublists = [sublist for sublist in solution if len(sublist) == min_length]

    return shortest_sublists


def filter_data(data, unbalance_values=None, formula_key='Diff_formula', element_key=None, min_count=0, max_count=3):
    """
    Filter dictionaries based on a list of unbalance values and element count in a specified formula key.

    This function filters the input list of dictionaries based on the specified list of unbalance values and
    the count of a specific element within a given formula key. It returns dictionaries that match any of the
    unbalance criteria and where the element count falls within the specified range.

    Args:
        data (list of dict): A list of dictionaries to be filtered.
        unbalance_values (list of str, optional): The values to filter by in the 'Unbalance' key. If None, this criterion is ignored.
        formula_key (str): The key in the dictionaries that contains the element counts. Defaults to 'Diff_formula'.
        element_key (str, optional): The element to filter by in the formula key. If None, this criterion is ignored.
        min_count (int): The minimum allowed count of the element. Defaults to 0.
        max_count (int): The maximum allowed count of the element. Defaults to infinity.

    Returns:
        list of dict: A list of dictionaries filtered based on the criteria.
    """
    filtered_data = []
    
    for item in data:
        # Check for unbalance condition
        unbalance_matches = (unbalance_values is None or item.get('Unbalance') in unbalance_values)

        # Check for element count condition
        element_count = item.get(formula_key, {}).get(element_key, 0)
        element_matches = (element_key is None or min_count <= element_count <= max_count)

        if unbalance_matches and element_matches:
            filtered_data.append(item)

    return filtered_data



def remove_duplicates_by_key(data, key_function):
    """
    Remove duplicate entries from a list based on a unique key for each entry.

    Parameters:
    data (list): A list of data entries (dictionaries, objects, etc.).
    key_function (function): A function that takes an entry from `data` and returns a key for duplicate check.

    Returns:
    list: A list of unique entries, based on the unique keys generated.

    Example:
    >>> data = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}, {'name': 'Alice', 'age': 30}]
    >>> remove_duplicates_by_key(data, lambda x: (x['name'], x['age']))
    [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    """

    # Set to keep track of already seen keys
    seen_keys = set()
    unique_data = []

    for entry in data:
        # Generate a key for each entry using the provided key function
        key = frozenset(key_function(entry))
        # Add entry to unique_data if key hasn't been seen before
        if key not in seen_keys:
            seen_keys.add(key)
            unique_data.append(entry)

    return unique_data

def sort_by_key_length(data, key_function):
    """
    Sort a list of entries based on the length of a specific key.

    Parameters:
    data (list): A list of data entries.
    key_function (function): A function that takes an entry from `data` and returns a key whose length is to be used for sorting.

    Returns:
    list: A list of entries sorted by the length of the specified key.

    Example:
    >>> data = [{'name': 'Alice', 'skills': ['Python', 'Java']}, {'name': 'Bob', 'skills': ['HTML']}]
    >>> sort_by_key_length(data, lambda x: x['skills'])
    [{'name': 'Bob', 'skills': ['HTML']}, {'name': 'Alice', 'skills': ['Python', 'Java']}]
    """

    # Sorting the data based on the length of the key returned by key_function
    return sorted(data, key=lambda x: len(key_function(x)))


def add_missing_key_to_dicts(data, dict_key, missing_key, default_value):
    """
    Iterates through a list of dictionaries and adds a specified key with a default value to a specified 
    dictionary within each main dictionary, if the key is not already present. Returns a new list with the updates.

    Parameters:
    data (list): A list of dictionaries.
    dict_key (str): The key in the main dictionaries that points to another dictionary where the check should be done.
    missing_key (str): The key to add if it's not present in the nested dictionary.
    default_value: The default value to assign to the missing key.

    Returns:
    list: A new list of dictionaries with the missing key added where necessary.

    Example:
    >>> data = [{'Composition': {'A': 1, 'B': 2}}, {'Composition': {'B': 3}}]
    >>> updated_data = add_missing_key_to_dicts(data, 'Composition', 'Q', 0)
    >>> updated_data
    [{'Composition': {'A': 1, 'B': 2, 'Q': 0}}, {'Composition': {'B': 3, 'Q': 0}}]
    """

    updated_data = []

    for entry in data:
        # Create a copy of the entry to avoid modifying the original data
        updated_entry = entry.copy()

        # Check if the missing key is not in the specified dictionary
        if missing_key not in updated_entry.get(dict_key, {}):
            # Ensure the dict_key points to a dictionary
            if dict_key not in updated_entry:
                updated_entry[dict_key] = {}

            # Add the missing key with the default value
            updated_entry[dict_key][missing_key] = default_value

        updated_data.append(updated_entry)

    return updated_data


def extract_results_by_key(data, key='new_reaction'):
    """
    Separate dictionaries from a list into two lists based on the presence of a specific key.

    Args:
        data (list of dict): A list of dictionaries to be separated.
        key (str): The key to check for in each dictionary. Defaults to 'new_reaction'.

    Returns:
        tuple of two lists: 
            - The first list contains dictionaries that have the specified key.
            - The second list contains dictionaries that do not have the specified key.
    """

    with_key = []
    without_key = []

    # Separate dictionaries based on the presence of the key
    for item in data:
        if key in item:
            with_key.append(item)
        else:
            without_key.append(item)

    return with_key, without_key
