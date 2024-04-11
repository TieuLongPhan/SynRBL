import json
import random

from typing import List, Dict, Set, Any
from typing import Optional, Union, Callable, Tuple
from collections import defaultdict


def save_database(database: list[dict], pathname: str = "./Data/database.json") -> None:
    """
    Save a database (a list of dictionaries) to a JSON file.

    Args:
        database: The database to be saved.
        pathname: The path where the database will be saved.
            Defaults to './Data/database.json'.

    Raises:
        TypeError: If the database is not a list of dictionaries.
        ValueError: If there is an error writing the file.
    """
    if not all(isinstance(item, dict) for item in database):
        raise TypeError("Database should be a list of dictionaries.")

    try:
        with open(pathname, "w") as f:
            json.dump(database, f)
    except IOError as e:
        raise ValueError(f"Error writing to file {pathname}: {e}")


def load_database(pathname: str = "./Data/database.json") -> List[Dict]:
    """
    Load a database (a list of dictionaries) from a JSON file.

    Args:
        pathname: The path from where the database will be loaded.
            Defaults to './Data/database.json'.

    Returns:
        The loaded database.

    Raises:
        ValueError: If there is an error reading the file.
    """
    try:
        with open(pathname, "r") as f:
            database = json.load(f)  # Load the JSON data from the file
        return database
    except IOError as e:
        raise ValueError(f"Error reading to file {pathname}: {e}")


def extract_atomic_elements(rules: List[Dict[str, Dict[str, int]]]) -> Set[str]:
    """
    Extracts the set of all atomic elements from a list of rules.

    Args:
        rules: A list of rules, where each rule is a dictionary representing a
            composition of atomic elements.

    Returns:
        A set of all atomic elements found in the rules.

    Example:
        ```python
        rules = [
            {"Composition": {"A": 1, "B": 1}},
            {"Composition": {"C": 1, "D": 1, "E": 1}},
        ]
        atomic_elements = extract_atomic_elements(rules)
        print(atomic_elements)  # Output: {'A', 'B', 'C', 'D', 'E'}
        ```
    """

    atomic_elements = set()

    for rule in rules:
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
        if max_comp_len < len(entry["Composition"]):
            # Update the maximum composition length if necessary
            max_comp_len = len(entry["Composition"])

    # Return the maximum composition length
    return max_comp_len


def find_shortest_sublists(solution: List[List[Dict]]) -> List[List[Dict]]:
    """
    Find all sublists of dictionaries that have the shortest length.

    Args:
        solution: A list containing lists of dictionaries.

    Returns:
        A list of all sublists with the shortest length.
    """
    if not solution:
        return []

    min_length = min(len(sublist) for sublist in solution)
    shortest_sublists = [sublist for sublist in solution if len(sublist) == min_length]

    return shortest_sublists


def filter_data(
    data: List[Dict[str, Any]],
    unbalance_values: Optional[List[str]] = None,
    formula_key: str = "Diff_formula",
    element_key: Optional[str] = None,
    min_count: int = 0,
    max_count: int = float("inf"),
) -> List[Dict[str, Any]]:
    """
    Filter dictionaries based on a list of unbalance values and element count
    in a specified formula key.

    This function filters the input list of dictionaries based on the specified
    list of unbalance values and the count of a specific element within a given
    formula key. It returns dictionaries that match any of the unbalance
    criteria and where the element count falls within the specified range.

    Args:
        data: A list of dictionaries to be filtered.
        unbalance_values: The values to filter by in the 'Unbalance' key. If
            None, this criterion is ignored.
        formula_key: The key in the dictionaries that contains the element
            counts. Defaults to 'Diff_formula'.
        element_key: The element to filter by in the formula key. If None, this
            criterion is ignored.
        min_count: The minimum allowed count of the element. Defaults to 0.
        max_count: The maximum allowed count of the element.
            Defaults to infinity.

    Returns:
        A list of dictionaries filtered based on the criteria.
    """
    filtered_data = []

    for item in data:
        # Check for unbalance condition
        unbalance_matches = (
            unbalance_values is None or item.get("Unbalance") in unbalance_values
        )

        # Check for element count condition
        if element_key is None:
            element_matches = True
        else:
            element_count = item.get(formula_key, {}).get(element_key, 0)
            element_matches = min_count <= element_count <= max_count

        if unbalance_matches and element_matches:
            filtered_data.append(item)

    return filtered_data


def remove_duplicates_by_key(
    data: List[dict], key_function: Callable[..., Any]
) -> List[dict]:
    """
    Remove duplicate entries from a list based on a unique key for each entry.

    Parameters:
    - `data` (List[dict]): A list of data entries (dictionaries, objects, etc.).
    - `key_function` (Callable[..., Any]): A function that takes an entry from
        `data` and returns a key for duplicate check.

    Returns:
    - `List[dict]`: A list of unique entries, based on the unique keys generated.

    Example:
    >>> data = [
        {"name": "Alice", "age": 30},
        {"name": "Bob", "age": 25},
        {"name": "Alice", "age": 30},
    ]
    >>> remove_duplicates_by_key(data, lambda x: (x["name"], x["age"]))
    [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    """

    seen_keys = set()
    unique_data = []

    for entry in data:
        key = frozenset(key_function(entry))
        if key not in seen_keys:
            seen_keys.add(key)
            unique_data.append(entry)

    return unique_data


def sort_by_key_length(
    data: List[Any], key_function: Callable[[Any], Any]
) -> List[Any]:
    """
    Sort a list of entries based on the length of a specific key.

    Args:
    - data (List[Any]): A list of data entries.
    - key_function (Callable[[Any], Any]): A function that takes an entry from
        `data` and returns a key whose length is to be used for sorting.

    Returns:
    - List[Any]: A list of entries sorted by the length of the specified key.
    """

    return sorted(data, key=lambda x: len(key_function(x)))


def add_missing_key_to_dicts(
    data: List[Dict[str, Dict[str, Any]]],
    dict_key: str,
    missing_key: str,
    default_value: Any,
) -> List[Dict[str, Dict[str, Any]]]:
    """
    Iterates through a list of dictionaries and adds a specified key with a
    default value to a specified dictionary within each main dictionary, if the
    key is not already present. Returns a new list with the updates.

    Args:
        data: A list of dictionaries.
        dict_key: The key in the main dictionaries that points to another
            dictionary where the check should be done.
        missing_key: The key to add if it's not present in the nested dictionary.
        default_value: The default value to assign to the missing key.

    Returns:
        A new list of dictionaries with the missing key added where necessary.

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


def extract_results_by_key(
    data: List[Dict[str, any]], key: str = "new_reaction"
) -> Tuple[List[Dict[str, any]], List[Dict[str, any]]]:
    """
    Separate dictionaries from a list into two lists based on the presence of a
    specific key.

    Args:
        data: A list of dictionaries to be separated.
        key: The key to check for in each dictionary.
            Defaults to 'new_reaction'.

    Returns:
        A tuple of two lists:
            - The first list contains dictionaries that have the specified key.
            - The second list contains dictionaries that do not have the
                specified key.
    """

    with_key: List[Dict[str, any]] = []
    without_key: List[Dict[str, any]] = []

    # Separate dictionaries based on the presence of the key
    for item in data:
        if key in item:
            with_key.append(item)
        else:
            without_key.append(item)

    return with_key, without_key


def get_random_samples_by_key(
    data: List[Dict[str, Any]],
    stratify_key: str,
    num_samples_per_group: int = 1,
    random_seed: int = None,
) -> List[Dict[str, Any]]:
    """
    Get random samples from data, grouped by a specified key.

    Parameters:
    - data: List of dictionaries containing various keys.
    - stratify_key: The key used for stratifying the data.
    - num_samples_per_group: Number of random samples to draw from each
        unique group.
    - random_seed: Seed for the random number generator for reproducibility.

    Returns:
    - A list of randomly selected samples from each unique group based on
        the stratify_key.
    """
    if random_seed is not None:
        random.seed(random_seed)

    # Function to create a sortable key from the item
    def sortable_key(item: Dict[str, Any]) -> Union[str, tuple]:
        key_value = item.get(stratify_key, None)
        if isinstance(key_value, dict):
            return tuple(sorted(key_value.items()))
        elif isinstance(key_value, list):
            return tuple(sorted(key_value))
        else:
            return key_value

    # Group data by the specified stratify_key
    grouped_data = defaultdict(list)
    for item in data:
        grouped_key = sortable_key(item)
        grouped_data[grouped_key].append(item)

    # Select random samples from each group
    random_samples = []
    for _, items in grouped_data.items():
        if len(items) >= num_samples_per_group:
            random_samples.extend(random.sample(items, num_samples_per_group))
        else:
            # If there aren't enough items, take all available
            random_samples.extend(items)

    return random_samples
