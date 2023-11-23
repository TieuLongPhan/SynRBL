import json
import json

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

import json

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