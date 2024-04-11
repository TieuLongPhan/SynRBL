import re

from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcNumRings
from copy import deepcopy


def remove_atom_mapping_from_reaction_smiles(reaction_smiles):
    """
    Remove atom mapping from a reaction SMILES string.

    Parameters:
    - reaction_smiles (str): A reaction SMILES string with atom mapping.

    Returns:
    - str: A reaction SMILES string without atom mapping.
    """
    parts = reaction_smiles.split(">>")
    cleaned_parts = [Chem.CanonSmiles(re.sub(r":\d+", "", part)) for part in parts]
    cleaned_reaction_smiles = ">>".join(cleaned_parts)
    return cleaned_reaction_smiles


def calculate_chemical_properties(dictionary_list):
    """
    Calculate chemical properties from a list of dictionaries containing
    reaction information.

    Parameters:
    - dictionary_list (list of dict): A list of dictionaries with
        reaction information.

    Returns:
    - list of dict: Updated list with added chemical properties.
    """
    updated_list = deepcopy(dictionary_list)
    for entry in updated_list:
        reactant_smiles = entry["reactants"]
        product_smiles = entry["products"]
        reactant_mol = Chem.MolFromSmiles(reactant_smiles)
        product_mol = Chem.MolFromSmiles(product_smiles)

        if reactant_mol is not None and product_mol is not None:
            num_carbon_reactants = sum(
                [atom.GetAtomicNum() == 6 for atom in reactant_mol.GetAtoms()]
            )
            num_carbon_products = sum(
                [atom.GetAtomicNum() == 6 for atom in product_mol.GetAtoms()]
            )
            entry["carbon_difference"] = abs(num_carbon_reactants - num_carbon_products)
            entry["total_carbons"] = num_carbon_reactants + num_carbon_products
            entry["total_bonds"] = abs(
                reactant_mol.GetNumBonds() - product_mol.GetNumBonds()
            )
            entry["total_rings"] = abs(
                CalcNumRings(reactant_mol) - CalcNumRings(product_mol)
            )
        else:
            entry["carbon_difference"] = "Invalid SMILES"
            entry["total_carbons"] = "Invalid SMILES"
            entry["total_bonds"] = "Invalid SMILES"
            entry["total_rings"] = "Invalid SMILES"
        reactant_fragment_count = len(reactant_smiles.split("."))
        product_fragment_count = len(product_smiles.split("."))
        total_fragment_count = reactant_fragment_count + product_fragment_count
        entry["fragment_count"] = total_fragment_count

    return updated_list


def count_boundary_atoms_products_and_calculate_changes(
    list_of_dicts, reaction_col="new_reaction", mcs_col="mcs"
):
    """
    Count boundary atoms in products and calculate changes in bonds and rings.

    Parameters:
    - list_of_dicts (list of dict): A list of dictionaries with reaction information.

    Returns:
    - list of dict: Updated list with calculated changes.
    """
    for item in list_of_dicts:
        count = 0
        bond_change = 0
        ring_change = 0

        if mcs_col in item.keys():
            for i in item[mcs_col]["boundary_atoms_products"]:
                if isinstance(i, dict):
                    count += 1
                elif isinstance(i, list):
                    for j in i:
                        if isinstance(j, dict):
                            count += 1

        reactant_product = item[reaction_col].split(">>")
        if len(reactant_product) == 2:
            reactant_smiles, product_smiles = reactant_product
            reactant_mol = Chem.MolFromSmiles(reactant_smiles)
            product_mol = Chem.MolFromSmiles(product_smiles)

            if reactant_mol and product_mol:
                bond_change = abs(
                    reactant_mol.GetNumBonds() - product_mol.GetNumBonds()
                )
                ring_change = abs(
                    CalcNumRings(reactant_mol) - CalcNumRings(product_mol)
                )

        item["num_boundary"] = count
        item["bond_change_merge"] = bond_change
        item["ring_change_merge"] = ring_change

    return list_of_dicts
