import copy
import re
from rdkit import Chem
from typing import List, Dict, Pattern, Any, Optional, Tuple


class RuleConstraint:
    """
    A class for applying specific chemical reaction constraints and modifications.

    Attributes
    ----------
    list_dict : list of dict
        A list containing chemical reaction data.
    ban_atoms : list of str
        A list of SMILES strings representing banned atoms or molecules in reactions.

    Methods
    -------
    fit():
        Applies oxidation rules to modify reactions and filters out reactions
        with banned atoms.
    """

    def __init__(
        self,
        list_dict: List[Dict[str, Any]],
        ban_atoms: Optional[List[str]] = None,
        ban_atoms_reactants: Optional[List[str]] = None,
    ) -> None:
        """
        Initializes the RuleConstraint class with a list of chemical reactions
        and optional banned atoms.

        Parameters
        ----------
        list_dict : List[Dict[str, Any]]
            A list of dictionaries, each representing a chemical reaction.
        ban_atoms : Optional[List[str]], optional
            A list of SMILES strings for atoms or molecules to be banned from
            the reactions. Defaults to a predefined list.
        """
        self.list_dict = copy.deepcopy(list_dict)
        self.ban_atoms = ban_atoms or [
            "[H]",
            "[O].[O]",
            "F-F",
            "Cl-Cl",
            "Br-Br",
            "I-I",
            "Cl-Br",
            "Cl-I",
            "Br-I",
        ]
        self.ban_atoms = [Chem.CanonSmiles(atom) for atom in self.ban_atoms]
        self.ban_pattern = re.compile("|".join(map(re.escape, self.ban_atoms)))
        self.ban_atoms_reactants = ban_atoms_reactants or [".[H]"]
        self.ban_pattern_reactants = re.compile(
            "|".join(map(re.escape, self.ban_atoms_reactants))
        )

    @staticmethod
    def reduction_oxidation_rules_modify(
        data: List[Dict[str, str]],
    ) -> List[Dict[str, str]]:
        """
        Modify the oxidation rules in the given data.

        Args:
            data (List[Dict[str, str]]): The input data containing oxidation rules.

        Returns:
            List[Dict[str, str]]: The modified data with updated oxidation rules.
        """
        modified_data = []
        for entry in data:
            if ".[H]" in entry["products"]:
                reactants = entry["reactants"].split(".")
                reactants = [
                    RuleConstraint.remove_atom_mapping(smiles) for smiles in reactants
                ]
                no_constraint = ["[Na]", "[K]", "[Li]", "[H-]"]
                contains_no_constraint = RuleConstraint.check_no_constraint(
                    reactants, no_constraint
                )
                if contains_no_constraint:
                    pass
                else:
                    if RuleConstraint.check_even(entry, "products", "[H]", "."):
                        hydrogen_count = entry["products"].count(".[H]")
                        hydrogen_count = int(hydrogen_count / 2)
                        entry["products"] = entry["products"].replace(".[H]", "")
                        entry["reactants"] += ".[O]" * hydrogen_count

                        if entry["products"]:
                            entry["products"] += ".O" * hydrogen_count
                        else:
                            entry["products"] = "O" * hydrogen_count
                    else:
                        pass

            if ".[O]" in entry["products"]:
                if RuleConstraint.check_even(entry, "products", "[O]", "."):
                    pass
                else:
                    oxygen_count = int(entry["products"].count(".[O]"))
                    entry["products"] = entry["products"].replace(".[O]", "")
                    entry["reactants"] += ".[H].[H]" * oxygen_count

                    if entry["products"]:
                        entry["products"] += ".O" * oxygen_count
                    else:
                        entry["products"] = "O" * oxygen_count

            elif ".OO" in entry["products"]:
                entry["products"] = entry["products"].replace(".OO", "")
                entry["reactants"] += ".[H].[H]"

                if entry["products"]:
                    entry["products"] += ".O.O"
                else:
                    entry["products"] = "O.O"

            new_reaction = f"{entry['reactants']}>>{entry['products']}"
            entry["new_reaction"] = new_reaction
            modified_data.append(entry)

        return modified_data

    @staticmethod
    def remove_banned_reactions(
        reaction_list: List[Dict[str, str]],
        ban_pattern: Pattern,
        ban_pattern_reactants: Pattern,
    ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Filters out reactions that contain banned atoms in their products.

        Parameters:
        reaction_list (List[Dict[str, str]]): A list containing reaction data.
        ban_pattern (Pattern): A compiled regular expression pattern that
            matches any of the banned atoms.

        Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]: The filtered list
            of reactions without and with banned atoms in their products.
        """
        filtered_reactions = [
            reaction
            for reaction in reaction_list
            if not ban_pattern.search(reaction.get("products", ""))
        ]
        reactions_with_banned_atoms = [
            reaction
            for reaction in reaction_list
            if ban_pattern.search(reaction.get("products", ""))
        ]

        # check validity of reactants, should not contain single [H] or single [O]
        filtered_reactions = [
            reaction
            for reaction in filtered_reactions
            if len(re.findall(ban_pattern_reactants, reaction.get("reactants", ""))) % 2
            == 0
        ]
        reactions_with_banned_atoms_reactants = [
            reaction
            for reaction in reaction_list
            if len(re.findall(ban_pattern_reactants, reaction.get("reactants", ""))) % 2
            != 0
        ]
        reactions_with_banned_atoms.extend(reactions_with_banned_atoms_reactants)
        return filtered_reactions, reactions_with_banned_atoms

    @staticmethod
    def check_even(data_dict: dict, key: str, frag: str, symbol: str = ">>") -> bool:
        """
        Check if the number of occurrences of a fragment in a list of SMILES
        strings is even.

        Args:
            data_dict (dict): A dictionary containing SMILES strings.
            key (str): The key to access the relevant value in the dictionary.
            frag (str): The fragment to count in the SMILES strings.
            symbol (str, optional): The delimiter used in the SMILES strings.
                Defaults to '>>'.

        Returns:
            bool: True if the number of occurrences of the fragment is even,
                False otherwise.
        """
        smiles_list = data_dict[key].split(symbol)
        count = smiles_list.count(frag)
        return count % 2 == 0

    def fit(self) -> List[Dict[str, Any]]:
        """
        Applies oxidation modification rules and filters out reactions with
        banned atoms.

        Returns
        -------
        List[Dict[str, Any]]
            The modified and filtered list of chemical reactions.
        """
        data_modified = self.reduction_oxidation_rules_modify(self.list_dict)
        return self.remove_banned_reactions(
            data_modified, self.ban_pattern, self.ban_pattern_reactants
        )

    @staticmethod
    def remove_atom_mapping(smiles: str) -> str:
        """
        Remove atom mapping numbers from a SMILES string.

        Atom mappings are typically represented by numbers following a
        colon (':') after the atom symbol. This function removes these mappings
        to return a SMILES string without them.

        Args:
            smiles (str): A SMILES string with atom mappings.

        Returns:
            str: A SMILES string without atom mappings.
        """
        # Regular expression to find and remove atom mappings
        # (numbers following a colon)
        mapping_pattern = re.compile(r":\d+")
        return mapping_pattern.sub("", smiles)

    @staticmethod
    def check_no_constraint(reactants, no_constraint):
        """
        Check if any elements in the no_constraint list match exactly with any
        elements in the reactants list.

        Args:
            reactants (List[str]): A list of reactant elements or compounds.
            no_constraint (List[str]): A list of elements to check against in
                the reactants list.

        Returns:
            bool: True if any element from no_constraint is found in reactants,
                False otherwise.
        """
        # Convert lists to sets for efficient membership testing
        reactants_set = set(reactants)
        no_constraint_set = set(no_constraint)

        # Check for intersection
        return not reactants_set.isdisjoint(no_constraint_set)
