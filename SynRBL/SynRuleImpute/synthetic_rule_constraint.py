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
        Applies oxidation rules to modify reactions and filters out reactions with banned atoms.
    """

    def __init__(
        self, 
        list_dict: List[Dict[str, Any]], 
        ban_atoms: Optional[List[str]] = None,
        ban_atoms_reactants: Optional[List[str]] = None
        ) -> None:
        """
        Initializes the RuleConstraint class with a list of chemical reactions and optional banned atoms.

        Parameters
        ----------
        list_dict : List[Dict[str, Any]]
            A list of dictionaries, each representing a chemical reaction.
        ban_atoms : Optional[List[str]], optional
            A list of SMILES strings for atoms or molecules to be banned from the reactions. 
            Defaults to a predefined list.
        """
        self.list_dict = copy.deepcopy(list_dict)
        self.ban_atoms = ban_atoms or ['O=O', 'F-F', 'Cl-Cl', 'Br-Br', 'I-I', 'Cl-Br', 'Cl-I', 'Br-I']
        self.ban_atoms = [Chem.CanonSmiles(atom) for atom in self.ban_atoms]
        self.ban_pattern = re.compile('|'.join(map(re.escape, self.ban_atoms)))
        self.ban_atoms_reactants = ban_atoms_reactants or ['[O]']
        self.ban_atoms_reactants = [Chem.CanonSmiles(atom) for atom in self.ban_atoms_reactants]
        self.ban_pattern_reactants = re.compile('|'.join(map(re.escape, self.ban_atoms_reactants)))
  
    @staticmethod
    def reduction_oxidation_rules_modify(
        data: List[Dict[str, str]]
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
            products = entry['products']
            reactants = entry['reactants']
            
            if '[H]' in products:
                products = products.replace('.[H]', '').replace('[H]', '')
                reactants += '.O=O'

                if products:
                    products += '.O'
                else:
                    products = 'O'
            
            if '[O]' in products:
                products = products.replace('.[O]', '').replace('[O]', '')
                reactants += '.[H].[H]'

                if products:
                    products += '.O'
                else:
                    products = 'O'

            new_reaction = f"{reactants}>>{products}"
            entry['new_reaction'] = new_reaction
            modified_data.append(entry)

        return modified_data
    
    @staticmethod
    def reduction_rules_modify(
        data: List[Dict[str, str]]
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
            products = entry['products']
            reactants = entry['reactants']
            if '[O]' in products:
                products = products.replace('.[O]', '').replace('[O]', '')
                reactants += '.[H].[H]'

                if products:
                    products += '.O'
                else:
                    products = 'O'

            new_reaction = f"{reactants}>>{products}"
            entry['new_reaction'] = new_reaction
            modified_data.append(entry)

        return modified_data


    @staticmethod
    def remove_banned_reactions(
        reaction_list: List[Dict[str, str]], 
        ban_pattern: Pattern,
        ban_pattern_reactants: Pattern
        ) -> Tuple[List[Dict[str, str]], List[Dict[str, str]]]:
        """
        Filters out reactions that contain banned atoms in their products.

        Parameters:
        reaction_list (List[Dict[str, str]]): A list containing reaction data.
        ban_pattern (Pattern): A compiled regular expression pattern that matches any of the banned atoms.

        Returns:
        Tuple[List[Dict[str, str]], List[Dict[str, str]]]: The filtered list of reactions without and with banned atoms in their products.
        """
        filtered_reactions = [reaction for reaction in reaction_list if not ban_pattern.search(reaction.get('products', ''))]
        reactions_with_banned_atoms = [reaction for reaction in reaction_list if ban_pattern.search(reaction.get('products', ''))]

        filtered_reactions = [reaction for reaction in filtered_reactions if not ban_pattern_reactants.search(reaction.get('reactants', ''))]
        reactions_with_banned_atoms_reactants = [reaction for reaction in filtered_reactions if ban_pattern.search(reaction.get('reactants', ''))]
        reactions_with_banned_atoms.extend(reactions_with_banned_atoms_reactants)
        return filtered_reactions, reactions_with_banned_atoms
    
    def fit(self
        ) -> List[Dict[str, Any]]:
        """
        Applies oxidation modification rules and filters out reactions with banned atoms.

        Returns
        -------
        List[Dict[str, Any]]
            The modified and filtered list of chemical reactions.
        """
        data_modified = self.reduction_oxidation_rules_modify(self.list_dict)
        return self.remove_banned_reactions(data_modified, self.ban_pattern, self.ban_pattern_reactants)
        
