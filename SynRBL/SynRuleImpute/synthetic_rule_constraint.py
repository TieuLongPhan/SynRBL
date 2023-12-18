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
        self.ban_atoms = ban_atoms or ['[H]','[O].[O]', 'F-F', 'Cl-Cl', 'Br-Br', 'I-I', 'Cl-Br', 'Cl-I', 'Br-I']
        self.ban_atoms = [Chem.CanonSmiles(atom) for atom in self.ban_atoms]
        self.ban_pattern = re.compile('|'.join(map(re.escape, self.ban_atoms)))
        self.ban_atoms_reactants = ban_atoms_reactants or ['[H]', '[O]']
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
            
            if '[H]' in entry['products']:
                if RuleConstraint.check_even(entry, 'products', '[H]'):
                    entry['products'] = entry['products'].replace('.[H]', '').replace('[H]', '')
                    entry['reactants'] += '.[O].[O]'

                    if entry['products']:
                        entry['products'] += '.O'
                    else:
                        entry['products'] = 'O'
                else:
                    pass
            
            if '[O]' in entry['products']:
                if RuleConstraint.check_even(entry, 'products', '[O]'):
                    pass
                else:
                    entry['products'] = entry['products'].replace('.[O]', '').replace('[O]', '')
                    entry['reactants'] += '.[H].[H]'

                    if entry['products']:
                        entry['products'] += '.O'
                    else:
                        entry['products'] = 'O'

            new_reaction = f"{entry['reactants']}>>{entry['products']}"
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

        # check validity of reactants, should not contain single [H] or single [O]
        filtered_reactions = [reaction for reaction in filtered_reactions if len(re.findall(ban_pattern_reactants, reaction.get('reactants', ''))) % 2 == 0]
        reactions_with_banned_atoms_reactants = [reaction for reaction in reaction_list if len(re.findall(ban_pattern_reactants, reaction.get('reactants', ''))) % 2 != 0]
        reactions_with_banned_atoms.extend(reactions_with_banned_atoms_reactants)
        return filtered_reactions, reactions_with_banned_atoms
    

    @staticmethod
    def check_even(data_dict: dict, key: str, frag: str, symbol: str = '>>') -> bool:
        """
        Check if the number of occurrences of a fragment in a list of SMILES strings is even.

        Args:
            data_dict (dict): A dictionary containing SMILES strings.
            key (str): The key to access the relevant value in the dictionary.
            frag (str): The fragment to count in the SMILES strings.
            symbol (str, optional): The delimiter used in the SMILES strings. Defaults to '>>'.

        Returns:
            bool: True if the number of occurrences of the fragment is even, False otherwise.
        """
        smiles_list = data_dict[key].split(symbol)
        count = smiles_list.count(frag)
        return count % 2 == 0

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
        
