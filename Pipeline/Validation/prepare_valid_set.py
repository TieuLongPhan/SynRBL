import sys
from pathlib import Path
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
from SynRBL.rsmi_utils import load_database, save_database
import pandas as pd
from SynRBL.SynExtract.rsmi_processing import RSMIProcessing
from SynRBL.SynExtract import RSMIDecomposer  
from SynRBL.SynExtract.rsmi_comparator import RSMIComparator
import random
from typing import List, Dict, Optional, Tuple

def get_unbalance(data: pd.DataFrame, n_jobs: int = 4) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Processes a DataFrame containing reaction data to identify balanced and unbalanced reactions.

    The function uses RSMIProcessing to split reaction SMILES, RSMIDecomposer to decompose 
    reactions into reactants and products, and RSMIComparator to compare and identify 
    balanced and unbalanced reactions.

    Args:
    data (pd.DataFrame): A DataFrame containing reaction data with a column 'reactions'.
    n_jobs (int): The number of jobs to run in parallel. Default is 4.

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: Two DataFrames, the first containing balanced reactions
                                       and the second containing unbalanced reactions.
    """
    # Assuming RSMIProcessing, RSMIDecomposer, and RSMIComparator are predefined classes/functions
    process = RSMIProcessing(data=data, rsmi_col='reactions', parallel=True, n_jobs=n_jobs, 
                             save_json=False, save_path_name=None)
    reactions = process.data_splitter()

    decompose = RSMIDecomposer(smiles=None, data=reactions, reactant_col='reactants', 
                               product_col='products', parallel=True, n_jobs=n_jobs, verbose=1)
    react_dict, product_dict = decompose.data_decomposer()

    # Compare dict and check balance
    comp = RSMIComparator(reactants=react_dict, products=product_dict, n_jobs=n_jobs)
    unbalance, _ = comp.run_parallel(reactants=react_dict, products=product_dict)
    unbalance_idx = [key for key, value in enumerate(unbalance) if value != 'Balance']
    balance_idx = [key for key, value in enumerate(unbalance) if value == 'Balance']
    
    return data.loc[balance_idx, :], data.loc[unbalance_idx, :]




def find_appropriate_reactions(reactions: List[str]) -> List[str]:
    """
    Identifies valid reactions based on specific criteria.

    Args:
    reactions (List[str]): List of reaction strings.
    reactants (List[str]): List of reactant strings.
    products (List[str]): List of product strings.

    Returns:
    List[str]: List of valid reactions.
    """
    reactants = [x['reactants'].split('.') for x in reactions]
    products = [x['products'].split('.') for x in reactions]
    super_balance = [key for key, _ in enumerate(reactions) if len(reactants[key]) == 1 and len(products[key]) == 1]
    
    product_1_key = [key for key, value in enumerate(products) if len(value) == 1]


    valid_reactions = [reactions[key] for key in range(len(reactions)) if key not in super_balance and key in product_1_key]

    return valid_reactions




def generate_artificial_validation(reaction_list: List[Dict[str, str]], 
                      select: str = 'longest', 
                      transfer_two: bool = False, 
                      random_state: Optional[int] = None) -> List[Dict[str, str]]:
    """
    Process a list of chemical reaction data.

    For each reaction dictionary in the list, the function selects the longest or shortest SMILES string 
    from the 'reactants' based on the 'select' parameter, or two random SMILES strings if 'transfer_two' is True.
    The selected SMILES string(s) are then moved to a new key 'ground truth' in the reaction dictionary.

    Args:
    reaction_list (List[Dict[str, str]]): A list of dictionaries containing reaction data.
    select (str): Selection criteria for SMILES strings ('longest' or 'shortest').
    transfer_two (bool): If True, transfer two random SMILES strings to 'ground truth' if there are more than two reactants.
    random_state (Optional[int]): Seed for the random number generator.

    Returns:
    List[Dict[str, str]]: A new list of dictionaries with updated reaction data.
    """

    processed_list = []
    
    # Set the random seed for reproducibility if specified
    if random_state is not None:
        random.seed(random_state)

    for reaction in reaction_list:
        # Create a copy of the reaction dictionary
        reaction_copy = reaction.copy()

        # Splitting the reactants SMILES string
        reactants_smiles = reaction_copy['reactants'].split('.')

        if transfer_two and len(reactants_smiles) > 2:
            # Randomly select two SMILES strings for ground truth
            selected_smiles = random.sample(reactants_smiles, 2)
        else:
            # Selecting the longest or shortest SMILES string
            if select == 'longest':
                selected_smiles = [max(reactants_smiles, key=len)]
            elif select == 'shortest':
                selected_smiles = [min(reactants_smiles, key=len)]
            else:
                raise ValueError("select argument must be 'longest' or 'shortest'")

        # Remove the selected SMILES from reactants and update the dictionary
        for smiles in selected_smiles:
            reactants_smiles.remove(smiles)

        reaction_copy['reactants'] = '.'.join(reactants_smiles)
        reaction_copy['ground truth'] = '.'.join(selected_smiles)

        # Add the updated copy to the processed list
        processed_list.append(reaction_copy)
    
    return processed_list


if __name__ == '__main__':
    
   
    # 1. Golden dataset
    golden = pd.read_csv(f'{root_dir}/Data/Raw_data/Golden/raw_data.csv')
    golden['id'] = [f'golden_{str(x)}' for x in golden.index]
    golden.rename(columns={'mapped_rxn': 'reactions'}, inplace=True)
    balance_golden, unbalance_golden = get_unbalance(golden, n_jobs=4)
    unbalance_golden.to_csv(f'{root_dir}/Data/Validation_set/golden_dataset.csv')


    # 2. Nature
    complex = pd.read_csv(f'{root_dir}/Data/Raw_data/NatComm/complex.csv')
    complex['id'] = [f'complex_{str(x)}' for x in complex.index]
    patent = pd.read_csv(f'{root_dir}/Data/Raw_data/NatComm/patent.csv')
    patent['id'] = [f'patent_{str(x)}' for x in patent.index]
    typical = pd.read_csv(f'{root_dir}/Data/Raw_data/NatComm/typical.csv')
    typical['id'] = [f'typical_{str(x)}' for x in typical.index]

    nature = pd.concat([complex, patent, typical], axis=0)
    nature.drop(['mapped_reaction'], axis =1, inplace=True)
    nature.reset_index(drop=True, inplace=True)
    nature.rename(columns={'reaction':'reactions'}, inplace=True)
    balance_nature, unbalance_nature = get_unbalance(nature, n_jobs=4)
    unbalance_nature.to_csv(f'{root_dir}/Data/Validation_set/nature.csv')

    # 3. USPTO
    balance_USPTO = pd.DataFrame(load_database(f'{root_dir}/Data/balance_reactions.json.gz'))
    balance_USPTO['id'] = [f'USPTO{str(x)}' for x in balance_USPTO.index]
    balance_USPTO = balance_USPTO[['id', 'reactions']]


    # 4. Generate artificial validation set
    new_balance = pd.concat([balance_nature, balance_golden, balance_USPTO], axis=0)
    new_balance.reset_index(drop=True, inplace=True)
    process = RSMIProcessing(data=new_balance, rsmi_col='reactions', parallel=True, n_jobs=4, 
                             save_json =False, save_path_name=None)
    reactions = process.data_splitter().to_dict('records')

    valid_reactions = find_appropriate_reactions(reactions)

    # Process the data to get the longest SMILES
    valid1=generate_artificial_validation(valid_reactions, select='longest', transfer_two=False, random_state=42)
    print(len(valid1))
    # Process the data to get the shortest SMILES
    valid2=generate_artificial_validation(valid_reactions, select='shortest', transfer_two=False, random_state=42)
    print(len(valid1))
    save_database(valid1, f'{root_dir}/Data/Validation_set/artificial_valid1.json.gz')
    save_database(valid2, f'{root_dir}/Data/Validation_set/artificial_valid2.json.gz')
 



