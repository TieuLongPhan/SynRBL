import sys
from pathlib import Path
root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
from SynRBL.rsmi_utils import load_database, save_database, filter_data
import pandas as pd
from SynRBL.SynProcessor import RSMIProcessing, RSMIDecomposer, RSMIComparator, BothSideReact, CheckCarbonBalance
import random
from typing import List, Dict, Optional, Tuple
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict


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
    comp = RSMIComparator(reactants=react_dict, products=product_dict, n_jobs=-2)
    unbalance, diff_formula = comp.run_parallel(reactants=react_dict, products=product_dict)

    # solve the both side reaction
    both_side = BothSideReact(react_dict, product_dict, unbalance, diff_formula)
    diff_formula, unbalance= both_side.fit()

    reactions_clean = pd.concat([pd.DataFrame(reactions), pd.DataFrame([unbalance]).T.rename(columns={0: 'Unbalance'}),
                                    pd.DataFrame([diff_formula]).T.rename(columns={0: 'Diff_formula'})], axis=1).to_dict(orient='records')
    
    unbalanced_reactions_clean = filter_data(reactions_clean, unbalance_values=['Reactants', 'Products', 'Both'], 
                                 formula_key='Diff_formula', element_key='C', min_count=0, max_count=100)
    balanced_reactions_clean = filter_data(reactions_clean, unbalance_values=['Balance'], 
                                 formula_key='Diff_formula', element_key='C', min_count=0, max_count=100)

    
    
    return balanced_reactions_clean, unbalanced_reactions_clean




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




def count_carbons(smiles):
    return smiles.count('C')

def sample_reactions(reactions, N, random_state=None):
    np.random.seed(random_state)

    grouped_reactions = defaultdict(lambda: defaultdict(list))
    for reaction in reactions:
        unbalance_category = reaction.get('Unbalance', 'None')
        grouped_reactions[reaction['class']][unbalance_category].append(reaction)

    sampled_reactions = []

    for class_, unbalance_groups in grouped_reactions.items():
        for unbalance, reactions in unbalance_groups.items():
            sample_size = min(N, len(reactions))  # Adjust sample size if necessary

            carbon_counts = [count_carbons(reaction['reactants']) for reaction in reactions]
            total_carbons = sum(carbon_counts)
            probabilities = [count / total_carbons for count in carbon_counts]

            sampled_indices = np.random.choice(len(reactions), size=sample_size, replace=False, p=probabilities)
            sampled_reactions.extend([reactions[i] for i in sampled_indices])

    return sampled_reactions


if __name__ == '__main__':
    
   
    # # 1. Golden dataset
    golden = pd.read_csv(f'{root_dir}/Data/Raw_data/Golden/Golden.csv')
    golden['id'] = [f'Golden_{str(x)}' for x in golden.index]
 
    # golden.to_csv(f'{root_dir}/Data/Validation_set/golden_dataset.csv', index=False)


    # # 2. Jaworski

    complex = pd.read_csv(f'{root_dir}/Data/Raw_data/Jaworski/complex.csv')
    complex['id'] = [f'complex_{str(x)}' for x in complex.index]
    patent = pd.read_csv(f'{root_dir}/Data/Raw_data/Jaworski/patent.csv')
    patent['id'] = [f'patent_{str(x)}' for x in patent.index]
    typical = pd.read_csv(f'{root_dir}/Data/Raw_data/Jaworski/typical.csv')
    typical['id'] = [f'typical_{str(x)}' for x in typical.index]

    Jaworski = pd.concat([complex, patent, typical], axis=0)
    Jaworski.drop(['mapped_reaction'], axis =1, inplace=True)
    Jaworski.reset_index(drop=True, inplace=True)
    Jaworski.rename(columns={'reaction':'reactions'}, inplace=True)

    # Jaworski.to_csv(f'{root_dir}/Data/Validation_set/Jaworski.csv')

    # 3. USPTO
    # USPTO_50K = pd.read_csv(f'{root_dir}/Data/Raw_data/USPTO/USPTO_50K.csv')
    # USPTO_balance, USPTO_unbalance = get_unbalance(USPTO_50K, n_jobs=4)
    # print(USPTO_balance)
    # sampled = sample_reactions(USPTO_unbalance, N=50, random_state=42)
    # sampled_df = pd.DataFrame(sampled)
    # sampled_df['id'] = sampled_df['R-id']
    # sampled_df =  sampled_df[['id','class', 'reactions']]
    #sampled_df.to_csv(f'{root_dir}/Data/Validation_set/USPTO_random_class.csv', index=False)
    #USPTO_50K.to_csv(f'{root_dir}/Data/Validation_set/USPTO_50K.csv', index=False)
 


    # 4. Generate artificial validation set for MCS-based validation
    golden.drop(['mapped_rxn'], axis =1, inplace=True)
    USPTO_50K = pd.read_csv(f'{root_dir}/Data/Raw_data/USPTO/USPTO_50K.csv')
    USPTO_50K.drop(['class'], axis =1, inplace=True)
    USPTO_50K.drop_duplicates(subset=['reactions'], inplace=True)
    USPTO_50K.reset_index(drop=True, inplace=True)
    USPTO_50K['id'] = [f'USPTO_50K_{str(x)}' for x in USPTO_50K.index]

    new_data = pd.concat([golden, Jaworski, USPTO_50K], axis=0)
    new_data.reset_index(drop=True, inplace=True)
    balance_data, _ = get_unbalance(new_data, n_jobs=4)
    balance_data = pd.DataFrame(balance_data)
    
    process = RSMIProcessing(data=balance_data, rsmi_col='reactions', parallel=True, n_jobs=4, 
                              save_json =False, save_path_name=None)
    reactions = process.data_splitter().to_dict('records')
    

    valid_reactions = find_appropriate_reactions(reactions)

    # Process the data to get the longest SMILES
    valid1=generate_artificial_validation(valid_reactions, select='longest', transfer_two=False, random_state=42)
    for item in valid1:
        item['carbon_balance_check'] = 'reactants'

    for key, value in enumerate(valid1):
        valid1[key]['reactions'] = '>>'.join([value['reactants'], value['products']])
    # Process the data to get the shortest SMILES
    valid2=generate_artificial_validation(valid_reactions, select='shortest', transfer_two=False, random_state=42)
    for item in valid2:
        item['carbon_balance_check'] = 'reactants'
    for key, value in enumerate(valid2):
        valid2[key]['reactions'] = '>>'.join([value['reactants'], value['products']])
    save_database(valid1, f'{root_dir}/Data/Validation_set/artificial_data_1/mcs_based_reactions.json.gz')
    save_database(valid2, f'{root_dir}/Data/Validation_set/artificial_data_2/mcs_based_reactions.json.gz')
 



