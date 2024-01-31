import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
import os
from pathlib import Path

root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))

from SynRBL.SynProcessor import RSMIProcessing, RSMIDecomposer, RSMIComparator, BothSideReact, CheckCarbonBalance

from SynRBL.SynRuleImputer import SyntheticRuleImputer
from SynRBL.SynRuleImputer.synthetic_rule_constraint import RuleConstraint
from SynRBL.rsmi_utils import save_database, load_database, filter_data, extract_results_by_key, get_random_samples_by_key
from SynRBL.SynVis import ReactionVisualizer
from rdkit import  RDLogger
import rdkit
import matplotlib
from collections import defaultdict

def count_carbons(smiles):
    return smiles.count('C')

def sample_reactions_by_class(reactions, N, random_state=None):
    np.random.seed(random_state)

    grouped_reactions = defaultdict(list)
    for reaction in reactions:
        grouped_reactions[reaction['class']].append(reaction)

    sampled_reactions = []

    for class_, class_reactions in grouped_reactions.items():
        sample_size = min(N, len(class_reactions))  # Adjust sample size if necessary

        carbon_counts = [count_carbons(reaction['reactants']) for reaction in class_reactions]
        total_carbons = sum(carbon_counts)
        probabilities = [count / total_carbons for count in carbon_counts]

        sampled_indices = np.random.choice(len(class_reactions), size=sample_size, replace=False, p=probabilities)
        sampled_reactions.extend([class_reactions[i] for i in sampled_indices])

    return sampled_reactions


def main(data_name = 'golden_dataset', n_jobs= 4, save = False, rules_extension= False):


    df = pd.read_csv(root_dir /f'Data/Validation_set/{data_name}.csv')
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)

    # Save solved and unsolved reactions
    save_dir = root_dir / 'Data/Validation_set' / data_name
    if not save_dir.exists():
        os.mkdir(save_dir)

    # 1. process data
    process = RSMIProcessing(data=df, rsmi_col='reactions', parallel=True, n_jobs=n_jobs, data_name=data_name,
                             save_json =False, save_path_name=None)
    reactions = process.data_splitter().to_dict('records')


    # 2. check carbon balance
    check = CheckCarbonBalance(reactions, rsmi_col='reactions', symbol='>>', atom_type='C', n_jobs=n_jobs)
    reactions = check.check_carbon_balance()
    rules_based = [reactions[key] for key, value in enumerate(reactions) if value['carbon_balance_check'] == 'balanced']
    mcs_based = [reactions[key] for key, value in enumerate(reactions) if value['carbon_balance_check'] != 'balanced']
    
    # 3. decompose into dict of symbols
    decompose = RSMIDecomposer(smiles=None, data=rules_based, reactant_col='reactants', product_col='products', parallel=True, n_jobs=n_jobs, verbose=1)
    react_dict, product_dict = decompose.data_decomposer()

    # 4. compare dict and check balance
    comp = RSMIComparator(reactants=react_dict, products=product_dict, n_jobs=n_jobs)
    unbalance, diff_formula = comp.run_parallel(reactants=react_dict, products=product_dict)

    # 5. solve the both side reaction
    both_side = BothSideReact(react_dict, product_dict, unbalance, diff_formula)
    diff_formula, unbalance= both_side.fit(n_jobs=n_jobs)

    reactions_clean = pd.concat([pd.DataFrame(rules_based), pd.DataFrame([unbalance]).T.rename(columns={0: 'Unbalance'}),
                                 pd.DataFrame([diff_formula]).T.rename(columns={0: 'Diff_formula'})], axis=1).to_dict(orient='records')

    save_database(reactions_clean, save_dir / 'reactions_clean.json.gz')
    if rules_extension:
        rules = load_database(root_dir / 'Data/Rules/rules_manager_extension.json.gz')
    else:
        rules = load_database(root_dir / 'Data/Rules/rules_manager.json.gz')

    # 6. Filter data based on specified criteria
    balance_reactions = filter_data(reactions_clean, unbalance_values=['Balance'], 
                                    formula_key='Diff_formula', element_key=None, min_count=0, max_count=0)
    print('Number of Balanced Reactions:', len(balance_reactions))

    unbalance_reactions = filter_data(reactions_clean, unbalance_values=['Reactants', 'Products'], 
                                    formula_key='Diff_formula', element_key=None, min_count=0, max_count=0)
    print('Number of Unbalanced Reactions in one side:', len(unbalance_reactions))

    both_side_reactions = filter_data(reactions_clean, unbalance_values=['Both'], 
                                        formula_key='Diff_formula', element_key=None, min_count=0, max_count=0)
    print('Number of Both sides Unbalanced Reactions:', len(both_side_reactions))

    # 7. Rule-based imputeration - Configure RDKit logging
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR)
    RDLogger.DisableLog('rdApp.info') 
    rdkit.RDLogger.DisableLog('rdApp.*')

    # Initialize SyntheticRuleImputer and perform parallel imputation
    imp = SyntheticRuleImputer(rule_dict=rules, select='all', ranking='ion_priority')
    expected_result = imp.parallel_impute(unbalance_reactions, n_jobs=n_jobs)

    # Extract solved and unsolved results
    solve, unsolve = extract_results_by_key(expected_result)
    print('Solved:', len(solve))
    print('Unsolved in rules based method:', len(unsolve))

    unsolve = both_side_reactions + unsolve
    print('Total unsolved:', len(unsolve))


    # 8. Handle uncertainty in imputation
    constrain = RuleConstraint(solve, ban_atoms=['[O].[O]', 'F-F', 'Cl-Cl', 'Br-Br', 'I-I', 'Cl-Br', 'Cl-I', 'Br-I'])
    certain_reactions, uncertain_reactions = constrain.fit()

    id_uncertain = [entry['R-id'] for entry in uncertain_reactions]
    new_uncertain_reactions = [entry for entry in reactions_clean if entry['R-id'] in id_uncertain]

    unsolve = unsolve + new_uncertain_reactions
    mcs_based = mcs_based+unsolve



    # if data_name == 'USPTO_50K':
    #     USPTO_diff = root_dir / 'Data/Validation_set' / 'USPTO_diff'
    #     USPTO_diff_check = root_dir / 'Data/Validation_set' / 'USPTO_diff' / 'check'
    #     if not USPTO_diff_check.exists():
    #         os.mkdir(USPTO_diff_check)
    #     certain_reactions_diff = get_random_samples_by_key(certain_reactions, num_samples_per_group=30, random_seed=42, stratify_key = 'Diff_formula')
    #     uncertain_reactions_diff = get_random_samples_by_key(mcs_based, num_samples_per_group=30, random_seed=42, stratify_key = 'Diff_formula')
    #     print(len(uncertain_reactions_diff))
    #     print(len(certain_reactions_diff))
        # save_database(certain_reactions_diff, USPTO_diff / 'rule_based_reactions.json.gz')
        # save_database(uncertain_reactions_diff, USPTO_diff / 'mcs_based_reactions.json.gz')
        # vis = ReactionVisualizer()
        # for i in range(0, len(certain_reactions_diff),1):
        #     vis.plot_reactions(certain_reactions_diff[i],'reactions', 'new_reaction', compare=True, savefig=True, pathname=USPTO_diff_check/ f'{i}.png')
        #     matplotlib.pyplot.close()

        
        
        # USPTO_unb = root_dir / 'Data/Validation_set' / 'USPTO_unbalance_class'
        # USPTO_unb_check = root_dir / 'Data/Validation_set' / 'USPTO_unbalance_class' / 'check'
        # certain_reactions_class = sample_reactions_by_class(certain_reactions, N=30, random_state=42)
        # uncertain_reactions_class = sample_reactions_by_class(mcs_based, N=30, random_state=42)
        # print(len(uncertain_reactions_class))
        # save_database(certain_reactions_class, USPTO_unb / 'rule_based_reactions.json.gz')
        # save_database(uncertain_reactions_class, USPTO_unb / 'mcs_based_reactions.json.gz')
        
        
        # certain_reactions = certain_reactions_class

    # for d in mcs_based:
    #     d.pop('Unbalance', None)  # Remove 'Unbalance' key if it exists
    #     d.pop('Diff_formula', None)  # Remove 'Diff_formula' key if it exists

   
    print('Solved reactions by rule based method:', len(certain_reactions))
    print('Reactions for MCS based method:', len(mcs_based))



    # save_database(certain_reactions, save_dir / 'rule_based_reactions.json.gz')
    # save_database(mcs_based, save_dir / 'mcs_based_reactions.json.gz')

    # check_dir = root_dir / 'Data/Validation_set' / data_name / 'check'
    # if not check_dir.exists():
    #     os.mkdir(check_dir)

    # vis = ReactionVisualizer()
    # for i in range(0, len(certain_reactions),1):
    #     vis.plot_reactions(certain_reactions[i],'reactions', 'new_reaction', compare=True, savefig=True, pathname=check_dir/ f'{i}.png')
    #     #vis.plot_reactions(certain_reactions[i],'reactions', 'new_reaction', compare=True, savefig=True, pathname=USPTO_unb_check/ f'{i}.png')
    #     matplotlib.pyplot.close()

        

if __name__ == "__main__":
    import time

    # get the start time
    st = time.time()
    #main('Jaworski', n_jobs=-1)
    #main('golden_dataset', n_jobs=-1)
    #main('USPTO_50K', n_jobs=-1)
    #main('USPTO_random_class', n_jobs=-1)
    main('USPTO_diff', n_jobs=-1)
    #main('USPTO_unbalance_class', n_jobs=-1)

        # get the end time
    et = time.time()

    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')
