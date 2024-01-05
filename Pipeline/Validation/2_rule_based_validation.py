import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
import os
from pathlib import Path

root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))

from SynRBL.SynExtract.rsmi_processing import RSMIProcessing
from SynRBL.SynCleaning import SMILESStandardizer
from SynRBL.SynExtract import RSMIDecomposer  
from SynRBL.SynExtract.rsmi_comparator import RSMIComparator
from SynRBL.SynExtract.rsmi_both_side_process import BothSideReact
from SynRBL.SynRuleImpute import SyntheticRuleImputer
from SynRBL.SynRuleImpute.synthetic_rule_constraint import RuleConstraint
from SynRBL.rsmi_utils import save_database, load_database, filter_data, extract_results_by_key, get_random_samples_by_key
from SynRBL.SynVis import ReactionVisualizer
from rdkit import  RDLogger
import rdkit
import matplotlib
from collections import defaultdict

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

def main(data_name = 'golden_dataset', n_jobs=4, save = False, rules_extension= False):


    df = pd.read_csv(root_dir /f'Data/Validation_set/{data_name}.csv')

    # Save solved and unsolved reactions
    save_dir = root_dir / 'Data/Validation_set' / data_name
    if not save_dir.exists():
        os.mkdir(save_dir)

    # process data
    process = RSMIProcessing(data=df, rsmi_col='reactions', parallel=True, n_jobs=n_jobs, 
                             save_json =False, save_path_name=None)
    reactions = process.data_splitter()

    # decompose into dict of symbols
    decompose = RSMIDecomposer(smiles=None, data=reactions, reactant_col='reactants', product_col='products', parallel=True, n_jobs=n_jobs, verbose=1)
    react_dict, product_dict = decompose.data_decomposer()

    # compare dict and check balance
    comp = RSMIComparator(reactants=react_dict, products=product_dict, n_jobs=n_jobs)
    unbalance, diff_formula = comp.run_parallel(reactants=react_dict, products=product_dict)

    # solve the both side reaction
    both_side = BothSideReact(react_dict, product_dict, unbalance, diff_formula)
    diff_formula, unbalance= both_side.fit()

    reactions_clean = pd.concat([pd.DataFrame(reactions), pd.DataFrame([unbalance]).T.rename(columns={0: 'Unbalance'}),
                                 pd.DataFrame([diff_formula]).T.rename(columns={0: 'Diff_formula'})], axis=1).to_dict(orient='records')

    save_database(reactions_clean, save_dir / 'reactions_clean.json.gz')
    if rules_extension:
        rules = load_database(root_dir / 'Data/Rules/rules_manager_extension.json.gz')
    else:
        rules = load_database(root_dir / 'Data/Rules/rules_manager.json.gz')

    # Filter data based on specified criteria
    no_C_reactions = filter_data(reactions_clean, unbalance_values=['Reactants', 'Products'], 
                                 formula_key='Diff_formula', element_key='C', min_count=0, max_count=0)
    print('Number of Non-Carbon Reactions Unbalanced in one side:', len(no_C_reactions))

    un_C_reactions = filter_data(reactions_clean, unbalance_values=['Reactants', 'Products'], 
                                 formula_key='Diff_formula', element_key='C', min_count=1, max_count=1000)
    print('Number of Carbon Reactions Unbalanced in one side:', len(un_C_reactions))

    both_side_reactions = filter_data(reactions_clean, unbalance_values=['Both'], 
                                      formula_key='Diff_formula', element_key='C', min_count=0, max_count=1000)
    print('Number of Both sides Unbalanced Reactions:', len(both_side_reactions))

    balance_reactions = filter_data(reactions_clean, unbalance_values=['Balance'], 
                                    formula_key='Diff_formula', element_key='C', min_count=0, max_count=1000)
    print('Number of Balanced Reactions:', len(balance_reactions))

    # Configure RDKit logging
    lg = RDLogger.logger()
    lg.setLevel(RDLogger.ERROR)
    RDLogger.DisableLog('rdApp.info') 
    rdkit.RDLogger.DisableLog('rdApp.*')

    # Initialize SyntheticRuleImputer and perform parallel imputation
    imp = SyntheticRuleImputer(rule_dict=rules, select='all', ranking='ion_priority')
    expected_result = imp.parallel_impute(no_C_reactions)

    # Extract solved and unsolved results
    solve, unsolve = extract_results_by_key(expected_result)
    print('Solved:', len(solve))
    print('Unsolved in rules based method:', len(unsolve))

    # Combine all unsolved cases
    unsolve = un_C_reactions + both_side_reactions + unsolve
    #print('Total unsolved:', len(unsolve))

    # Handle uncertainty in imputation
    constrain = RuleConstraint(solve, ban_atoms=['[O].[O]', 'F-F', 'Cl-Cl', 'Br-Br', 'I-I', 'Cl-Br', 'Cl-I', 'Br-I'])
    certain_reactions, uncertain_reactions = constrain.fit()

    id_uncertain = [entry['R-id'] for entry in uncertain_reactions]
    new_uncertain_reactions = [entry for entry in reactions_clean if entry['R-id'] in id_uncertain]

    unsolve = unsolve + new_uncertain_reactions
    print('Solved:', len(certain_reactions))
    print('Total unsolved:', len(unsolve))



    save_database(certain_reactions, save_dir / 'Solved_reactions.json.gz')
    save_database(unsolve, save_dir / 'Unsolved_reactions.json.gz')

    check_dir = root_dir / 'Data/Validation_set' / data_name / 'check'
    if not check_dir.exists():
        os.mkdir(check_dir)


    if data_name == 'USPTO_50K':
        check_dir_diff = root_dir / 'Data/Validation_set' / data_name / 'check_diff'
        if not check_dir_diff.exists():
            os.mkdir(check_dir_diff)
        certain_reactions_diff = get_random_samples_by_key(certain_reactions, num_samples_per_group=30, random_seed=42, stratify_key = 'Diff_formula')
        uncertain_reactions_diff = get_random_samples_by_key(uncertain_reactions, num_samples_per_group=30, random_seed=42, stratify_key = 'Diff_formula')

        save_database(certain_reactions_diff, save_dir / 'Solved_reactions_diff.json.gz')
        save_database(uncertain_reactions_diff, save_dir / 'Unsolved_reactions_diff.json.gz')
        vis = ReactionVisualizer()
        for i in range(0, len(certain_reactions_diff),1):
            vis.plot_reactions(certain_reactions_diff[i],'reactions', 'new_reaction', compare=True, savefig=True, pathname=check_dir_diff/ f'{i}.png')
            matplotlib.pyplot.close()


        certain_reactions_class = sample_reactions(certain_reactions, N=30, random_state=42)
        uncertain_reactions_class = sample_reactions(uncertain_reactions, N=30, random_state=42)
        save_database(certain_reactions_class, save_dir / 'Solved_reactions_class.json.gz')
        save_database(uncertain_reactions_class, save_dir / 'Unsolved_reactions_class.json.gz')
        
        
        certain_reactions = certain_reactions_class

    vis = ReactionVisualizer()
    for i in range(0, len(certain_reactions),1):
        vis.plot_reactions(certain_reactions[i],'reactions', 'new_reaction', compare=True, savefig=True, pathname=check_dir/ f'{i}.png')
        matplotlib.pyplot.close()

        

if __name__ == "__main__":
    #main('Jaworski')
    main('golden_dataset')
    #main('USPTO_50K')
    #main('USPTO_random_class')
