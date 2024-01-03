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
    constrain = RuleConstraint(solve, ban_atoms=['[H]','[O].[O]', 'F-F', 'Cl-Cl', 'Br-Br', 'I-I', 'Cl-Br', 'Cl-I', 'Br-I'])
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
        certain_reactions = get_random_samples_by_key(certain_reactions, num_samples_per_group=10, random_seed=42, stratify_key = 'Diff_formula')
        print(len(certain_reactions))

    vis = ReactionVisualizer()
    for i in range(0, len(certain_reactions),1):
        vis.plot_reactions(certain_reactions[i],'reactions', 'new_reaction', compare=True, savefig=True, pathname=check_dir/ f'{i}.png')
        matplotlib.pyplot.close()

        

if __name__ == "__main__":
    #main('golden_dataset')
    #main('nature')
    main('USPTO_50K')
