import sys
from pathlib import Path
from rdkit import  RDLogger
import rdkit

def main(n_jobs=4):
    root_dir = Path(__file__).parents[1]
    sys.path.append(str(root_dir))

    from SynRBL.rsmi_utils import save_database, load_database, filter_data, extract_results_by_key
    from SynRBL.SynRuleImpute import SyntheticRuleImputer

    rules = load_database(root_dir / 'Data/Rules/rules_manager.json.gz')
    reactions_clean = load_database(root_dir / 'Data/reaction_clean.json.gz')

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
    print('Total unsolved:', len(unsolve))

    # Save solved and unsolved reactions
    save_database(solve, root_dir / 'Data/Solved_reactions.json.gz')
    save_database(unsolve, root_dir / 'Data/Unsolved_reactions.json.gz')


if __name__ == "__main__":
    main()