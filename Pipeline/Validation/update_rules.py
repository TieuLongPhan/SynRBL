import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
import os
from pathlib import Path

root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
from SynRBL.SynRuleEngine import AutomaticSmilesExtraction, AutomaticRulesExtraction
from SynRBL.rsmi_utils import load_database, save_database


data_name = [ 'golden_dataset', 'nature', 'USPTO_50K']
rules = load_database(root_dir / 'Data/Rules/rules_manager.json.gz')

for i in data_name:
    reactions = load_database(root_dir / f'Data/Validation_set/{i}/reactions_clean.json.gz')
    smi_extractor = AutomaticSmilesExtraction(reactions, n_jobs=-1, verbose=1)

    # Example usage of get_fragments
    input_dict = {
        'smiles': smi_extractor.smiles_list,
        'mw': smi_extractor.mw,
        'n_C': smi_extractor.n_C
    }
    filtered_fragments = AutomaticSmilesExtraction.get_fragments(input_dict, mw=500, n_C=0, combination='intersection')
    print("Filtered Fragments:", len(filtered_fragments))


    extractor = AutomaticRulesExtraction(existing_database=[], n_jobs=-1, verbose=1)
    extractor.add_new_entries(filtered_fragments)
    automated_rules = extractor.extract_rules()
    rules.extend(automated_rules)


rules_df = pd.DataFrame(rules)
rules_unique  = rules_df.drop_duplicates(subset=['Composition']).to_dict('records')


# Function to remove dictionaries from the list based on the 'formula' key
def remove_dict_by_formula(chemical_list, formula_to_remove):
    return [chem for chem in chemical_list if chem['formula'] != formula_to_remove]

# Specify the formula of the dictionary you want to remove
formula_to_remove = 'O2'  # For example, removing the dictionary with formula 'O'

# Remove the specified dictionaries from the list
rules_unique = remove_dict_by_formula(rules_unique, formula_to_remove)

print(len(rules_unique))

save_database(rules_unique, root_dir / 'Data/Rules/rules_manager_extension.json.gz')
