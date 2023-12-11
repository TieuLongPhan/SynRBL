import sys
from pathlib import Path

def main(n_jobs=4):
    root_dir = Path(__file__).parents[1]
    sys.path.append(str(root_dir))
    from SynRBL.rsmi_utils import save_database, load_database, filter_data, sort_by_key_length
    from SynRBL.SynRuleEngine.rule_data_manager import RuleImputeManager
    from SynRBL.SynRuleEngine import AutomaticSmilesExtraction, AutomaticRulesExtraction

    # 1 Manual Rules Extraction
    rules = load_database(root_dir / 'Data/Rules/rules_manager.json.gz')
    former_len = len(rules)
    db = RuleImputeManager(rules)

    entries = [{'formula': 'CO2', 'smiles': 'C=O'}, {'formula': 'Invalid', 'smiles': 'Invalid'}]
    invalid_entries = db.add_entries(entries)
    print(f"Invalid entries: {invalid_entries}")

    rules = filter_data(rules, formula_key='Composition', element_key='C', min_count=0, max_count=1)
    rules = sort_by_key_length(rules, lambda x: x['Composition'])

    if len(rules) > former_len:
        save_database(rules, root_dir / 'Data/Rules/rules_manager.json.gz')
    else:
        print("Not saved")


    # 2. Automated Rules Extraction
    reactions = load_database(root_dir / 'Data/reaction_clean.json.gz')

    # Create an instance of the AutomaticSmilesExtraction class with parallel processing
    smi_extractor = AutomaticSmilesExtraction(reactions, n_jobs=n_jobs, verbose=1)

    # Prepare input for get_fragments method
    input_dict = {
        'smiles': smi_extractor.smiles_list,
        'mw': smi_extractor.mw,
        'n_C': smi_extractor.n_C
    }

    # Extract filtered fragments
    filtered_fragments = AutomaticSmilesExtraction.get_fragments(input_dict, mw=500, n_C=0, combination='intersection')
    print("Filtered Fragments:", len(filtered_fragments))

    # Create an instance of the AutomaticRulesExtraction class
    extractor = AutomaticRulesExtraction(existing_database=[], n_jobs=n_jobs, verbose=1)

    # Add new entries to the extractor
    extractor.add_new_entries(filtered_fragments)

    # Extract rules from the added entries
    automated_rules = extractor.extract_rules()
    print("Extracted Rules:", len(automated_rules))

    save_database(automated_rules, root_dir / 'Data/Rules/automated_rules.json.gz')
        

if __name__ == "__main__":
    main()
