# SynRBL: Synthesis Rebalancing Framework

SynRBL (Synthesis Rebalancing Framework) is a specialized toolkit designed for computational chemistry. Its primary focus is on rebalancing incomplete chemical reactions and providing rule-based methodologies for data standardization and analysis.

![screenshot](./Image/test.png)


## Table of Contents
- [Repository Structure](#repository-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Repository Structure

SynRBL is organized into several key components, each dedicated to a specific aspect of chemical data processing:

### Main Components

- `SynRBL/`: Main package directory
  - `SynExtract/`: Data extraction module
  - `SynRuleEngine/`: Rule engine module
  - `SynRuleImpute/`: Rule-based imputation module
  - `SynMCS/`: MCS-based imputation module
  - `SynVis/`: Data visualization module

### Test Suite

- `tests/`: Test scripts and related files
  - `SynExtract/`: Tests for SynExtract module
  - `SynRuleEngine/`: Tests for SynRuleEngine module
  - `SynRuleImpute/`: Tests for SynRuleImpute module
  - `SynMCS/`: Tests for MCS-based imputation module
  - `SynVis/`: Tests for SynVis module

### Pipeline

- `Pipeline/`: Main scripts
  - `Notebook/`: Jupyter notebook examples
  - `Validation/`: Validation scripts


### Additional Resources

- `License`: License document
- `README.md`: Overview and documentation
- `setup.py`: Installation
- `.gitignore`: Configuration for ignoring certain files and directories



## Installation

To install and set up the SynRBL framework, follow these steps. Please ensure you have Python 3.9 or later installed on your system.

### Prerequisites

- Python 3.9+
- RDKit
- tmap
- map4

### Step-by-Step Installation Guide

1. **Python Installation:**
  Ensure that Python 3.9 or later is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. **Creating a Virtual Environment (Optional but Recommended):**
  It's recommended to use a virtual environment to avoid conflicts with other projects or system-wide packages. Use the following commands to create and activate a virtual environment:

  ```bash
  python -m venv synrbl-env
  source synrbl-env/bin/activate  # On Windows use `synrbl-env\Scripts\activate`
  ```
  Or Conda

  ```bash
  conda create --name synrbl-env python=3.9
  conda activate synrbl-env
  ```


3. **Installing Required Packages:**
  Install the necessary packages using pip. RDKit might require additional steps to install, which you can find in the RDKit documentation.

  ```bash
  pip install rdkit networkx pysmiles tmap map4
  ```

4. **Cloning and Installing SynRBL:**
  Clone the SynRBL repository from GitHub and install it:

  ```bash
  git clone https://github.com/TieuLongPhan/SynRBL.git
  cd SynRBL
  pip install .
  ```

5. **Verify Installation:**
  After installation, you can verify that SynRBL is correctly installed by running a simple test or checking the package version.

  ```python
  python -c "import SynRBL; print(SynRBL.__version__)"
  ```


## Usage

The SynRBL framework provides a comprehensive suite of tools for computational chemistry, focusing on synthesis rebalancing. There are two-main strategies: rule-based method for non-carbon compounds imputation, mcs-based method for carbon-compounds imputation.

### 1. Rule-based Imputation


#### Step 1: Processing Reaction SMILES (RSMI) Data
```python
from SynRBL.SynExtract.rsmi_processing import RSMIProcessing
import pandas as pd

# Sample DataFrame with reaction SMILES
df = pd.DataFrame({'reactions': ['CCCO>>CCC', 'CCO.Cl>>CCCl']})

# Initialize RSMIProcessing with DataFrame
process = RSMIProcessing(data=df, data_name='USPTO_50K', rsmi_col='reactions', parallel=True, n_jobs=10, 
                            save_json =False, save_path_name= '../../Data/reaction.json.gz')
processed_data = process.data_splitter().to_dict('records')

print(processed_data)
```

#### Step 2: Checking unbalance based on carbon number
```python
from SynRBL.SynProcessor import CheckCarbonBalance

check = CheckCarbonBalance(processed_data, rsmi_col='reactions', symbol='>>', atom_type='C', n_jobs=4)
processed_data = check.check_carbon_balance()

rules_based = [reactions[key] for key, value in enumerate(processed_data) if value['carbon_balance_check'] == 'balanced']
mcs_based = [reactions[key] for key, value in enumerate(processed_data) if value['carbon_balance_check'] != 'balanced']
print(len(rules_based), len(mcs_based))
```

#### Step 3: Molecular decomposer
```python
from SynRBL.SynProcessor import RSMIDecomposer  

decompose = RSMIDecomposer(smiles=None, data=rules_based, reactant_col='reactants', product_col='products', parallel=True, n_jobs=-1, verbose=1)
react_dict, product_dict = decompose.data_decomposer()
```

#### Step 4: Molecular comparator
```python
from SynRBL.SynProcessor import RSMIComparator
from SynRBL.SynUtils.data_utils import save_database, load_database
import pandas as pd

comp = RSMIComparator(reactants= react_dict, products=product_dict, n_jobs=-1)
unbalance , diff_formula= comp.run_parallel(reactants= react_dict, products=product_dict)
```

#### Step 5: Fix reactions with bothside missing compounds
```python
from SynRBL.SynProcessor import BothSideReact

both_side = BothSideReact(react_dict, product_dict, unbalance, diff_formula)
diff_formula, unbalance= both_side.fit()
reactions_clean = pd.concat([pd.DataFrame(reactions), pd.DataFrame([unbalance]).T.rename(columns={0:'Unbalance'}),
           pd.DataFrame([diff_formula]).T.rename(columns={0:'Diff_formula'})], axis=1).to_dict(orient='records')
reactions_clean[0]
```

### Step 6: Rule generation:

```python
from SynRBL.SynUtils.data_utils import save_database, load_database
from SynRBL.SynRuleImputer.rule_data_manager import RuleImputeManager
import pandas as pd

# Initialize RuleImputeManager without an existing database and add one entry
db = RuleImputeManager()
try:
    db.add_entry('H2O', 'O')  # Adding an entry to the database
except ValueError as e:
    print(e)

# Using an existing list of dictionaries to initialize the database
existing_database = [{'formula': 'H2O', 'smiles': 'O', 'composition': {1: 2, 8: 1}}]
db = RuleImputeManager(existing_database)
entries = [{'formula': 'CO2', 'smiles': 'C=O'}, {'formula': 'Invalid', 'smiles': 'Invalid'}]
invalid_entries = db.add_entries(entries)  # Adding multiple entries
print(f"Invalid entries: {invalid_entries}")

# Initializing with an existing pandas DataFrame
existing_dataframe = pd.DataFrame([{'formula': 'H2O', 'smiles': 'O', 'composition': {1: 2, 8: 1}}])
db = RuleImputeManager(existing_dataframe)
entries = [{'formula': 'CO2', 'smiles': 'C=O'}, {'formula': 'Invalid', 'smiles': 'Invalid'}]
invalid_entries = db.add_entries(entries)
print(f"Invalid entries: {invalid_entries}")

# Adding entries to a new database
db = RuleImputeManager()
entries = [{'formula': 'H2O', 'smiles': 'O'}, {'formula': 'Invalid', 'smiles': 'Invalid'}]
invalid_entries = db.add_entries(entries)
print(f"Invalid entries: {invalid_entries}")
```


### Step 7: Rebalancing

```python
from SynRBL.SynUtils.data_utils import save_database, load_database, filter_data, extract_results_by_key
from SynRBL.SynRuleImputer import SyntheticRuleImputer

rules = load_database('../../Data/Rules/rules_manager.json.gz')
#reactions_clean = load_database('../../Data/reaction_clean.json.gz')

# Filter data based on specified criteria

balance_reactions = filter_data(reactions_clean, unbalance_values=['Balance'], 
                                formula_key='Diff_formula', element_key=None, min_count=0, max_count=0)
print('Number of Balanced Reactions:', len(balance_reactions))

unbalance_reactions = filter_data(reactions_clean, unbalance_values=['Reactants', 'Products'], 
                                formula_key='Diff_formula', element_key=None, min_count=0, max_count=0)
print('Number of Unbalanced Reactions in one side:', len(unbalance_reactions))

both_side_reactions = filter_data(reactions_clean, unbalance_values=['Both'], 
                                    formula_key='Diff_formula', element_key=None, min_count=0, max_count=0)
print('Number of Both sides Unbalanced Reactions:', len(both_side_reactions))

# Configure RDKit logging
from rdkit import Chem
import rdkit
lg = RDLogger.logger()
lg.setLevel(RDLogger.ERROR)
RDLogger.DisableLog('rdApp.info') 
rdkit.RDLogger.DisableLog('rdApp.*')

# Initialize SyntheticRuleImputer and perform parallel imputation
imp = SyntheticRuleImputer(rule_dict=rules, select='all', ranking='ion_priority')
expected_result = imp.parallel_impute(unbalance_reactions)

# Extract solved and unsolved results
solve, unsolve = extract_results_by_key(expected_result)
print('Solved:', len(solve))
print('Unsolved in rules based method:', len(unsolve))



# Combine all unsolved cases
unsolve = both_side_reactions + unsolve
print('Total unsolved:', len(unsolve))
```

### Step 8: Uncertainty estimation

```python
from SynRBL.rsmi_utils import  save_database, load_database
from SynRBL.SynRuleImputer.synthetic_rule_constraint import RuleConstraint
constrain = RuleConstraint(solve, ban_atoms=['[H]','[O].[O]', 'F-F', 'Cl-Cl', 'Br-Br', 'I-I', 'Cl-Br', 'Cl-I', 'Br-I'])
certain_reactions, uncertain_reactions = constrain.fit()

id_uncertain = [entry['R-id'] for entry in uncertain_reactions]
new_uncertain_reactions = [entry for entry in reactions_clean if entry['R-id'] in id_uncertain]

unsolve = unsolve + new_uncertain_reactions


for d in unsolve:
    d.pop('Unbalance', None)  # Remove 'Unbalance' key if it exists
    d.pop('Diff_formula', None)  # Remove 'Diff_formula' key if it exists

mcs_based = mcs_based+unsolve
```


### Step 8: Visualizing Reactions
```python
from SynRBL.SynVis import ReactionVisualizer

# Initialize the ReactionVisualizer
visualizer = ReactionVisualizer()

# Visualize a reaction before and after processing
old_reaction = 'CCO>>CCOCC'
new_reaction = 'CC>>CC'

visualizer.plot_reactions(old_reaction, new_reaction)
```






## Features

- **SynProcess:** Automated extraction and decomposition of chemical data .
- **SynRuleImputer:** Application of rule-based algorithms for rebalancing non-carbon compounds.
- **SynMCSImputer:** Application of mcs-based algorithms for rebalancing carbon compounds.
- **SynChemImputer:** Application of domain knowlegde for rebalancing.
- **SynVis:** Advanced visualization tools for chemical data.

## Contributing

(Instructions for how to contribute to the SynRBL project.)

## License

This project is licensed under MIT License - see the [License](LICENSE) file for details.

## Acknowledgments

- Acknowledge contributors, inspirations, and any used resources.
