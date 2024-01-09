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

### Additional Resources

- `License`: License document
- `README.md`: Overview and documentation
- `setup.py`: Installation
- `.gitignore`: Configuration for ignoring certain files and directories
- `Example.ipynb`: Jupyter Notebook with usage examples
- `Deployment.ipynb`: Jupyter Notebook for deployment guidance


## Installation

To install and set up the SynRBL framework, follow these steps. Please ensure you have Python 3.9 or later installed on your system.

### Prerequisites

- Python 3.9+
- RDKit
- NetworkX
- PySmiles
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

The SynRBL framework provides a comprehensive suite of tools for computational chemistry, focusing on synthesis rebalancing, data extraction, rule-based processing, and visualization. Below are some examples demonstrating the use of different modules within SynRBL.

### Example 1: Standardizing SMILES Strings

```python
import warnings
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  # Disables RDKit warnings globally
# Alternatively, you can catch warnings in a specific part of your code
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
from SynRBL.SynCleaning import SMILESStandardizer

# Initialize the SMILESStandardizer
standardizer = SMILESStandardizer()

# Single smiles
smiles = 'C1=CC=CC=C1'
standardized_smiles = standardizer.standardize_smiles(smiles)
print(standardized_smiles)

# Dict of SMILES
original_smiles = [{'id': 'US05849732',
  'class': 6,
  'reactions': 'COC(=O)[C@H](CCCCNC(=O)OCc1ccccc1)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O>>COC(=O)[C@H](CCCCN)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O',
  'reactants': 'COC(=O)[C@H](CCCCNC(=O)OCc1ccccc1)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O',
  'products': 'COC(=O)[C@H](CCCCN)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O'},...]

new_dict_standardized_smiles = standardizer.standardize_dict_smiles(data_input=original_smiles, key='reactants', visualize=False, parallel = True, n_jobs = 4,normalizer = standardizer.normalizer, tautomer = standardizer.tautomer, salt_remover = standardizer.salt_remover)
print("Standardized SMILES:", new_dict_standardized_smiles)
>> [Parallel(n_jobs=-1)]: Done 50016 out of 50016 | elapsed:  7.7min finished
```



### Example 2: Processing Reaction SMILES (RSMI) Data
```python
from SynRBL.SynExtract.rsmi_processing import RSMIProcessing
import pandas as pd

# Sample DataFrame with reaction SMILES
df = pd.DataFrame({'reactions': ['CCO>>CCOCC', 'CC>>C']})

# Initialize RSMIProcessing with DataFrame
process = RSMIProcessing(data=df, rsmi_col='reactions', parallel=True)
processed_data = process.data_splitter()

print(processed_data)
```

### Example 3: Rule generation:

```python
from SynRBL.rsmi_utils import *
from SynRBL.SynRuleEngine.rule_data_manager import RuleImputeManager
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


### Example 4: Rebalancing

```python
from SynRBL.SynRuleImpute import SyntheticRuleImputer

# Example: Initializing the SyntheticRuleImputer with a set of rules
# (Assuming `rules` is a dictionary containing your rule-based logic for imputation)
rules = {
    # Example rules (replace with actual rules from your domain knowledge)
    'H2O': {'smiles': 'O', 'composition': {1: 2, 8: 1}},
    'CO2': {'smiles': 'C=O', 'composition': {6: 1, 8: 2}}
}

imp = SyntheticRuleImputer(rule_dict=rules)

# Example: Imputing missing components in a dataset of chemical reactions
# (Assuming `reactions_clean` is a list or DataFrame containing reaction data)
[{'id': 'US05849732',
  'class': 6,
  'reactions': 'COC(=O)[C@H](CCCCNC(=O)OCc1ccccc1)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O>>COC(=O)[C@H](CCCCN)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O',
  'reactants': 'COC(=O)[C@H](CCCCNC(=O)OCc1ccccc1)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O',
  'products': 'COC(=O)[C@H](CCCCN)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O',
  'Unbalance': 'Products',
  'Diff_formula': {'C': 8, 'O': 2, 'H': 6}},
 {'id': 'US20120114765A1',
  'class': 2,
  'reactions': 'Nc1cccc2cnccc12.O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1>>O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1',
  'reactants': 'Nc1cccc2cnccc12.O=C(O)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1',
  'products': 'O=C(Nc1cccc2cnccc12)c1cc([N+](=O)[O-])c(Sc2c(Cl)cncc2Cl)s1',
  'Unbalance': 'Products',
  'Diff_formula': {'O': 1, 'H': 2}},]

# Selecting a subset of reactions for imputation
subset_for_imputation = reactions_clean

# Performing the imputation
dict_impute = imp.impute(missing_dict=subset_for_imputation)
print(dict_impute)
```


### Example 5: Visualizing Reactions
```python
from SynRBL.SynVis import ReactionVisualizer

# Initialize the ReactionVisualizer
visualizer = ReactionVisualizer()

# Visualize a reaction before and after processing
old_reaction = 'CCO>>CCOCC'
new_reaction = 'CC>>CC'

visualizer.plot_reactions(old_reaction, new_reaction)
```

### Example 6: Finding Most Similar Molecule
```python
from SynRBL.SynRuleImpute import FormulaSimilarityFinder

# Reference SMILES and list of candidate molecules
ref_smiles = 'CC(C)CCOC(C)=O'
candidates = ['CCCCCO', 'CCCC(O)C', 'CCC(O)CC', 'CC(O)(C)CC', 'CC(C)CCO', 'CC(C)(C)CO']

# Initialize the Formula Similarity Finder
similarity_finder = FormulaSimilarityFinder(ref_smiles)

# Find the most similar molecule
most_similar = similarity_finder.find_most_similar(candidates)
print("Most similar molecule:", most_similar)
```




## Features

- **SynClearing:** Data cleaning and preprocessing tools.
- **SynExtract:** Automated extraction of chemical data.
- **SynRuleEngine:** Application of rule-based algorithms for data analysis.
- **SynVis:** Advanced visualization tools for chemical data.

## Contributing

(Instructions for how to contribute to the SynRBL project.)

## License

This project is licensed under MIT License - see the [License](LICENSE) file for details.

## Acknowledgments

- Acknowledge contributors, inspirations, and any used resources.
