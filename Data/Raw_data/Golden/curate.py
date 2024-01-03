import sys
import pandas as pd
from pathlib import Path
root_dir = Path(__file__).parents[3]
sys.path.append(str(root_dir))
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynVis.reaction_visualizer import ReactionVisualizer

import re
from rdkit import Chem
def remove_atom_mapping(smiles, symbol = '>>'):
    # Regular expression to find atom mappings (numbers following a colon)
    mapping_pattern = re.compile(r':\d+')

    reactants, products = smiles.split(symbol)
    def remove_mapping(component):
        return mapping_pattern.sub('', component)

    # Apply the function to each component
    reactants = remove_mapping(reactants)
    products = remove_mapping(products)

    # Recombine the reaction
    return symbol.join([Chem.CanonSmiles(reactants), Chem.CanonSmiles(products)])


golden  = pd.read_csv(root_dir / 'Data/Raw_data/Golden/raw_data.csv')
golden['id'] = [f'golden_{str(x)}' for x in golden.index]
golden['reactions'] = golden['mapped_rxn'].apply(remove_atom_mapping)
data = golden.to_dict('records')

import matplotlib
vis = ReactionVisualizer()
for i in range(0, len(data),1):
    vis.plot_reactions(data[i],'mapped_rxn', 'reactions', compare=True, savefig=True, pathname=root_dir / f'Data/Raw_data/Golden/curate/{i}.png')
    matplotlib.pyplot.close()
