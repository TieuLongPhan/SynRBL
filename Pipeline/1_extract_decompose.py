import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
from pathlib import Path

def main(standardize = False, n_jobs=4, save = True):
    root_dir = Path(__file__).parents[1]
    sys.path.append(str(root_dir))

    from SynRBL.SynProcessor import RSMIProcessing, RSMIDecomposer, RSMIComparator, BothSideReact
    from SynRBL.SynUtils.data_utils import save_database

    df = pd.read_csv(root_dir /'Data/USPTO_50K.csv')

    # process data
    process = RSMIProcessing(data=df, rsmi_col='reactions', parallel=True, n_jobs=n_jobs, 
                             save_json =False, save_path_name=root_dir / 'Data/reaction.json.gz')
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
    if save:
        save_database(reactions_clean, root_dir / 'Data/reaction_clean.json.gz')

if __name__ == "__main__":
    main()
