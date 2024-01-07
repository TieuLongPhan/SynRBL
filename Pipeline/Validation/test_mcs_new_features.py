import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import sys
import os
from pathlib import Path

root_dir = Path(__file__).parents[2]
sys.path.append(str(root_dir))
from SynRBL.SynUtils.chem_utils import CheckCarbonBalance
from SynRBL.SynMCS.SubStructure.mcs_process import single_mcs
from SynRBL.rsmi_utils import load_database, save_database
from SynRBL.SynMCS.MissingGraph.find_graph_dict import find_graph_dict



def main():
    rsmi_1 = {'reactions':'O=Cc1ccccc1>>O=C(c1ccccc1)C(O)c2ccccc2', 'R-id':'R1'}
    rsmi_2 = {'reactions':'COC(=O)C(Cc1ccc(Cl)c(Cl)c1)NC(=O)c1ccc(Cl)cc1NS(=O)(=O)c1cccc2nsnc12.O>>O=C(NC(Cc1ccc(Cl)c(Cl)c1)C(=O)O)c1ccc(Cl)cc1NS(=O)(=O)c1cccc2nsnc12',  'R-id':'R2'}
    rsmi_3 = {'reactions':'[CH2:9]([CH2:10][CH2:11][CH2:12][CH2:13][CH2:14][CH2:15][CH2:16][C:17]([O:18][CH2:19][CH3:20])=[O:21])[CH3:8].[CH:35]1=[CH:36][C:31](=[CH:32][CH:33]=[CH:34]1)[Si:23]([C:24]=2[CH:29]=[CH:28][CH:27]=[CH:26][CH:25]=2)([CH3:30])[Cl:22].[CH3:1][CH:2]([CH3:3])[N-:4][CH:5]([CH3:6])[CH3:7]>>[CH:29]=1[C:24](=[CH:25][CH:26]=[CH:27][CH:28]=1)[Si:23]([C:31]=2[CH:36]=[CH:35][CH:34]=[CH:33][CH:32]=2)([CH:16]([C:17](=[O:21])[O:18][CH2:19][CH3:20])[CH2:15][CH2:14][CH2:13][CH2:12][CH2:11][CH2:10][CH2:9][CH3:8])[CH3:30]',
        'R-id':'R3'}
    rsmi_1['reactants'], rsmi_1['products'] = rsmi_1['reactions'].split('>>')
    rsmi_2['reactants'], rsmi_2['products'] = rsmi_2['reactions'].split('>>')
    rsmi_3['reactants'], rsmi_3['products'] = rsmi_3['reactions'].split('>>')
    reactions = [rsmi_1, rsmi_2, rsmi_3]
    checker = CheckCarbonBalance(reactions, rsmi_col='reactions')
    checker.check_carbon_balance()
    mcs_results = [single_mcs(reaction_dict,  sort='MCIS', method='MCIS') for reaction_dict in reactions]
    save_database(mcs_results,root_dir / 'Pipeline/Validation/test_mcs_results.json.gz')

    graph_results  = find_graph_dict(msc_dict_path=root_dir / 'Pipeline/Validation/test_mcs_results.json.gz', 
                                                    save_path=root_dir / 'Pipeline/Validation/test_graph_results.json.gz')
    
    for key, value in enumerate(graph_results):
        graph_results[key]['mcs_results'] = mcs_results[key]['mcs_results']
        graph_results[key]['sorted_reactants'] = mcs_results[key]['sorted_reactants']
        graph_results[key]['carbon_balance_check'] = mcs_results[key]['carbon_balance_check']
    print(graph_results)






if __name__ == "__main__":
    main()

