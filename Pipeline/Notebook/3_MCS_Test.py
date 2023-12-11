
import sys
sys.path.append('../')
from SynRBL.rsmi_utils import *
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from SynRBL.rsmi_utils import load_database, filter_data
reactions_clean = load_database('./Data/reaction_clean.json.gz')
filtered_data_1 = filter_data(reactions_clean, unbalance_values=['Reactants', 'Products'], formula_key='Diff_formula', element_key='C', min_count=1, max_count=10000)
len(filtered_data_1)

# |%%--%%| <sYmiwVBJuR|SfQ8Pw6yON>

filtered_data_2 = filter_data(reactions_clean, unbalance_values=['Both'], formula_key='Diff_formula', element_key='C', min_count=0, max_count=10000)
len(filtered_data_2)

# |%%--%%| <SfQ8Pw6yON|xssb3Ws5SX>

filtered_data = filtered_data_1 + filtered_data_2
len(filtered_data)

# |%%--%%| <xssb3Ws5SX|YNz0vyeG7x>

from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit.Chem import rdFMCS

class MCSMissingGraphAnalyzer:
    """A class for detecting missing graph in reactants and products using MCS and RDKit."""

    def __init__(self):
        """Initialize the MolecularOperations class."""
        pass
    
    @staticmethod
    def get_smiles(reaction_dict):
        """
        Extract reactant and product SMILES strings from a reaction dictionary.

        Parameters:
        - reaction_dict: dict
            A dictionary containing 'reactants' and 'products' as keys.

        Returns:
        - tuple
            A tuple containing reactant SMILES and product SMILES strings.
        """
        return reaction_dict['reactants'], reaction_dict['products']

    @staticmethod
    def convert_smiles_to_molecule(smiles):
        """
        Convert a SMILES string to an RDKit molecule object.

        Parameters:
        - smiles: str
            The SMILES string representing a molecule.

        Returns:
        - rdkit.Chem.Mol
            The RDKit molecule object.
        """
        return Chem.MolFromSmiles(smiles)

    @staticmethod
    def mol_to_smiles(mol):
        """
        Convert an RDKit molecule object to a SMILES string.

        Parameters:
        - mol: rdkit.Chem.Mol
            The RDKit molecule object.

        Returns:
        - str or None
            The SMILES string representation of the molecule, or None if the molecule is None.
        """
        return Chem.MolToSmiles(mol) if mol else None

    @staticmethod
    def mol_to_smarts(mol):
        """
        Convert an RDKit molecule object to a SMARTS string.

        Parameters:
        - mol: rdkit.Chem.Mol
            The RDKit molecule object.

        Returns:
        - str or None
            The SMARTS string representation of the molecule, or None if the molecule is None.
        """
        return Chem.MolToSmarts(mol) if mol else None

    @staticmethod
    def find_maximum_common_substructure(mol1, mol2, ringMatchesRingOnly=True):
        """
        Find the maximum common substructure (MCS) between two molecules.

        Parameters:
        - mol1, mol2: rdkit.Chem.Mol
            The RDKit molecule objects to compare.

        Returns:
        - rdkit.Chem.Mol or None
            The RDKit molecule object representing the MCS, or None if MCS search was canceled.
        """
        mcs_result = rdFMCS.FindMCS([mol1, mol2], ringMatchesRingOnly=ringMatchesRingOnly)
        if mcs_result.canceled:
            return None
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        return mcs_mol

    def IterativeMCSReactionPairs(reactant_mol_list, product_mol, params=None):
        """
        Find the MCS for each reactant fragment with the product, updating the product after each step.
        Sorts the reactants based on the size of their MCS with the product.

        Parameters:
        - reactant_mol_list: list of rdkit.Chem.Mol
            List of RDKit molecule objects for reactants.
        - product_mol: rdkit.Chem.Mol
            RDKit molecule object for the product.

        Returns:
        - list of rdkit.Chem.Mol
            List of RDKit molecule objects representing the MCS for each reactant-product pair.
        - list of rdkit.Chem.Mol
            Sorted list of reactant molecule objects.
        """
        # Calculate the MCS for each reactant with the product
        mcs_results = [(reactant, rdFMCS.FindMCS([reactant, product_mol], params)) for reactant in reactant_mol_list]

        # Filter out any canceled MCS results and sort by size of MCS
        mcs_results = [(reactant, mcs_result) for reactant, mcs_result in mcs_results if not mcs_result.canceled]
        sorted_reactants = sorted(mcs_results, key=lambda x: x[1].numAtoms, reverse=True)

        mcs_list = []
        current_product = product_mol

        # Process the sorted reactants
        for reactant, mcs_result in sorted_reactants:
            mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
            mcs_list.append(mcs_mol)

            # Update the product by removing the MCS substructure
            current_product = Chem.DeleteSubstructs(Chem.RWMol(current_product), mcs_mol)
            current_product = Chem.RemoveHs(current_product)
            try:
                Chem.SanitizeMol(current_product)
            except:
                pass

        # Extract only the reactant molecules from sorted_reactants for return
        sorted_reactant_mols = [reactant for reactant, _ in sorted_reactants]

        return mcs_list, sorted_reactant_mols

    
    @staticmethod
    def add_hydrogens_to_radicals(mol):
        """
        Add hydrogen atoms to radical sites in a molecule.

        Parameters:
        - mol: rdkit.Chem.Mol
            RDKit molecule object.

        Returns:
        - rdkit.Chem.Mol
            The modified molecule with added hydrogens.
        """
        if mol:
            # Create a copy of the molecule
            mol_with_h = Chem.RWMol(mol)

            # Add explicit hydrogens (not necessary if they are already present in the input molecule)
            mol_with_h = rdmolops.AddHs(mol_with_h)

            # Find and process radical atoms
            for atom in mol_with_h.GetAtoms():
                num_radical_electrons = atom.GetNumRadicalElectrons()
                if num_radical_electrons > 0:
                    atom.SetNumExplicitHs(atom.GetNumExplicitHs() + num_radical_electrons)
                    atom.SetNumRadicalElectrons(0)
            curate_mol = Chem.RemoveHs(mol_with_h)
            # Return the molecule with added hydrogens
            return curate_mol

    @staticmethod
    def fit(reaction_dict, params=None):
        """
        Process a reaction dictionary to find MCS, missing parts in reactants and products.

        Parameters:
        - reaction_dict: dict
            A dictionary containing 'reactants' and 'products' as keys.

        Returns:
        - tuple
            A tuple containing lists of MCS, missing parts in reactants, missing parts in products,
            reactant molecules, and product molecules.
        """
        reactant_smiles, product_smiles = MCSMissingGraphAnalyzer.get_smiles(reaction_dict)
        reactant_mol_list = [MCSMissingGraphAnalyzer.convert_smiles_to_molecule(smiles) for smiles in reactant_smiles.split('.')]
        product_mol = MCSMissingGraphAnalyzer.convert_smiles_to_molecule(product_smiles)

        mcs_list, sorted_reactants = MCSMissingGraphAnalyzer.IterativeMCSReactionPairs(reactant_mol_list, product_mol,  params)

        return mcs_list , sorted_reactants, product_mol

# |%%--%%| <YNz0vyeG7x|NVcAbOkYAT>

from rdkit.Chem import AllChem, rdChemReactions
#from SynRBL.SynMCS.mcs_missing_graph_analyzer import MCSMissingGraphAnalyzer
from SynRBL.SynMCS.find_missing_graphs import find_missing_parts_pairs 

def display(chem):
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw
    img = Draw.MolToImage(chem)
    fig, ax = plt.subplots(1, 1, figsize=[6,2])
    ax.imshow(img)
    plt.show()

def display_reaction(reaction_dict, reaction_key='reactions', use_smiles=True):
    """
    Displays a chemical reaction using RDKit.

    Parameters:
    - reaction_dict: A dictionary containing reaction data.
    - reaction_key: Key to access reaction information in the dictionary.
    - use_smiles: Whether to use SMILES format for the reaction.
    """
    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw
    r = rdChemReactions.ReactionFromSmarts(reaction_dict[reaction_key], useSmiles=use_smiles)
    img = Draw.ReactionToImage(r)
    plt.imshow(img)
    plt.show()
    return 

def test_case(data_list, indices, params=None, display_reactions=True):
    """
    Tests and displays reactions for a given set of indices.

    Parameters:
    - data_list: List containing reaction data.
    - indices: List of indices or a single index to process.
    - params: Optional parameters for MCS analysis.
    - display_reactions: Flag to control the display of reactions.
    """
    from collections import defaultdict
    import traceback

    # Ensure indices is a list even if a single index is provided
    if not isinstance(indices, list):
        indices = [indices]

    len_dict = defaultdict(lambda: 0)
    for index in indices:
        if display_reactions:
            display_reaction(data_list[index])

        test_reaction = data_list[index]
        analyzer = MCSMissingGraphAnalyzer()
        try:
            mcs_list, sorted_reactants, product_mol = analyzer.fit(test_reaction, params=params)
            impute_product_frags, boundary_atoms_products, nearest_neighbor_products = find_missing_parts_pairs(sorted_reactants, mcs_list)
        except Exception as e:
            print(f"Error {type(e).__name__} in reaction {index}.")
            continue

        l = len(impute_product_frags)
        len_dict[str(l)] += 1
        if l == 0 or l > 2:
            display_reaction(test_reaction)
            for i, frag in enumerate(impute_product_frags):
                display(frag)
                print('Boundary list for fragment', i, ':', boundary_atoms_products[i])
                print('Nearest neighbor list for fragment', i, ':', nearest_neighbor_products[i])

    print(len_dict.items())



# |%%--%%| <NVcAbOkYAT|uqaPDGXJ7D>

#test_case(data_list=filtered_data, indices=list(range(0,1000)), params=None, display_reactions=False)

# |%%--%%| <uqaPDGXJ7D|0Ld2NEsqkC>

Fail = [8]

rare = [13]
#|%%--%%| <0Ld2NEsqkC|gXEx0Epyj6>
from rdkit.Chem import rdmolops
from SynRBL.SynMCS.mol_merge import merge, merge_mols, merge_expand, plot_mols
import matplotlib.pyplot as plt
from collections import defaultdict

def _normalize(impute_product_frags, boundary_atoms_products, nearest_neighbor_products):
    frags = []
    boundary_atoms = []
    neighbors = []
    print(impute_product_frags, boundary_atoms_products)
    assert len(impute_product_frags) == len(boundary_atoms_products)
    assert len(impute_product_frags) == len(nearest_neighbor_products)
    for mol, bounds, nns in zip(impute_product_frags, boundary_atoms_products, nearest_neighbor_products):
        mol_frags = rdmolops.GetMolFrags(mol, asMols=True)
        if len(mol_frags) != len(bounds) or len(mol_frags) != len(nns):
            raise ValueError('Substructure mismatch.')
        frags.extend(mol_frags)
        _bounds= []
        _nns = []
        for bi, ni in zip(bounds, nns):
            assert len(bi.keys()) == 1
            _bounds.append(list(bi.values())[0]) 
            _nns.append([k for k in ni.keys()])
        boundary_atoms.append(_bounds)
        neighbors.append(_nns)
    return frags, boundary_atoms, neighbors

def impute_single(frags, bounds, neighbors, verbose=False):
    frags_len = len(frags)
    frags, bounds, neighbors = _normalize(frags, bounds, neighbors)
    results = []
    if len(frags) != frags_len:
        for f, b, n in zip(frags, bounds, neighbors):
            results.append(impute_single([f], b, n))
    else:
        m1 = frags[0]
        if verbose:
            plot_mols([m1], includeAtomNumbers=False, figsize=(4,1))
            plt.show()
            print('Mol1={}'.format(Chem.MolToSmiles(m1)))
        b1 = bounds[0]
        n1 = neighbors[0]
        m2 = merge_expand(m1, b1, n1)
        if verbose:
            plot_mols([m1, Chem.RemoveHs(m2['mol'])], includeAtomNumbers=False, figsize=(1,1))
        results.append(m2['mol'])
    return results

def impute_double(frags, bounds, neighbors, verbose=False):
    frags, bounds, neighbors = _normalize(frags, bounds, neighbors)
    m1, m2 = frags[0], frags[1]
    if verbose:
        print('Mol1={} Mol2={}'.format(Chem.MolToSmiles(m1), Chem.MolToSmiles(m2)))
    b1, b2 = bounds[0], bounds[1]
    n1, n2 = neighbors[0], neighbors[1]
    i1 = b1[0]
    i2 = b2[0]
    merge_result = merge_mols(m1, m2, i1, i2)
    m3 = Chem.RemoveHs(merge_result['mol'])
    if verbose:
        plot_mols([m1, m2, m3], includeAtomNumbers=False, figsize=(1,1))
        plt.show()

idx = 1
def impute(data, idx, verbose=False):
    analyzer = MCSMissingGraphAnalyzer()
    reaction = data[idx]
    mcs_list, sorted_reactants, product_mol = analyzer.fit(reaction)

    frags, bounds, neighbors = find_missing_parts_pairs(sorted_reactants, mcs_list)
    if verbose:
        print('----------')
        print("Idx={}({}) boundary={} neighbors={}".format(idx, len(frags), bounds, neighbors))
        for i, f in enumerate(frags):
            print("  Mol {}: {}".format(i, Chem.MolToSmiles(f)))
        display_reaction(reaction)
        plot_mols(frags, figsize=(4, 1), titles=["Input" for _ in range(len(frags))])
        plt.show()
    mmol = merge(frags, bounds, neighbors)
    if verbose:
        plot_mols([Chem.RemoveHs(m['mol']) for m in mmol], figsize=(3, 1), titles=['Merged' for _ in range(len(mmol))])
        plt.show()
    return mmol


s = 1000
n = 0
correct = []
incorrect = []
crules = defaultdict(lambda: [])
mrules = defaultdict(lambda: [])
for i in range(n, n + s):
    try:
        mmols = impute(filtered_data, i)
        for mmol in mmols:
            for r in mmol.get('compound_rules', []):
                crules[r.name].append(i)
            for r in mmol.get('merge_rules', []):
                mrules[r.name].append(i)
        correct.append(i) 
    except Exception as e:
        #import traceback
        #traceback.print_exc()
        print('[{}]'.format(i), e)
        incorrect.append(i)

#|%%--%%| <gXEx0Epyj6|InYhbn1oVE>

print('Compound rule useage:')
for k, v in crules.items():
    print("  Rule '{}' was used {} times.".format(k, len(v)))
print('Merge rule useage:')
for k, v in mrules.items():
    print("  Rule '{}' was used {} times.".format(k, len(v)))
print('Correct merges:', len(correct))
print('Extracted incorrect:', len(incorrect))

#|%%--%%| <InYhbn1oVE|ErWqmJQDiS>
import random

#indices = crules['append O to Si'][0:5]
#indices = crules['append O to C-C bond'][0:5] 
indices = mrules['halogen bond restriction']
#indices = mrules['silicium radical'][0:5]
#indices = incorrect #[11, 20, 59, 64, 90]
#indices = random.choices(correct, k=5)

for i in indices:
    try:
        impute(filtered_data, i, verbose=True) 
    except Exception as e:
        #import traceback
        #traceback.print_exc()
        print('[{}]'.format(i), e)
