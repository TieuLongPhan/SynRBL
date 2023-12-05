import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

def plot_mols(mols, includeAtomNumbers=False):
    if type(mols) is not list:
        mols = [mols]
    fig, ax = plt.subplots(1, len(mols))
    for i, mol in enumerate(mols):
        a = ax
        if len(mols) > 1:
            a = ax[i]
        if includeAtomNumbers:
            for atom in mol.GetAtoms():
                atom.SetProp('atomLabel', str(atom.GetIdx()))
        mol_img = Draw.MolToImage(mol)
        a.axis('off')
        a.imshow(mol_img)

def _remove_Hs(mol, idx, n):
    cnt = 0
    for n_atom in mol.GetAtoms()[idx].GetNeighbors():
        if n_atom.GetAtomicNum() == 1:
            mol.RemoveAtom(n_atom.GetIdx())
            cnt += 1
            if cnt == n:
                return
    raise RuntimeError(f"Could not remove {n} neighboring H atoms.")

def _count_H(atom):
    return len([a for a in atom.GetNeighbors() if a.GetAtomicNum() == 1])

def _validate_bond_type(min_Hs, bond_type):
    if bond_type is None and min_Hs > 1:
        print(("[WARNING] Merge molecules with single bond where " + 
               "double bond would be possible"))
    bond_type = 'single' if bond_type is None else bond_type
    if min_Hs == 1 and bond_type not in ['single']:
        raise ValueError(f"Invalid bond type '{bond_type}'")
    elif min_Hs == 2 and bond_type not in ['single', 'double']:
        raise ValueError(f"Invalid bond type '{bond_type}'")
    elif min_Hs == 3 and bond_type not in ['single', 'double', 'triple']:
        raise ValueError(f"Invalid bond type '{bond_type}'")
    elif min_Hs == 0 or min_Hs > 3:
        raise ValueError("Invalid number of H atoms.")

    if bond_type == 'single':
        return (1, Chem.rdchem.BondType.SINGLE)
    elif bond_type == 'double':
        return (2, Chem.rdchem.BondType.DOUBLE)
    elif bond_type == 'triple':
        return (3, Chem.rdchem.BondType.TRIPLE)
    else:
        raise ValueError(f"Invalid bond type '{bond_type}'")

def merge(mol1, mol2, atom_idx1, atom_idx2, bond_type=None):
    """ 
    Merge two molecules at the given atom indices. 
    This operation neglacts the byproduct in the form of bond_type * 2H 
    in the returned molecule.

    Args:
        mol1 (rdkit.Chem.rdchem.Mol): First molecule to merge
        mol2 (rdkit.Chem.rdchem.Mol): Second molecule to merge
        atom_idx1 (int): Merge index in first molecule.
        atom_idx2 (int): Merge index in second molecule.
        bond_type (str): Bond type used to merge the molecules.
            If not set a single bond is used. 
            Possible Values: [None, 'single', 'double', 'triple']

    Returns:
        
    """
    mol1 = Chem.AddHs(mol1)
    mol2 = Chem.AddHs(mol2)
    mol1_atoms = mol1.GetAtoms()
    mol2_atoms = mol2.GetAtoms()
    min_Hs = np.min([
        _count_H(mol1_atoms[atom_idx1]), 
        _count_H(mol2_atoms[atom_idx2])
    ])
    bond_nr, bond_type = _validate_bond_type(min_Hs, bond_type)
    idx1, idx2 = (atom_idx1, len(mol1_atoms) + atom_idx2 - bond_nr)
    mol = Chem.RWMol(Chem.CombineMols(mol1, mol2))
    _remove_Hs(mol, idx1, bond_nr)
    _remove_Hs(mol, idx2, bond_nr)
    mol.AddBond(idx1, idx2, order=bond_type) 
    Chem.SanitizeMol(mol)
    return mol 

if __name__ == "__main__":
    missing_smiles_reactant = ['CC1(C)OBOC1(C)C', 'CCC']
    boundary_atoms_list = [{'B': 7}, {'C':1}]
    result = 'CC1(C)OB(-C(C)C)OC1(C)C'
    result_w = 'CC1(C)OBCCCOC1(C)C'

    mol1 = Chem.MolFromSmiles(missing_smiles_reactant[0])
    mol2 = Chem.MolFromSmiles(missing_smiles_reactant[1])
    mol_res = Chem.MolFromSmiles(result)

    m_mol = Chem.RemoveHs(merge(mol1, mol2, 4, 1))
    plot_mols([mol1, mol2, m_mol], includeAtomNumbers=False)
    plt.show()
