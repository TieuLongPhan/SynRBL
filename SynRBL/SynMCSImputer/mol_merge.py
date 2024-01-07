import rdkit.Chem
import rdkit.Chem.Draw
import rdkit.Chem.rdmolops as rdmolops
import rdkit.Chem.rdmolfiles as rdmolfiles
import matplotlib.pyplot as plt
from SynRBL.SynMCS.rule_formation import Property
from SynRBL.SynMCS.merge_rule import *


class SubstructureError(Exception):
    """
    Exception if the structure of bound configuration to be merged is invalid.
    """

    def __init__(self):
        """
        Exception if the structure of bound configuration to be merged is
        invalid.
        """
        super().__init__("Substructure mismatch.")


class InvalidAtomDict(Exception):
    """
    Exception if the atom dictionary is invalid.
    """

    def __init__(self, expected, actual, index, smiles):
        """
        Exception if the atom dictionary is invalid.

        Arguments:
            expected (str): The expected atom at the given index.
            actual (str): The actual atom at the index.
            index (int): The atom index in the molecule.
            smiles (str): The SMILES representation of the molecule.
        """
        super().__init__(
            (
                "Atom dict is invalid for molecule '{}'. "
                + "Expected atom '{}' at index {} but found '{}'."
            ).format(smiles, expected, index, actual)
        )


class AtomTracker:
    """
    A class to track atoms through the merge process.
    """

    def __init__(self, indices):
        """
        A class to track atoms through the merge process. After instantiation
        call the add_atoms method to initialize the tracker with the atom
        objects.

        Arguments:
            indices (list[int]): A list of atom indices to track.
        """
        self.__track_dict = {}
        if indices is not None:
            for idx in indices:
                self.__track_dict[str(idx)] = {}

    def add_atoms(self, mol, offset=0):
        """
        Add atom objects to the tracker. This is a necessary initialization
        step.

        Arguments:
            mol (rdkit.Chem.Mol): The molecule in which to track atoms.
            offset (int, optional): The atom index offset.
        """
        atoms = mol.GetAtoms()
        for k in self.__track_dict.keys():
            self.__track_dict[k]["atom"] = atoms[int(k) + offset]

    def to_dict(self):
        """
        Convert the tracker into a mapping dictionary.

        Returns:
            dict: A dictionary where keys are the old indices and the values
                represent the atom indices in the new molecule.
        """
        sealed_dict = {}
        for k in self.__track_dict.keys():
            sealed_dict[k] = self.__track_dict[k]["atom"].GetIdx()
        return sealed_dict




def plot_mols(mols, includeAtomNumbers=False, titles=None, figsize=None):
    if type(mols) is not list:
        mols = [mols]
    if len(mols) == 0:
        return
    _, ax = plt.subplots(1, len(mols), figsize=figsize)
    for i, mol in enumerate(mols):
        a = ax
        if len(mols) > 1:
            a = ax[i]
        if includeAtomNumbers:
            for atom in mol.GetAtoms():
                atom.SetProp("atomLabel", str(atom.GetIdx()))
        mol_img = rdkit.Chem.Draw.MolToImage(mol)
        if titles is not None and i < len(titles):
            a.set_title(titles[i])
        a.axis("off")
        a.imshow(mol_img)




def merge_mols(mol1, mol2, idx1, idx2, mol1_track=None, mol2_track=None):
    """
    Merge two molecules. How and if the molecules are merge is defined by
    merge rules. For more details on merge rules see the
    SynRBL.SynMCS.mol_merge.MergeRule class documentation and the rule
    configuration in SynRBL/SynMCS/merge_rules.json.

    Arguments:
        mol1 (rdkit.Chem.Mol): First molecule to merge.
        mol2 (rdkit.Chem.Mol): Second molecule to merge.
        idx1 (int): Atom index in mol1 where the new bond is formed.
        idx2 (int): Atom index in mol2 where the new bond is formed.
        mol1_track (list[int], optional): A list of atom indices in mol1 that
            should be tracked during merging. The index mapping is part of the
            result with key 'aam1'.
        mol2_track (list[int], optional): A list of atom indices in mol2 that
            should be tracked during merging. The index mapping is part of the
            result with key 'aam2'.

    Returns:
        dict: A dictionary with the merged molecule at key 'mol' and optional
            atom index mappings at 'aam1' and 'aam2' as well as the applied
            merge rule at 'rule'.
    """
    mol1_tracker = AtomTracker(mol1_track)
    mol2_tracker = AtomTracker(mol2_track)

    mol1 = rdmolops.AddHs(mol1)
    mol2 = rdmolops.AddHs(mol2)
    mol = rdmolops.RWMol(rdmolops.CombineMols(mol1, mol2))
    mol2_offset = len(mol1.GetAtoms())
    mol1_tracker.add_atoms(mol)
    mol2_tracker.add_atoms(mol, offset=mol2_offset)
    atom1 = mol.GetAtoms()[idx1]
    atom2 = mol.GetAtoms()[mol2_offset + idx2]
    merge_rule = None
    for rule in get_merge_rules():
        if not rule.can_apply(atom1, atom2):
            continue
        rule.apply(mol, atom1, atom2)
        merge_rule = rule
        break
    if not merge_rule:
        raise NoMergeRuleError(atom1, atom2, mol1, mol2)
    rdmolops.SanitizeMol(mol)
    return {
        "mol": mol,
        "merge_rules": [merge_rule],
        "aam1": mol1_tracker.to_dict(),
        "aam2": mol2_tracker.to_dict(),
    }


def merge_expand(mol, bound_indices, neighbors=None):
    """
    Expand and merge a single molecule with the product of a suitable compound
    rule. For more informatino on compound rules see the
    SynRBL.SynMCS.mol_merge.CompoundRule class documentation and the compound
    rule config in SynRBL/SynMCS/compound_rules.json.

    Arguments:
        mol (rdkit.Chem.Mol): The molecule to expand.
        bound_indices (list[int]): A list of boundary atom indices.
        neighbors (list[str]): The neighboring atom for each boundary atom.

    Returns:
        dict: A dictionary containing the expanded molecule at key 'mol', the
        used compound rules at 'compound_rules' and the used merge rules at
        'merge_rules'.
    """
    if not isinstance(bound_indices, list):
        raise ValueError("bound_indices must be of type list")
    bound_len = len(bound_indices)
    if neighbors is None:
        neighbors = [None for _ in range(bound_len)]
    if len(neighbors) != bound_len:
        raise ValueError(
            "neighbors list must be of same length as bound_indices. "
            + "(bound_indices={}, neighbors={})".format(bound_indices, neighbors)
        )

    merged_mol = mol
    used_compound_rules = []
    used_merge_rules = []
    for i in range(bound_len):
        atom = merged_mol.GetAtoms()[bound_indices[i]]
        comp, rule = get_compound(atom, neighbors[i])
        used_compound_rules.append(rule)
        if comp is not None:
            merge_result = merge_mols(
                merged_mol,
                comp["mol"],
                bound_indices[i],
                comp["index"],
                mol1_track=bound_indices,
            )
            bound_indices = [merge_result["aam1"][str(idx)] for idx in bound_indices]
            merged_mol = merge_result["mol"]
            used_merge_rules.extend(merge_result["merge_rules"])
    return {
        "mol": merged_mol,
        "compound_rules": used_compound_rules,
        "merge_rules": used_merge_rules,
    }


def _check_atoms(mol, atom_dict):
    """
    Check if the atom dict matches the actual molecule. If the atom dictionary
    is not valid a InvalidAtomDict exception is raised.

    Arguments:
        mol (rdkit.Chem.Mol): The molecule on which the atom dictionary is
            checked.
        atom_dict (dict, list[dict]): The atom dictionary or a list of atom
            dictionaries to check on the molecule.
    """
    if isinstance(atom_dict, list):
        for e in atom_dict:
            _check_atoms(mol, e)
    elif isinstance(atom_dict, dict):
        sym, idx = next(iter(atom_dict.items()))
        actual_sym = mol.GetAtomWithIdx(idx).GetSymbol()
        if actual_sym != sym:
            raise InvalidAtomDict(sym, actual_sym, idx, rdmolops.MolToSmiles(mol))
    else:
        raise ValueError("atom_dict must be either a list or a dict.")


def _ad2t(atom_dict):
    """
    Convert atom dict to symbol and index tuple.

    Arguments:
        atom_dict (dict): Atom dictionary in the for of {<symbol>: <index>}

    Returns:
        (str, int): Atom dictionary as tuple (<symbol>, <index>}
    """
    if not isinstance(atom_dict, dict) or len(atom_dict) != 1:
        raise ValueError("atom_dict must be of type {<symbol>: <index>}")
    return next(iter(atom_dict.items()))


def _adl2t(atom_dict_list):
    """
    Split atom dict into symbol and indices lists.

    Arguments:
        atom_dict_list (list[dict]): The atom dictionary list.

    Returns:
        (list[str], list[ind]): Returns the symbol and indices lists.
    """
    sym_list = []
    idx_list = []
    for a in atom_dict_list:
        sym, idx = _ad2t(a)
        sym_list.append(sym)
        idx_list.append(idx)
    return sym_list, idx_list


def _split_mol(mol, bounds, neighbors):
    """
    Split not connected compounds in the molecule object into individual
    fragments and correct the bounds and neighbors lists accordingly.

    Arguments:
        mol (rdkit.Chem.Mol): The molecule to check for splits.
        bounds (list[dict]): Atom dict list of boundary atoms.
        neighbors (list[dict]): Atom dict list of boundary neighboring atoms.

    Returns:
        (list[rdkit.Chem.Mol], list[list[dict]], list[list[dict]]): Returns
            a list of compouns and the adjusted bounds and neighbors atom
            dict lists.
    """
    if not (
        isinstance(bounds, list) and len(bounds) > 0 and isinstance(bounds[0], dict)
    ):
        raise ValueError("bounds must be a list of atom dicts.")
    if not (
        isinstance(neighbors, list)
        and len(neighbors) == len(bounds)
        and isinstance(neighbors[0], dict)
    ):
        raise ValueError(
            "neighbors must be a list of atom dicts with the "
            + "same length as bounds."
        )

    frags = list(rdmolops.GetMolFrags(mol, asMols=True))
    offsets = [0]
    for i, f in enumerate(frags):
        offsets.append(offsets[i] + len(f.GetAtoms()))
    _bounds = [[] for _ in range(len(frags))]
    _neighbors = [[] for _ in range(len(frags))]
    for b, n in zip(bounds, neighbors):
        sym, idx = _ad2t(b)
        for i in range(len(offsets) - 1):
            if idx >= offsets[i] and idx < offsets[i + 1]:
                _bounds[i].append({sym: idx - offsets[i]})
                _neighbors[i].append(n)
    return frags, _bounds, _neighbors


def merge(mols, bounds, neighbors):
    """
    Merge molecules. This works for either a single molecule which is extended
    by a compound rule or for two molecules where a suitable merge rule exists.
    For additional information on compound and merge rules see the MergeRule
    and CompoundRule class documentation in module SynRBL.SynMCS.mol_merge and
    the rule configuration in merge_rules.json and compound_rules.json in
    SynRBL/SynMCS/.

    Arguments:
        mols (list[rdkit.Chem.Mol]): A list of molecules. Merging is only
            supported for individual expansions and molecule pairs.
        bounds (list[list[dict]]): A list of boundary atom dictionaries for
            each molecule.
        neighbors (list[list[dict]]): A list of neighboring atom dictionaries
            for each molecule.

    Returns:
        dict: Returns a dictionary with the merge molecule at key 'mol', the
            list of used merge rules at 'merge_rules', and the list of used
            compound rules at 'compound_rules'.
    """
    merged_mols = []
    if len(mols) == 1:
        mols, bounds, neighbors = _split_mol(mols[0], bounds[0], neighbors[0])
        for m, b, n in zip(mols, bounds, neighbors):
            _check_atoms(m, b)
            _, indices = _adl2t(b)
            nsyms, _ = _adl2t(n)
            merged_mol = merge_expand(m, indices, nsyms)
            merged_mols.append(merged_mol)
    elif len(mols) == 2:
        mol1, mol2 = mols[0], mols[1]
        _check_atoms(mol1, bounds[0])
        _check_atoms(mol2, bounds[1])
        if len(bounds[0]) != 1 or len(bounds[1]) != 1:
            raise SubstructureError()
        _, idx1 = _ad2t(bounds[0][0])
        _, idx2 = _ad2t(bounds[1][0])
        merged_mol = merge_mols(mol1, mol2, idx1, idx2)
        merged_mols.append(
            {
                "mol": merged_mol["mol"],
                "merge_rules": merged_mol["merge_rules"],
                "compound_rules": [],
            }
        )
    elif len(mols) > 2:
        raise NotImplementedError(
            "Merging of {} molecules is not supported.".format(len(mols))
        )
    return merged_mols


class CompletionCompound:
    def __init__(self, reactant, product, new_reactant, new_product):
        self.reactant = rdmolfiles.MolFromSmiles(reactant)
        self.product = rdmolfiles.MolFromSmiles(product)
        self.new_reactant = new_reactant
        self.new_product = new_product
        self.boundaries = []

    def add_broken_bond(self, boundary_atom: dict, dock_atom: dict | None = None):
        """
        Add a missing bond to the completion compound. A missing bond is the
        bond how the compound was connected in the source molecule. This is
        most likly the bond that was broken in the reaction.

        Arguments:
            boundary_atom (dict): The atom that was connected to the MCS.
                Dictionary in the form: {'<symbol>': index}.
            dock_atom (dict): The atom in the MCS that was connected to the
                boundary_atom. Dictionary in the form: {'<symbol>': index}.
        """
        _check_atoms(self.product, boundary_atom)
        b_idx = list(boundary_atom.values())[0]
        b_atom = self.product.GetAtomWithIdx(b_idx)
        if dock_atom is not None:
            _check_atoms(self.reactant, dock_atom)
            d_idx = list(dock_atom.values())[0]
            d_atom = self.reactant.GetAtomWithIdx(d_idx)
        else:
            d_atom = None
        self.boundaries.append((b_atom, d_atom))

    def complete_reaction(self, reactants, products):
        if self.new_reactant:
            reactants = rdmolops.CombineMols(reactants, self.reactant)
        if self.new_product:
            products = rdmolops.CombineMols(products, self.product)
        return reactants, products


import rdkit.Chem.rdmolfiles as rdmolfiles

reactants = "COC(=O)[C@H](CCCCNC(=O)OCc1ccccc1)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O"
products = "COC(=O)[C@H](CCCCN)NC(=O)Nc1cc(OC)cc(C(C)(C)C)c1O"
mol3 = rdmolfiles.MolFromSmiles(reactants)
mol4 = rdmolfiles.MolFromSmiles(products)
mol5 = rdmolfiles.MolFromSmiles("O=COCc1ccccc1")
# plot_mols([mol3, mol4, mol5], includeAtomNumbers=True)
cc1 = CompletionCompound(reactants, "O=COCc1ccccc1", False, True)
cc1.add_broken_bond({"C": 1}, {"N": 9})
cc2 = CompletionCompound("O", "O", True, True)
cc2.add_broken_bond({"O": 0}, None)
mol6, mol7 = cc1.complete_reaction(mol3, mol4)
mol8, mol9 = cc2.complete_reaction(mol6, mol7)
plot_mols([mol6, mol7])
plot_mols([mol8, mol9])

ccs = [cc1, cc2]
for i, ci in enumerate(ccs):
    for bi in ci.boundaries:
        for cj in ccs[i + 1 :]:
            if ci == cj:
                assert False
            for bj in cj.boundaries:
                print(bi[0].GetSymbol(), bj[0].GetSymbol())

#r = Reaction('CCO>>C(=O)C')
#cc = CompletionCompound.from_missing_product_substructure('', '')
#r.add_completion_compound(cc)
#r.impute_missing()
#r.merge()

# r.merge():
#   reactant_collection.merge()
#   procut_collection.merge()

# compound_collection.merge():
#   for i, comp_i in enumerate(compounds):
# cc.add_missing_bond({})
