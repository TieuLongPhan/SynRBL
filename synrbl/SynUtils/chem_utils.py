import re
import numpy as np

import rdkit.Chem as Chem
import rdkit.DataStructs as DataStructs
import rdkit.Chem.rdFingerprintGenerator as rdFingerprintGenerator
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.rdmolfiles as rdmolfiles
from collections import Counter
from typing import List, Dict, Tuple, Optional, Union
from fgutils import FGQuery


class CheckCarbonBalance:
    def __init__(
        self, reactions_data: List[Dict[str, str]], rsmi_col="reactions", symbol=">>"
    ):
        """
        Initialize the CheckCarbonBalance class with reaction data.

        Parameters:
        reactions_data (List[Dict[str, str]]): A list of dictionaries, each
            containing reaction information.
        """
        self.reactions_data = reactions_data
        self.rsmi_col = rsmi_col
        self.symbol = symbol

    @staticmethod
    def count_carbon_atoms(smiles: str) -> int:
        """
        Count the number of carbon atoms in a molecule represented by a SMILES
        string.

        Parameters:
        smiles (str): A SMILES string.

        Returns:
        int: The number of carbon atoms in the molecule. Returns 0 if the
            SMILES string is invalid.
        """
        mol = Chem.MolFromSmiles(smiles)
        return (
            sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == "C") if mol else 0
        )

    def check_carbon_balance(self) -> None:
        """
        Check and update the carbon balance status for each reaction in the
        reactions data.

        The method updates each reaction dictionary in the reactions data with
        a new key 'carbon_balance_check'. This key will have the value
        'products' if the number of carbon atoms in the products is greater
        than or equal to the reactants, and 'reactants' otherwise.
        """
        for reaction in self.reactions_data:
            try:
                reactants_smiles, products_smiles = reaction[self.rsmi_col].split(
                    self.symbol
                )
                reactants_carbon = sum(
                    self.count_carbon_atoms(smiles)
                    for smiles in reactants_smiles.split(".")
                )
                products_carbon = sum(
                    self.count_carbon_atoms(smiles)
                    for smiles in products_smiles.split(".")
                )

                if reactants_carbon >= products_carbon:
                    reaction["carbon_balance_check"] = "products"
                else:
                    reaction["carbon_balance_check"] = "reactants"
            except KeyError as e:
                print(f"Key error: {e}")
            except ValueError as e:
                print(f"Value error: {e}")

    def is_carbon_balance(self) -> None:
        """
        Check and update the carbon balance status for each reaction in the
        reactions data.

        The method updates each reaction dictionary in the reactions data with
        a new key 'carbon_balance_check'. This key will have the value
        'products' if the number of carbon atoms in the products is greater
        than or equal to the reactants, and 'reactants' otherwise.
        """
        for reaction in self.reactions_data:
            try:
                reactants_smiles, products_smiles = reaction[self.rsmi_col].split(
                    self.symbol
                )
                reactants_carbon = sum(
                    self.count_carbon_atoms(smiles)
                    for smiles in reactants_smiles.split(".")
                )
                products_carbon = sum(
                    self.count_carbon_atoms(smiles)
                    for smiles in products_smiles.split(".")
                )
                reaction["is_carbon_balance"] = reactants_carbon == products_carbon

            except KeyError as e:
                print(f"Key error: {e}")
            except ValueError as e:
                print(f"Value error: {e}")


def calculate_net_charge(sublist: list[dict[str, Union[str, int]]]) -> int:
    """
    Calculate the net charge from a list of molecules represented as SMILES
    strings.

    Args:
        sublist: A list of dictionaries, each with a 'smiles' string and a
            'Ratio' integer.

    Returns:
        The net charge of the sublist as an integer.
    """
    total_charge = 0
    for item in sublist:
        if "smiles" in item and "Ratio" in item:
            mol = Chem.MolFromSmiles(item["smiles"])
            if mol:
                charge = (
                    sum(abs(atom.GetFormalCharge()) for atom in mol.GetAtoms())
                    * item["Ratio"]
                )
                total_charge += charge
    return total_charge


def canon_smiles(smiles: str):
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    try:
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return Chem.MolToSmiles(mol)


def remove_atom_mapping(smiles: str) -> str:
    pattern = re.compile(r":\d+")
    smiles = pattern.sub("", smiles)
    pattern = re.compile(r"\[(?P<atom>(B|C|N|O|P|S|F|Cl|Br|I){1,2})(?:H\d?)?\]")
    smiles = pattern.sub(r"\g<atom>", smiles)
    return smiles


def remove_stereo_chemistry(smiles: str) -> str:
    pattern = re.compile(r"\[(?P<atom>(\w+))@+\w+\]")
    smiles = pattern.sub(r"[\g<atom>]", smiles)
    return smiles


def count_atoms(smiles: str) -> int:
    pattern = re.compile(r"(B|C|N|O|P|S|F|Cl|Br|I|c|n|o)")
    return len(pattern.findall(smiles))


def normalize_smiles(smiles: str) -> str:
    smiles = remove_stereo_chemistry(smiles)
    if ">>" in smiles:
        return ">>".join([normalize_smiles(t) for t in smiles.split(">>")])
    elif "." in smiles:
        token = smiles.split(".")
        token = [normalize_smiles(t) for t in token]
        token.sort(
            key=lambda x: (count_atoms(x), sum(ord(c) for c in x)),
            reverse=True,
        )
        return ".".join(token)
    else:
        smiles = remove_atom_mapping(smiles)
        return canon_smiles(smiles)


def _get_diff_mol(smiles1, smiles2):
    smiles1 = normalize_smiles(smiles1)
    smiles2 = normalize_smiles(smiles2)
    diff_1 = []
    diff_2 = []
    for s1, s2 in zip(smiles1.split("."), smiles2.split(".")):
        if s1 != s2:
            diff_1.append(s1)
            diff_2.append(s2)
    mol_1, mol_2 = Chem.RWMol(), Chem.RWMol()
    if len(diff_1) > 0:
        s_1 = ".".join(diff_1)
        mol_1 = rdmolfiles.MolFromSmiles(s_1)
    if len(diff_2) > 0:
        s_2 = ".".join(diff_2)
        mol_2 = rdmolfiles.MolFromSmiles(s_2)
    return mol_1, mol_2


def wc_similarity(expected_smiles, result_smiles, method="pathway"):
    """
    Compute the worst case similarity between two SMILES. The method compares
    the molecules that are only present on one sie of the reaction to not bias
    the similarity by the size of the compounds. The similarity is the lower
    value from educt and product similarity.

    Args:
        expected_smiles (str): The expected reaction SMILES.
        result_smiles (str): The actual reaction SMILES.
        method (str): The method used to compute similarity.
            Possible values are: ['pathway', 'ecfp', 'ecfp_inv']
    Returns:
        float: The similarity value between 0 and 1
    """

    def _fp(mol1, mol2):
        sim = 0
        if method == "pathway":
            fpgen = AllChem.GetRDKitFPGenerator(maxPath=5, minPath=1)
            fp1 = fpgen.GetFingerprint(mol1)
            fp2 = fpgen.GetFingerprint(mol2)
            sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        elif method == "ecfp":
            fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2)
            fp1 = fpgen.GetSparseCountFingerprint(mol1)
            fp2 = fpgen.GetSparseCountFingerprint(mol2)
            sim = DataStructs.DiceSimilarity(fp1, fp2)
        elif method == "ecfp_inv":
            invgen = rdFingerprintGenerator.GetMorganFeatureAtomInvGen()
            ffpgen = rdFingerprintGenerator.GetMorganGenerator(
                radius=2, atomInvariantsGenerator=invgen
            )
            fp1 = ffpgen.GetSparseCountFingerprint(mol1)
            fp2 = ffpgen.GetSparseCountFingerprint(mol2)
            sim = DataStructs.DiceSimilarity(fp1, fp2)
        else:
            raise ValueError("'{}' is not a valid similarity method.".format(method))
        return sim

    exp = normalize_smiles(expected_smiles)
    res = normalize_smiles(result_smiles)
    if exp == res:
        return 1
    exp_e, exp_p = exp.split(">>")
    res_e, res_p = res.split(">>")
    exp_ed, res_ed = _get_diff_mol(exp_e, res_e)
    exp_pd, res_pd = _get_diff_mol(exp_p, res_p)
    return np.min([_fp(exp_ed, res_ed), _fp(exp_pd, res_pd)])


def check_for_isolated_atom(smiles: str, atom: Optional[str] = "H") -> bool:
    """
    Checks if a specified type of isolated atom (hydrogen by default, or oxygen)
    exists in a SMILES string.
    """
    # Pattern to find isolated atoms; not connected to any other atoms
    pattern = rf"\[{atom}\](?![^[]*\])"
    return bool(re.search(pattern, smiles))


def count_radical_atoms(smiles: str, atomic_num: int) -> int:
    """
    Counts isolated radical atoms in SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    radical_count = 0

    # Iterate over all atoms in the molecule
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == atomic_num and atom.GetNumRadicalElectrons() > 0:
            # Further check if the atom is isolated (has no neighbors)
            if len(atom.GetNeighbors()) == 0:
                radical_count += 1

    return radical_count


def list_difference(
    list1: List[str], list2: List[str]
) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Compares two lists and returns dictionaries that count unique occurrences
    Parameters:
    list1 (List[str]): First list of items for comparison.
    list2 (List[str]): Second list of items for comparison.

    Returns:
    Tuple[Dict[str, int], Dict[str, int]]: A tuple of two dictionaries:
        - First dictionary: Items unique to the first list with their counts.
        - Second dictionary: Items unique to the second list with their counts.
    """
    count1 = Counter(list1)
    count2 = Counter(list2)
    unique_to_list1 = {}
    unique_to_list2 = {}

    for key, count in count1.items():
        if key not in count2:
            unique_to_list1[key] = count
        elif count > count2[key]:
            unique_to_list1[key] = count - count2[key]

    for key, count in count2.items():
        if key not in count1:
            unique_to_list2[key] = count
        elif count > count1[key]:
            unique_to_list2[key] = count - count1[key]

    return unique_to_list1, unique_to_list2


def find_functional_reactivity(reaction_smiles: str) -> Tuple[List[str], List[str]]:
    """
    Analyzes functional groups

    Parameters:
    reaction_smiles (str): SMILES string of the reaction
    Returns:
    Tuple[List[str], List[str]]: Two lists containing unique functional groups
    in reactants and products, respectively.
    """
    query = FGQuery()
    reactant, product = reaction_smiles.split(">>")
    fg_reactant = query.get(reactant)
    fg_product = query.get(product)
    fg_reactant = [value[0] for value in fg_reactant]
    fg_product = [value[0] for value in fg_product]
    reactant_fg, product_fg = list_difference(fg_reactant, fg_product)
    return list(reactant_fg.keys()), list(product_fg.keys())
