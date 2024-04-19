import re
import numpy as np

import rdkit.Chem as Chem
import rdkit.DataStructs as DataStructs
import rdkit.Chem.rdFingerprintGenerator as rdFingerprintGenerator
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.rdmolfiles as rdmolfiles


from typing import List, Dict
from typing import Union


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


def remove_atom_mapping(smiles: str) -> str:
    pattern = re.compile(r":\d+")
    smiles = pattern.sub("", smiles)
    pattern = re.compile(r"\[(?P<atom>(B|C|N|O|P|S|F|Cl|Br|I){1,2})(?:H\d?)?\]")
    smiles = pattern.sub(r"\g<atom>", smiles)
    return smiles


def normalize_smiles(smiles: str) -> str:
    smiles = smiles.replace("@", "")
    if ">>" in smiles:
        return ">>".join([normalize_smiles(t) for t in smiles.split(">>")])
    elif "." in smiles:
        token = sorted(
            smiles.split("."),
            key=lambda x: (sum(1 for c in x if c.isupper()), sum(ord(c) for c in x)),
            reverse=True,
        )
        token = [normalize_smiles(t) for t in token]
        token.sort(
            key=lambda x: (sum(1 for c in x if c.isupper()), sum(ord(c) for c in x)),
            reverse=True,
        )
        return ".".join(token)
    else:
        return Chem.CanonSmiles(remove_atom_mapping(smiles))


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
