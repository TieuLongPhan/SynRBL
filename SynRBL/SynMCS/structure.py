import rdkit.Chem.rdChemReactions as rdChemReactions
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdmolops as rdmolops

from .utils import check_atom_dict


class Reaction:
    """
    Container class for balancing a reaction.

    Attributes:
        reaction: (rdkit.Chem.rdChem.Reactions.ChemicalReaction): The rdkit
            reaction object.
    """

    def __init__(self, reaction_smiles):
        """
        Container class for balancing a reaction.

        Arguments:
            reaction_smiles (str): The reaction smiles string of the unbalanced
                reaction.
        """
        self.reaction = rdChemReactions.ReactionFromSmarts(
            reaction_smiles, useSmiles=True
        )

    @property
    def smiles(self) -> str:
        """
        Returns the smiles representation of the reaction.
        """
        return rdChemReactions.ReactionToSmiles(self.reaction)


class ReactionCompound:
    """
    Class for managing a compound in a reaction. This includes the transition
    from reactant to product.

    Attributes:
        reactant (str): The source structure of the compound.
        product (str): The structure on the product side.
        is_new_reactant (bool): True if the reactant is already part of the
            reaction or False if the compound should be added.
        is_new_product (bool): True if the product is already part of the
            reaction or False if the compound should be added.
        boundaries (list[(rdkit.Chem.rdchem.Atom, rdkit.Chem.rdchem.Atom | None)]):
            A list of atom tuples for the boundaries, where the first atom is
            the boundary atom in the product compound and the second atom is
            the neighboring atom in the source structure if available.
    """

    def __init__(self, reactant, product, is_new_reactant, is_new_product):
        self.reactant = rdmolfiles.MolFromSmiles(reactant)
        self.product = rdmolfiles.MolFromSmiles(product)
        self.is_new_reactant = is_new_reactant
        self.is_new_product = is_new_product
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
        check_atom_dict(self.product, boundary_atom)
        b_idx = list(boundary_atom.values())[0]
        b_atom = self.product.GetAtomWithIdx(b_idx)
        if dock_atom is not None:
            check_atom_dict(self.reactant, dock_atom)
            d_idx = list(dock_atom.values())[0]
            d_atom = self.reactant.GetAtomWithIdx(d_idx)
        else:
            d_atom = None
        self.boundaries.append((b_atom, d_atom))

    def complete_reaction(self, reactants, products):
        if self.is_new_reactant:
            reactants = rdmolops.CombineMols(reactants, self.reactant)
        if self.is_new_product:
            products = rdmolops.CombineMols(products, self.product)
        return reactants, products
