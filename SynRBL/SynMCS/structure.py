from __future__ import annotations
import rdkit.Chem as Chem
import rdkit.Chem.rdChemReactions as rdChemReactions
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdmolops as rdmolops

from .utils import check_atom_dict


class Boundary:
    def __init__(
        self,
        compound: Compound,
        index: int,
        symbol: str | None = None,
        neighbor_symbol: str | None = None,
    ):
        self.index = index
        self.symbol = symbol
        self.neighbor_symbol = neighbor_symbol
        self.is_merged = False

        self.__compound = compound

    @property
    def compound(self) -> Compound:
        return self.__compound

    def verify(self):
        if self.symbol is None:
            return
        mol = self.compound.mol
        sym = mol.GetAtomWithIdx(self.index).GetSymbol()
        if sym != self.symbol:
            raise ValueError(
                (
                    "Invalid boundary atom symbol. "
                    + "Expected '{}' but found '{}' at index {}."
                ).format(self.symbol, sym, self.index)
            )


class Compound:
    def __init__(
        self,
        smiles: str,
        src_smiles: str | None = None,
    ):
        self.mol = rdmolfiles.MolFromSmiles(smiles)
        self.src_smiles = src_smiles
        self.__boundaries: list[Boundary] = []

    @property
    def boundary_len(self) -> int:
        return len(self.__boundaries)

    def add_boundary(
        self, index, symbol: str | None = None, neighbor_symbol: str | None = None
    ):
        b = Boundary(self, index, symbol, neighbor_symbol)
        b.verify()
        self.__boundaries.append(b)

    def get_boundary(self, i) -> Boundary:
        return self.__boundaries[i]


def merge(boundary1: Boundary, boundary2: Boundary):
    boundary1.is_merged = True
    boundary2.is_merged = True


class CompoundCollection:
    def __init__(self):
        self.compounds: list[Compound] = []

    def merge(self):
        j_b = 0
        for i, comp_i in enumerate(self.compounds):
            for i_b in range(comp_i.boundary_len):
                bound_i = comp_i.get_boundary(i_b)
                for _, comp_j in enumerate(self.compounds[i + 1 :], start=i + 1):
                    for j_b in range(comp_j.boundary_len):
                        bound_j = comp_j.get_boundary(j_b)
                        assert (
                            bound_i is not bound_j
                        ), "Bounds should never be the same."
                        merge(bound_i, bound_j)
                if not bound_i.is_merged:
                    raise ValueError("Merge failed. Missing compound.")


    def get_merge_list(self):
        i_c = 0
        j_c = 0
        j_b = 0
        merge_list = []
        while i_c < len(self.compounds):
            comp_i = self.compounds[i_c]
            for i_b in range(len(comp_i.boundaries)):
                j_c = i_c + 1
                if len(self.compounds) <= j_c:
                    raise ValueError(
                        "Failed to merge compound collection. "
                        + "Missing second compound."
                    )
                comp_j = self.compounds[j_c]
                merge_list.append((i_c, i_b, j_c, j_b))
                j_b += 1
                if j_b >= len(comp_j.boundaries):
                    j_b = 0
                    j_c += 1
            i_c = j_c + 1
        return merge_list


class Reaction:
    def __init__(self, reaction_smiles: str | None = None):
        self.reactant_collection = CompoundCollection()
        self.product_collection = CompoundCollection()
