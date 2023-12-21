from __future__ import annotations
import rdkit.Chem as Chem
import rdkit.Chem.rdChemReactions as rdChemReactions
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdmolops as rdmolops


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
        self.compound = compound
        self.is_merged = False

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
        self.boundaries: list[Boundary] = []

    @property
    def smiles(self) -> str:
        return rdmolfiles.MolToSmiles(self.mol) 

    def add_boundary(
        self, index, symbol: str | None = None, neighbor_symbol: str | None = None
    ) -> Boundary:
        b = Boundary(self, index, symbol, neighbor_symbol)
        b.verify()
        self.boundaries.append(b)
        return b

class CompoundCollection:
    def __init__(self):
        self.compounds: list[Compound] = []


class Reaction:
    def __init__(self, reaction_smiles: str | None = None):
        self.reactant_collection = CompoundCollection()
        self.product_collection = CompoundCollection()
