from __future__ import annotations
import rdkit.Chem as Chem
import rdkit.Chem.rdchem as rdchem
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
        self.compound = compound
        self.index = index
        self.symbol = symbol
        if self.symbol is None:
            self.symbol = self.compound.mol.GetAtomWithIdx(index).GetSymbol()
        self.neighbor_symbol = neighbor_symbol
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

    def get_atom(self) -> rdchem.Atom:
        mol = self.compound.mol
        return mol.GetAtomWithIdx(self.index)


class Compound:
    def __init__(
        self,
        mol: str | rdchem.Mol,
        src_smiles: str | None = None,
    ):
        if isinstance(mol, str):
            self.mol = rdmolfiles.MolFromSmiles(mol)
        elif isinstance(mol, rdchem.Mol):
            self.mol = mol
        else:
            raise ValueError(
                "Argument 'mol' must be either a valid smiles or an rdkit molecule."
            )
        self.src_smiles = src_smiles
        self.boundaries: list[Boundary] = []
        self.rules = []

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

    def update(self, new_mol: rdchem.Mol, merged_boundary: Boundary):
        self.mol = new_mol
        self.boundaries.remove(merged_boundary)


class CompoundCollection:
    def __init__(self):
        self.compounds: list[Compound] = []


class Reaction:
    def __init__(self, reaction_smiles: str | None = None):
        self.reactant_collection = CompoundCollection()
        self.product_collection = CompoundCollection()
