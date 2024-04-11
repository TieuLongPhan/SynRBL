from __future__ import annotations

import rdkit.Chem.rdchem as rdchem
import rdkit.Chem.rdmolfiles as rdmolfiles


class Boundary:
    def __init__(
        self,
        compound: Compound,
        index: int,
        symbol: str | None = None,
        neighbor_index: int | None = None,
        neighbor_symbol: str | None = None,
    ):
        self.compound = compound
        self.index = index
        self.neighbor_index = neighbor_index
        self.symbol = symbol
        if self.symbol is None:
            self.symbol = self.compound.mol.GetAtomWithIdx(index).GetSymbol()
        self.neighbor_symbol = neighbor_symbol
        if self.neighbor_index is not None and self.neighbor_symbol is None:
            src_mol = self.promise_src()
            self.neighbor_symbol = src_mol.GetAtomWithIdx(neighbor_index).GetSymbol()
        self.is_merged = False

    def __str__(self) -> str:
        return "Boundary '{}' @ {} in '{}' from '{}'.".format(
            self.symbol, self.index, self.compound.smiles, self.compound.src_smiles
        )

    def promise_src(self) -> rdchem.Mol:
        mol = self.compound.src_mol
        if mol is None:
            raise ValueError("Missing src_mol in compound.")
        return mol

    def promise_neighbor_index(self) -> int:
        if self.neighbor_index is None:
            raise ValueError("Missing neighbor index.")
        return self.neighbor_index

    def verify(self):
        if self.symbol is not None:
            mol = self.compound.mol
            sym = mol.GetAtomWithIdx(self.index).GetSymbol()
            if sym != self.symbol:
                raise ValueError(
                    (
                        "Invalid boundary atom symbol. "
                        + "Expected '{}' but found '{}' at index {}."
                    ).format(self.symbol, sym, self.index)
                )
        if self.neighbor_index is not None:
            src_mol = self.promise_src()
            sym = src_mol.GetAtomWithIdx(self.neighbor_index).GetSymbol()
            if sym != self.neighbor_symbol:
                raise ValueError(
                    (
                        "Invalid neighboring atom symbol. "
                        + "Expected '{}' but found '{}' at index {}."
                    ).format(self.neighbor_symbol, sym, self.neighbor_index)
                )

    def get_atom(self) -> rdchem.Atom:
        mol = self.compound.mol
        return mol.GetAtomWithIdx(self.index)

    def get_neighbor_atom(self) -> rdchem.Atom | None:
        mol = self.compound.src_mol
        if mol is None:
            return None
        return mol.GetAtomWithIdx(self.neighbor_index)


def _to_mol(value: str | rdchem.Mol | None) -> rdchem.Mol | None:
    if isinstance(value, str):
        return rdmolfiles.MolFromSmiles(value)
    elif isinstance(value, rdchem.Mol):
        return value
    elif value is None:
        return None
    else:
        raise TypeError("Value must be str or rdchem.Mol.")


class Compound:
    def __init__(
        self,
        mol: str | rdchem.Mol,
        src_mol: str | rdchem.Mol | None = None,
        compound_set: CompoundSet | None = None,
    ):
        self.mol: rdchem.Mol = _to_mol(mol)
        if self.mol is None:
            raise ValueError(
                (
                    "Argument 'mol' must be either a valid smiles or an "
                    + "rdkit molecule. (value='{}')"
                ).format(mol)
            )
        self.src_mol = _to_mol(src_mol)
        self.boundaries: list[Boundary] = []
        self.rules = []
        self.active = True
        self.__compound_set = compound_set

    def __str__(self):
        return "Compound '{}' | Boundaries: {} ".format(
            self.smiles, [(b.symbol, b.index) for b in self.boundaries]
        )

    @property
    def smiles(self) -> str:
        return rdmolfiles.MolToSmiles(self.mol)

    @property
    def num_compounds(self) -> int:
        return len(self.smiles.split("."))

    @property
    def src_smiles(self) -> str | None:
        if self.src_mol is None:
            return None
        return rdmolfiles.MolToSmiles(self.src_mol)

    @property
    def is_catalyst(self) -> bool:
        return self.smiles == self.src_smiles and len(self.boundaries) == 0

    @property
    def compound_set(self) -> CompoundSet:
        if self.__compound_set is None:
            raise RuntimeError("This compound does not belong to a set.")
        return self.__compound_set

    def add_boundary(
        self,
        index,
        symbol: str | None = None,
        neighbor_index: int | None = None,
        neighbor_symbol: str | None = None,
    ) -> Boundary:
        b = Boundary(
            self,
            index,
            symbol=symbol,
            neighbor_index=neighbor_index,
            neighbor_symbol=neighbor_symbol,
        )
        b.verify()
        self.boundaries.append(b)
        return b

    def update(self, new_mol: rdchem.Mol, merged_boundary: Boundary):
        self.mol = new_mol
        self.boundaries.remove(merged_boundary)

    def concat(self, compound: Compound):
        if self.compound_set != compound.compound_set:
            raise ValueError("Compounds are not from the same set.")
        if len(self.boundaries) > 0 or len(self.boundaries) > 0:
            raise ValueError(
                "Can not concat compounds with open boundaries. Try merging them."
            )
        src_mol = None
        try:
            src_mol = _to_mol("{}.{}".format(self.src_smiles, compound.src_smiles))
        except Exception:
            pass
        self.mol = _to_mol("{}.{}".format(self.smiles, compound.smiles))
        self.src_mol = src_mol
        self.rules.extend(compound.rules)
        self.compound_set.remove_compound(compound)

    def reset_compound_set(self):
        self.__compound_set = None


class CompoundSet:
    def __init__(self):
        self.__compounds: list[Compound] = []

    def __len__(self):
        return len(self.__compounds)

    @property
    def compounds(self) -> list[Compound]:
        return self.__compounds

    @property
    def boundaries(self) -> list[Boundary]:
        boundaries = []
        for c in self.compounds:
            boundaries.extend(c.boundaries)
        return boundaries

    def add_compound(
        self, mol: str | rdchem.Mol, src_mol: str | rdchem.Mol | None = None
    ):
        compound = Compound(mol, src_mol, compound_set=self)
        self.__compounds.append(compound)
        return compound

    def remove_compound(self, compound: Compound):
        if compound.compound_set != self:
            raise ValueError("Compound is not from this set.")
        self.__compounds.remove(compound)
        compound.reset_compound_set()
