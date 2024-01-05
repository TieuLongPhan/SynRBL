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
    ):
        self.mol: rdchem.Mol = _to_mol(mol)
        if self.mol is None:
            raise ValueError(
                "Argument 'mol' must be either a valid smiles or an rdkit molecule."
            )
        self.src_mol = _to_mol(src_mol)
        self.boundaries: list[Boundary] = []
        self.rules = []

    @property
    def smiles(self) -> str:
        return rdmolfiles.MolToSmiles(self.mol)

    @property
    def src_smiles(self) -> str | None:
        if self.src_mol is None:
            return None
        return rdmolfiles.MolToSmiles(self.src_mol)

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


def build_compounds(data_dict) -> list[Compound]:
    src_smiles = data_dict["sorted_reactants"]
    smiles = data_dict["smiles"]
    boundaries = data_dict["boundary_atoms_products"]
    neighbors = data_dict["nearest_neighbor_products"]
    if len(smiles) != len(src_smiles):
        raise ValueError(
            "Smiles and sorted reactants are not of the same length. ({} != {})".format(
                len(smiles), len(src_smiles)
            )
        )
    if len(boundaries) != len(neighbors):
        raise ValueError(
            "Boundaries and nearest neighbors are not of the same length. ({} != {})".format(
                len(boundaries), len(neighbors)
            )
        )
    compounds = []
    s_i = 0
    for s, ss in zip(smiles, src_smiles):
        if s is None:
            continue
        if len(boundaries) <= s_i:
            raise ValueError("Missing boundary for substructure compound.")
        b = boundaries[s_i]
        n = neighbors[s_i]
        c = Compound(s, src_mol=ss)
        if len(b) != len(n):
            raise ValueError(
                "Boundary and neighbor missmatch. (boundary={}, neighbor={})".format(
                    b, n
                )
            )
        for bi, ni in zip(b, n):
            bi_s, bi_i = list(bi.items())[0]
            ni_s, ni_i = list(ni.items())[0]
            c.add_boundary(bi_i, symbol=bi_s, neighbor_index=ni_i, neighbor_symbol=ni_s)
        compounds.append(c)
        s_i += 1
    if len(boundaries) != s_i:
        raise ValueError(
            "Not enough compounds for boundaries. (smiles={}, boundaries={})".format(
                smiles, boundaries
            )
        )
    return compounds
