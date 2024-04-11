import copy
import itertools
import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.rdchem as rdchem
import rdkit.Chem.rdmolfiles as rdmolfiles


class FGConfig:
    def __init__(self, pattern, group_atoms=None, anti_pattern=[], depth=None):
        pattern = pattern if isinstance(pattern, list) else [pattern]
        for p in pattern:
            if p != Chem.CanonSmiles(p):
                # We don't fix the pattern smiles because group_atoms might
                # rely on the pattern
                raise ValueError(
                    (
                        "Pattern must be canonical smiles. (value: {}, expected: {})"
                    ).format(p, Chem.CanonSmiles(p))
                )

        self.pattern = [rdmolfiles.MolFromSmiles(p) for p in pattern]

        if group_atoms is None:
            self.groups = [rdmolfiles.MolFromSmiles(p) for p in pattern]
        else:
            self.groups = [rdchem.RWMol(p) for p in self.pattern]
            for g in self.groups:
                rm_indices = []
                for a in g.GetAtoms():
                    if a.GetIdx() not in group_atoms:
                        rm_indices.append(a.GetIdx())
                for i in sorted(rm_indices, reverse=True):
                    g.RemoveAtom(i)

        anti_pattern = (
            anti_pattern if isinstance(anti_pattern, list) else [anti_pattern]
        )
        self.anti_pattern = sorted(
            [rdmolfiles.MolFromSmiles(s) for s in anti_pattern],
            key=lambda x: len(x.GetAtoms()),
            reverse=True,
        )

        self.max_pattern_size = (
            depth
            if depth is not None
            else np.max([len(c.GetAtoms()) for c in self.pattern + self.anti_pattern])
        )


functional_group_config = {
    "phenol": FGConfig(
        ["Oc1ccccc1", "Oc1ccc[nH]1"], anti_pattern=["COc1ccccc1", "COc1ccc[nH]1"]
    ),
    "alcohol": FGConfig(
        "CO", anti_pattern=["C(=O)O", "C=CO", "COC", "OCO", "Oc1ccccc1", "Oc1ccc[nH]1"]
    ),
    "ether": FGConfig(
        "COC",
        anti_pattern=["OC=S", "C=O", "C=CO", "OCOC", "OC=O", "OCN"],
    ),
    "enol": FGConfig("C=CO"),
    "amid": FGConfig("NC=O", anti_pattern=["O=C(N)O"]),
    "acyl": FGConfig(
        "C=O",
        anti_pattern=[
            "CC(C)=O",
            "SC=O",
            "CC=O",
            "C(N)=O",
            "O=C(O)O",
            "O=C(C)OC",
            "O=C(C)O",
            "CC(=O)OC=O",
        ],
    ),
    "diol": FGConfig("OCO", anti_pattern=["OCOC", "O=C(O)O"]),
    "hemiacetal": FGConfig("COCO", anti_pattern=["COCOC", "O=C(O)O"]),
    "acetal": FGConfig("COCOC"),
    "urea": FGConfig("NC(=O)O"),
    "carbonat": FGConfig("O=C(O)O"),
    "anhydrid": FGConfig("CC(=O)OC=O"),
    "ester": FGConfig("COC(C)=O", group_atoms=[1, 2, 4], anti_pattern=["O=C(C)OC=O"]),
    "acid": FGConfig(
        "CC(=O)O",
        group_atoms=[1, 2, 3],
        anti_pattern=["O=CS", "O=C(C)OC=O", "O=C(C)OC"],
    ),
    "anilin": FGConfig("Nc1ccccc1"),
    "amin": FGConfig("CN", anti_pattern=["C=O", "Nc1ccccc1", "NC=O"]),
    "nitril": FGConfig("C#N"),
    "hydroxylamin": FGConfig("NO", anti_pattern=["O=NO"]),
    "nitrose": FGConfig("N=O", anti_pattern=["O=NO"]),
    "nitro": FGConfig("O=NO"),
    "thioether": FGConfig("CSC", anti_pattern=["O=CS"]),
    "thioester": FGConfig(["O=CS", "OC=S"]),
    "aldehyde": FGConfig(
        "CC=O",
        group_atoms=[1, 2],
        anti_pattern=["CC(C)=O", "SC(=O)C", "COC(C)=O", "NC=O", "CC(=O)O"],
    ),
    "keton": FGConfig("CC(C)=O", group_atoms=[1, 3]),
}


def trim_mol(mol: rdchem.Mol, index: int, depth: int) -> rdchem.Mol:
    def _dfs(
        mol: rdchem.RWMol,
        atom: rdchem.Atom,
        dist: int,
        dist_array: list[int],
    ):
        dist_array[atom.GetIdx()] = dist
        for neighbor in atom.GetNeighbors():
            idx = neighbor.GetIdx()
            if dist + 1 < dist_array[idx]:
                _dfs(mol, neighbor, dist + 1, dist_array)

    atom_cnt = len(mol.GetAtoms())
    dist_array = [atom_cnt + 1 for _ in range(atom_cnt)]
    rwmol = rdchem.RWMol(mol)
    atom = rwmol.GetAtomWithIdx(index)
    _dfs(mol, atom, 0, dist_array)
    for i, d in reversed(list(enumerate(dist_array))):
        if d > depth:
            rwmol.RemoveAtom(i)
    return rwmol, atom.GetIdx()


def get_mapping_permutations(match_symbols, sym_dict):
    mappings = []
    if len(sym_dict) >= len(match_symbols):
        sym_map = [(i, s) for i, s in enumerate(sym_dict)]
        for sym_permut in itertools.permutations(sym_map):
            mapping = []
            is_match = True
            for i, s1 in enumerate(match_symbols):
                if s1 == sym_permut[i][1]:
                    mapping.append((i, sym_permut[i][0]))
                else:
                    is_match = False
                    break
            if is_match:
                mappings.append(mapping)
    return mappings


def pattern_match(mol, anchor, pattern_mol, pattern_anchor=None):
    def _fits(atom, pattern_atom, visited_atoms=[], visited_pattern_atoms=[]):
        atom_idx = atom.GetIdx()
        pattern_atom_idx = pattern_atom.GetIdx()
        assert atom_idx not in visited_atoms
        assert pattern_atom_idx not in visited_pattern_atoms

        visited_atoms = copy.deepcopy(visited_atoms)
        visited_atoms.append(atom_idx)
        visited_pattern_atoms = copy.deepcopy(visited_pattern_atoms)
        visited_pattern_atoms.append(pattern_atom_idx)

        atom_neighbors = [
            (a.GetIdx(), a.GetSymbol(), a)
            for a in atom.GetNeighbors()
            if a.GetIdx() not in visited_atoms
        ]
        pattern_neighbors = [
            (a.GetIdx(), a.GetSymbol(), a)
            for a in pattern_atom.GetNeighbors()
            if a.GetIdx() not in visited_pattern_atoms
        ]

        fits = False
        match = []
        if atom.GetSymbol() == pattern_atom.GetSymbol():
            match.append((atom.GetIdx(), pattern_atom.GetIdx()))
            if len(pattern_neighbors) > 0:
                an_syms = [a[1] for a in atom_neighbors]
                pn_syms = [a[1] for a in pattern_neighbors]
                mappings = get_mapping_permutations(pn_syms, an_syms)
                for mapping in mappings:
                    valid_mapping = True
                    n_matches = set()
                    for pn_i, an_i in mapping:
                        atom_neighbor = atom_neighbors[an_i]
                        pattern_neighbor = pattern_neighbors[pn_i]
                        mol = atom_neighbor[2].GetOwningMol()
                        p_mol = pattern_neighbor[2].GetOwningMol()
                        bond = mol.GetBondBetweenAtoms(
                            atom_idx, atom_neighbor[0]
                        ).GetBondType()
                        p_bond = p_mol.GetBondBetweenAtoms(
                            pattern_atom_idx, pattern_neighbor[0]
                        ).GetBondType()
                        if bond != p_bond:
                            valid_mapping = False
                            break
                        n_fit, n_match = _fits(
                            atom_neighbor[2],
                            pattern_neighbor[2],
                            visited_atoms,
                            visited_pattern_atoms,
                        )
                        if not n_fit:
                            valid_mapping = False
                            break
                        else:
                            n_matches.update(n_match)
                    if valid_mapping:
                        fits = True
                        match.extend(n_matches)
                        break
            else:
                fits = True
        return fits, match

    if pattern_anchor is None:
        atom = mol.GetAtomWithIdx(anchor)
        for pattern_atom in pattern_mol.GetAtoms():
            result = _fits(atom, pattern_atom)
            if result[0]:
                return result
        return False, [[]]
    else:
        atom = mol.GetAtomWithIdx(anchor)
        pattern_atom = pattern_mol.GetAtomWithIdx(pattern_anchor)
        return _fits(atom, pattern_atom)


def check_functional_group(mol: rdchem.Mol, config: FGConfig, index: int) -> bool:
    is_func_group = False

    for p_mol, g_mol in zip(config.pattern, config.groups):
        is_match, _ = pattern_match(mol, index, p_mol)
        if is_match:
            is_match, _ = pattern_match(mol, index, g_mol)
            is_func_group = is_func_group or is_match

    last_len = config.max_pattern_size
    for ap_mol, ap_mol_size in sorted(
        [(m, len(m.GetAtoms())) for m in config.anti_pattern],
        key=lambda x: x[1],
        reverse=True,
    ):
        if not is_func_group:
            break
        ap_mol_size = len(ap_mol.GetAtoms())
        if last_len > ap_mol_size:
            last_len = ap_mol_size
        is_match, _ = pattern_match(mol, index, ap_mol)
        is_func_group = is_func_group and not is_match
    return is_func_group


def is_functional_group(mol: rdchem.Mol, group_name: str, index: int) -> bool:
    if group_name not in functional_group_config.keys():
        raise NotImplementedError(
            "Functional group '{}' is not implemented.".format(group_name)
        )
    config = functional_group_config[group_name]
    return check_functional_group(mol, config, index)
