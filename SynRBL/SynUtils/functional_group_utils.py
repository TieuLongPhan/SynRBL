import numpy as np
import rdkit.Chem as Chem
import rdkit.Chem.rdchem as rdchem
import rdkit.Chem.rdmolfiles as rdmolfiles


class FGConfig:
    def __init__(self, pattern, group_atoms=None, anti_pattern=[], depth=None):
        pattern = pattern if isinstance(pattern, list) else [pattern]
        for p in pattern:
            if p != Chem.CanonSmiles(p):
                # We don't fix the pattern smiles because group_atoms might rely on the pattern
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
    "phenol": FGConfig("Oc1ccccc1"),
    "alcohol": FGConfig(
        "CO", anti_pattern=["C(=O)O", "C=CO", "COC", "OCO", "Oc1ccccc1"]
    ),
    "ether": FGConfig(
        "COC", group_atoms=[1], anti_pattern=["OC=S", "C=O", "C=CO", "OCOC", "OC=O", "OCN"]
    ),
    "enol": FGConfig("C=CO"),
    "amid": FGConfig("NC=O", anti_pattern=["O=C(N)O"]),
    "acyl": FGConfig(
        "C=O", anti_pattern=["SC=O", "C(N)=O", "O=C(O)O", "O=C(C)OC", "O=C(C)O"]
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
    "thioether": FGConfig("CSC", group_atoms=[1], anti_pattern=["O=CS"]),
    "thioester": FGConfig(["O=CS", "OC=S"]),
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


def is_functional_group(mol: rdchem.Mol, group_name: str, index: int) -> bool:
    if group_name not in functional_group_config.keys():
        raise NotImplementedError(
            "Functional group '{}' is not implemented.".format(group_name)
        )
    config = functional_group_config[group_name]

    is_func_group = False

    rmol, index = trim_mol(mol, index, config.max_pattern_size - 1)
    for p_mol, g_mol in zip(config.pattern, config.groups):
        pattern_match = list(rmol.GetSubstructMatch(p_mol))
        if index in pattern_match:
            group_match = list(rmol.GetSubstructMatch(g_mol))
            is_func_group = is_func_group or index in group_match

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
            rmol, index = trim_mol(rmol, index, ap_mol_size - 1)
            last_len = ap_mol_size
        group_match = list(rmol.GetSubstructMatch(ap_mol))
        is_func_group = is_func_group and index not in group_match
    return is_func_group
