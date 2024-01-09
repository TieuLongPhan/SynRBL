import unittest
import rdkit.Chem.rdmolfiles as rdmolfiles
import SynRBL.SynUtils.functional_group_utils as fgutils

class TestReduce(unittest.TestCase):
    def test_1(self):
        mol = rdmolfiles.MolFromSmiles("CCCOC(C)=O")
        mol_r, _ = fgutils.trim_mol(mol, 6, 1)
        self.assertEqual("C=O", rdmolfiles.MolToSmiles(mol_r))

    def test_2(self):
        mol = rdmolfiles.MolFromSmiles("CCCOC(C)=O")
        mol_r, _ = fgutils.trim_mol(mol, 3, 1)
        self.assertEqual("COC", rdmolfiles.MolToSmiles(mol_r))

    def test_3(self):
        mol = rdmolfiles.MolFromSmiles("CCCOC(C)=O")
        mol_r, _ = fgutils.trim_mol(mol, 4, 1)
        self.assertEqual("CC(=O)O", rdmolfiles.MolToSmiles(mol_r))

    def test_4(self):
        mol = rdmolfiles.MolFromSmiles("COc1ccccc1")
        mol_r, _ = fgutils.trim_mol(mol, 1, 1)
        self.assertEqual("COc", rdmolfiles.MolToSmiles(mol_r))


class TestFGConfig(unittest.TestCase):
    def test_init_anti_pattern(self):
        config = fgutils.FGConfig("CN", anti_pattern=["C=O", "Nc1ccccc1", "CCC"])
        self.assertEqual(3, len(config.anti_pattern))
        self.assertEqual("Nc1ccccc1", rdmolfiles.MolToSmiles(config.anti_pattern[0]))
        self.assertEqual("CCC", rdmolfiles.MolToSmiles(config.anti_pattern[1]))
        self.assertEqual("C=O", rdmolfiles.MolToSmiles(config.anti_pattern[2]))

    def test_init_with_multiple_patterns(self):
        config = fgutils.FGConfig(["O=CS", "OC=S"])
        self.assertEqual(2, len(config.pattern))
        self.assertEqual(2, len(config.groups))
        self.assertEqual("O=CS", rdmolfiles.MolToSmiles(config.groups[0]))
        self.assertEqual("OC=S", rdmolfiles.MolToSmiles(config.groups[1]))


class TestFunctionalGroupCheck(unittest.TestCase):
    def __test_fg(self, smiles, group_name, indices=None):
        groups = list(fgutils.functional_group_config.keys())
        assert group_name in groups, "Functional group {} is not implemented.".format(
            group_name
        )

        mol = rdmolfiles.MolFromSmiles(smiles)
        indices = (
            [i for i in range(len(mol.GetAtoms()))] if indices is None else indices
        )
        for a in mol.GetAtoms():
            idx = a.GetIdx()
            pos_groups = []
            for g in groups:
                result = fgutils.is_functional_group(mol, g, idx)
                if result:
                    pos_groups.append(g)
            if idx in indices:
                if len(pos_groups) == 1 and group_name in pos_groups:
                    pass
                else:
                    self.assertTrue(
                        False,
                        (
                            "Atom {} ({}) in {} was expected to be in functional "
                            + "group {} but was identified as {}."
                        ).format(a.GetSymbol(), idx, smiles, group_name, pos_groups),
                    )
            else:
                self.assertTrue(
                    group_name not in pos_groups,
                    (
                        "Atom {} ({}) in {} was expected to be not part of "
                        + "functional group {} but was identified as {}."
                    ).format(a.GetSymbol(), idx, smiles, group_name, pos_groups),
                )

    def test_phenol(self):
        # 2,3 Xylenol
        self.__test_fg("Cc1cccc(O)c1C", "phenol", [1, 2, 3, 4, 5, 6, 7])

    def test_alcohol(self):
        # Ethanol
        self.__test_fg("CCO", "alcohol", [1, 2])

    def test_ether(self):
        # Methylether
        self.__test_fg("COC", "ether", [1])

    def test_enol(self):
        # 3-pentanone enol
        self.__test_fg("CCC(O)=CC", "enol", [2, 3, 4])

    def test_amid(self):
        # Asparagine
        self.__test_fg("NC(=O)CC(N)C(=O)O", "amid", [0, 1, 2])

    def test_acyl(self):
        # Acetyl cloride
        self.__test_fg("CC(=O)[Cl]", "acyl", [1, 2])

    def test_diol(self):
        self.__test_fg("OCO", "diol")

    def test_hemiacetal(self):
        self.__test_fg("COCO", "hemiacetal")

    def test_acetal(self):
        self.__test_fg("COCOC", "acetal")

    def test_urea(self):
        self.__test_fg("O=C(N)O", "urea")

    def test_carbonat(self):
        self.__test_fg("O=C(O)O", "carbonat")
        self.__test_fg("COC(=O)O", "carbonat", [1, 2, 3, 4])

    def test_ester(self):
        # Methyl acetate
        self.__test_fg("COC(C)=O", "ester", [1, 2, 4])

    def test_anhydrid(self):
        self.__test_fg("O=C(C)OC=O", "anhydrid")

    def test_acid(self):
        # Acetic acid
        self.__test_fg("CC(=O)O", "acid", [1, 2, 3])

    def test_anilin(self):
        self.__test_fg("Nc1ccccc1", "anilin")

    def test_amin(self):
        # Glycin
        self.__test_fg("NCC(=O)O", "amin", [0, 1])

    def test_nitril(self):
        self.__test_fg("C#N", "nitril")

    def test_hydroxylamin(self):
        self.__test_fg("NO", "hydroxylamin")

    def test_nitrose(self):
        self.__test_fg("N=O", "nitrose")

    def test_nitro(self):
        self.__test_fg("ON=O", "nitro")

    def test_thioether(self):
        # Diethylsulfid
        self.__test_fg("CCSCC", "thioether", [2])

    def test_thioester(self):
        # Methyl thionobenzonat
        self.__test_fg("CSC(=O)c1ccccc1", "thioester", [1, 2, 3])
        self.__test_fg("COC(=S)c1ccccc1", "thioester", [1, 2, 3])
