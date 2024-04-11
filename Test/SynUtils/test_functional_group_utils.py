import unittest
import rdkit.Chem.rdmolfiles as rdmolfiles
import synrbl.SynUtils.functional_group_utils as fgutils


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


class TestMappingPermutations(unittest.TestCase):
    def test_1(self):
        self.assertEqual([[]], fgutils.get_mapping_permutations([], ["C"]))

    def test_2(self):
        self.assertEqual(
            [[(0, 1)]], fgutils.get_mapping_permutations(["C"], ["S", "C"])
        )

    def test_3(self):
        self.assertEqual(
            [[(0, 0)], [(0, 1)]], fgutils.get_mapping_permutations(["C"], ["C", "C"])
        )

    def test_4(self):
        self.assertEqual(
            [[(0, 0), (1, 1)], [(0, 1), (1, 0)]],
            fgutils.get_mapping_permutations(["C", "C"], ["C", "C"]),
        )


class TestMCS(unittest.TestCase):
    def _call(self, smiles, anchor, pattern_smiles, panchor=None, expected_match=True):
        mol = rdmolfiles.MolFromSmiles(smiles)
        pattern_mol = rdmolfiles.MolFromSmiles(pattern_smiles)
        match, mapping = fgutils.pattern_match(
            mol, anchor, pattern_mol, pattern_anchor=panchor
        )
        self.assertEqual(expected_match, match)
        return mapping

    def test_simple_pattern_match(self):
        self.assertEqual([(2, 0)], self._call("CCO", 2, "O", 0))
        self.assertEqual([(2, 0), (1, 1)], self._call("CCO", 2, "OC", 0))
        self.assertEqual([(1, 1), (2, 0)], self._call("CCO", 1, "OC", 1))
        self.assertEqual([(2, 0), (1, 1), (0, 2)], self._call("CCO", 2, "OCC", 0))
        self.assertEqual([(2, 2), (1, 1), (0, 0)], self._call("CCO", 2, "CCO", 2))

    def test_ring_pattern_match(self):
        self.assertEqual([(1, 0), (3, 1), (2, 2)], self._call("CC1NO1", 1, "CON", 0))
        self.assertEqual([(1, 0), (3, 2), (2, 1)], self._call("CC1NO1", 1, "C1NO1", 0))

    def test_invalid_path_pattern(self):
        self.assertEqual(
            [(3, 0), (4, 1), (5, 2)], self._call("CC(O)CC(N)O", 3, "CCN", 0)
        )

    def test_match_star(self):
        self.assertEqual([(1, 1), (2, 2), (0, 0)], self._call("CC(C)C", 1, "CCC", 1))

    def test_find_no_pattern_match(self):
        self._call("CCO", 0, "O", 0, expected_match=False)
        self._call("CCO", 1, "O", 0, expected_match=False)
        self._call("CC1NO1", 0, "CON", 0, expected_match=False)
        self._call("CCNO", 1, "C1NO1", 0, expected_match=False)
        self._call("CC(O)CC(N)O", 2, "CCN", 0, expected_match=False)

    def test_fit_without_pattern_anchor(self):
        self.assertEqual([(0, 0), (1, 1), (2, 2)], self._call("CCO", 0, "CCO"))
        self.assertEqual([(1, 1), (2, 2), (0, 0)], self._call("CCO", 1, "CCO"))
        self.assertEqual([(2, 2), (1, 1), (0, 0)], self._call("CCO", 2, "CCO"))

    def test_match_with_bond_order(self):
        self._call("CC=O", 1, "CO", expected_match=False)
        self.assertEqual(
            [(1, 0), (2, 1)], self._call("CC=O", 1, "C=O", expected_match=True)
        )

    def test_find_double_bond_match(self):
        self.assertEqual(
            [(1, 1), (2, 0)], self._call("CP(=O)(O)O", 1, "O=P", expected_match=True)
        )


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
        self.__test_fg("Oc1ccc[nH]1", "phenol")

    def test_alcohol(self):
        # Ethanol
        self.__test_fg("CCO", "alcohol", [1, 2])

    def test_ether(self):
        # Methylether
        self.__test_fg("COC", "ether", [0, 1, 2])
        self.__test_fg("COc1cccc(OC)c1OC", "ether", [0, 1, 2, 9, 10, 11, 6, 7, 8])
        self.__test_fg("COc1ccc[nH]1", "ether", [0, 1, 2])
        self.__test_fg("COc1ccccc1", "ether", [0, 1, 2])

    def test_enol(self):
        # 3-pentanone enol
        self.__test_fg("CCC(O)=CC", "enol", [2, 3, 4])

    def test_amid(self):
        # Asparagine
        self.__test_fg("NC(=O)CC(N)C(=O)O", "amid", [0, 1, 2])

    # def test_acyl(self):
    #    # Acetyl cloride
    #    self.__test_fg("CC(=O)[Cl]", "acyl", [1, 2])

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
        # Methcatione
        self.__test_fg("CNC(C)C(=O)c1ccccc1", "amin", [0, 1, 2])

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
        self.__test_fg("CCSCC", "thioether", [1, 2, 3])

    def test_thioester(self):
        # Methyl thionobenzonat
        self.__test_fg("CSC(=O)c1ccccc1", "thioester", [1, 2, 3])
        self.__test_fg("COC(=S)c1ccccc1", "thioester", [1, 2, 3])

    def test_keton(self):
        # Methcatione
        self.__test_fg("CNC(C)C(=O)c1ccccc1", "keton", [4, 5])

    def test_aldehyde(self):
        # Methcatione
        self.__test_fg("CC=O", "aldehyde", [1, 2])
