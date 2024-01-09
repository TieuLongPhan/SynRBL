import unittest
import rdkit.Chem.rdmolfiles as rdmolfiles
import rdkit.Chem.rdchem as rdchem
import SynRBL.SynMCSImputer.rules as rules
import SynRBL.SynMCSImputer.structure as structure


class TestProperty(unittest.TestCase):
    def __test_prop(self, config, in_p=[], in_n=[], dtype: type[str | int] = str):
        prop = rules.Property(config, dtype=dtype)
        self.assertTrue(all(e in prop.pos_values for e in in_p))
        self.assertTrue(all(e in prop.neg_values for e in in_n))
        self.assertTrue(all(e not in prop.pos_values for e in in_n))
        self.assertTrue(all(e not in prop.neg_values for e in in_p))

    def test_parsing(self):
        self.__test_prop("A", in_p=["A"])
        self.__test_prop(["A", "B"], in_p=["A", "B"])
        self.__test_prop("!A", in_n=["A"])
        self.__test_prop(["!A", "!B"], in_n=["A", "B"])
        self.__test_prop(["!A", "B"], in_p=["B"], in_n=["A"])
        self.__test_prop(
            ["!-1", "!1", "-2", "2"], in_p=[-2, 2], in_n=[-1, 1], dtype=int
        )
        self.__test_prop(["1", "!2"], in_p=[1], in_n=[2], dtype=int)

    def test_parsing_none(self):
        prop = rules.Property(None)
        self.assertEqual(0, len(prop.neg_values))
        self.assertEqual(0, len(prop.pos_values))

    def test_passthrough_behaviour(self):
        prop = rules.Property(None)
        self.assertTrue(prop.check("A"))

    def test_allow_none(self):
        prop = rules.Property(allow_none=True)
        self.assertTrue(prop.check(None))

    def test_dont_allow_non(self):
        prop = rules.Property(allow_none=False)
        with self.assertRaises(ValueError):
            prop.check(None)

    def test_check_value_error(self):
        prop = rules.Property("1", dtype=int)
        with self.assertRaises(ValueError):
            prop.check("2")

    def test_check_property(self):
        prop = rules.Property("A")
        self.assertTrue(prop.check("A"))
        self.assertFalse(prop.check("B"))
        prop = rules.Property("!A")
        self.assertTrue(prop.check("B"))
        self.assertFalse(prop.check("A"))
        prop = rules.Property(["A", "B"])
        self.assertTrue(prop.check("A"))
        self.assertTrue(prop.check("B"))
        self.assertFalse(prop.check("C"))
        prop = rules.Property(["!A", "!B"])
        self.assertFalse(prop.check("A"))
        self.assertFalse(prop.check("B"))
        self.assertTrue(prop.check("C"))

    def test_invalid_config_type(self):
        with self.assertRaises(ValueError):
            rules.Property(8)  # type: ignore


class TestReduce(unittest.TestCase):
    def test_1(self):
        mol = rdmolfiles.MolFromSmiles("CCCOC(C)=O")
        mol_r, _ = rules.reduce(mol, 6, 1)
        self.assertEqual("C=O", rdmolfiles.MolToSmiles(mol_r))

    def test_2(self):
        mol = rdmolfiles.MolFromSmiles("CCCOC(C)=O")
        mol_r, _ = rules.reduce(mol, 3, 1)
        self.assertEqual("COC", rdmolfiles.MolToSmiles(mol_r))

    def test_3(self):
        mol = rdmolfiles.MolFromSmiles("CCCOC(C)=O")
        mol_r, _ = rules.reduce(mol, 4, 1)
        self.assertEqual("CC(=O)O", rdmolfiles.MolToSmiles(mol_r))

    def test_4(self):
        mol = rdmolfiles.MolFromSmiles("COc1ccccc1")
        mol_r, _ = rules.reduce(mol, 1, 1)
        self.assertEqual("COc", rdmolfiles.MolToSmiles(mol_r))


class TestFunctionalGroupCheck(unittest.TestCase):
    def __test_fg(self, smiles, group_name, indices=None):
        groups = list(rules.functional_group_config.keys())
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
                result = rules.is_functional_group(mol, g, idx)
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
        self.__test_fg("OCOC", "hemiacetal")

    def test_acetal(self):
        self.__test_fg("COCOC", "acetal")

    def test_urea(self):
        self.__test_fg("O=C(N)O", "urea")

    def test_carbamat(self):
        self.__test_fg("O=C(O)O", "carbamat")

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

    def test_nitril(self):
        self.__test_fg("C#N", "nitril")

    def test_hydroxylamin(self):
        self.__test_fg("NO", "hydroxylamin")

    def test_nitrose(self):
        self.__test_fg("N=O", "nitrose")

    def test_nitro(self):
        self.__test_fg("ON=O", "nitro")


class TestFunctionalGroupProperty(unittest.TestCase):
    def test_init(self):
        p = rules.FunctionalGroupProperty(["ether", "!ester"])
        self.assertIn("ether", p.pos_values)
        self.assertIn("ester", p.neg_values)

    def test_check_with_wrong_type(self):
        p = rules.FunctionalGroupProperty(["ether", "!ester"])
        values = ["test", 0, None]
        for v in values:
            with self.assertRaises(TypeError):
                p.check(v)

    def test_check_with_incomplete_boundary(self):
        p = rules.FunctionalGroupProperty(["ester"])
        c = structure.Compound("CCC")
        b = c.add_boundary(0)
        with self.assertRaisesRegex(ValueError, r"^[Mm]issing.*$"):
            p.check(b)

    def test_pos_successful_check(self):
        p = rules.FunctionalGroupProperty(["ester"])
        c = structure.Compound("CCC", src_mol="CC(=O)OCCC")
        b = c.add_boundary(0, neighbor_index=3)
        result = p.check(b)
        self.assertEqual(True, result)

    def test_pos_fail_check(self):
        p = rules.FunctionalGroupProperty(["amin"])
        c = structure.Compound("CCC", src_mol="CC(=O)OCCC")
        b = c.add_boundary(0, neighbor_index=3)
        result = p.check(b)
        self.assertEqual(False, result)

    def test_neg_successful_check(self):
        p = rules.FunctionalGroupProperty(["!ester"])
        c = structure.Compound("CCC", src_mol="CC(=O)OCCC")
        b = c.add_boundary(0, neighbor_index=3)
        result = p.check(b)
        self.assertEqual(False, result)

    def test_neg_fail_check(self):
        p = rules.FunctionalGroupProperty(["!amin"])
        c = structure.Compound("CCC", src_mol="CC(=O)OCCC")
        b = c.add_boundary(0, neighbor_index=3)
        result = p.check(b)
        self.assertEqual(True, result)

    def test_defaults_to_true(self):
        p = rules.FunctionalGroupProperty()
        c = structure.Compound("C")
        b = c.add_boundary(0)
        self.assertTrue(p.check(b))


class TestBoundaryCondition(unittest.TestCase):
    def __check_cond(
        self, cond, smiles, idx, expected_result, neighbor=None, src_smiles=None
    ):
        c = rules.Compound(smiles, src_mol=src_smiles)
        b = c.add_boundary(idx, neighbor_index=neighbor)
        actual_result = cond.check(b)
        self.assertEqual(expected_result, actual_result)

    def test_positive_check(self):
        cond = rules.BoundaryCondition(atom=["C", "O"])
        self.__check_cond(cond, "CO", 0, True)
        self.__check_cond(cond, "CO", 1, True)
        self.__check_cond(cond, "[Na+].[Cl-]", 0, False)

    def test_negative_check(self):
        cond = rules.BoundaryCondition(atom=["!Si", "!Cl"])
        self.__check_cond(cond, "C=[Si](C)C", 0, True)
        self.__check_cond(cond, "CO", 1, True)
        self.__check_cond(cond, "[Na+]", 0, True)
        self.__check_cond(cond, "C=[Si](C)C", 1, False)
        self.__check_cond(cond, "[Na+].[Cl-]", 1, False)

    def test_positive_check_with_neighbors(self):
        cond = rules.BoundaryCondition(atom="C", neighbors=["O", "N"])
        self.__check_cond(cond, "C", 0, True, src_smiles="CO", neighbor=1)
