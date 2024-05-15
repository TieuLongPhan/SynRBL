import unittest
import unittest.mock as mock
import rdkit.Chem.rdmolfiles as rdmolfiles

from synrbl.SynMCSImputer.mcs_based_method import impute_reaction, build_compounds


class TestBuildCompound(unittest.TestCase):
    def _get_dict(self, smiles, src_smiles, bounds, neighbors, mcs_results):
        return {
            "smiles": smiles,
            "sorted_reactants": src_smiles,
            "boundary_atoms_products": bounds,
            "nearest_neighbor_products": neighbors,
            "mcs_results": mcs_results,
        }

    def test_missing_boundaries(self):
        data = self._get_dict(
            ["O", "C"], ["CO", "CO"], [[]], [[]], ["something", "something"]
        )
        with self.assertRaises(ValueError):
            build_compounds(data)

    def test_simple(self):
        data = self._get_dict(["O"], ["CO"], [[{"O": 0}]], [[{"C": 0}]], ["something"])
        cset = build_compounds(data)
        self.assertEqual(1, len(cset))
        self.assertEqual(1, len(cset.compounds[0].boundaries))
        self.assertEqual("O", cset.compounds[0].boundaries[0].get_atom().GetSymbol())
        self.assertEqual("CO", rdmolfiles.MolToSmiles(cset.compounds[0].src_mol))

    def test_catalyst_compound1(self):
        data = self._get_dict([None], ["N"], [None], [None], [""])
        cset = build_compounds(data)
        self.assertEqual(1, len(cset))
        self.assertEqual(0, len(cset.compounds[0].boundaries))
        self.assertEqual("N", rdmolfiles.MolToSmiles(cset.compounds[0].src_mol))
        self.assertTrue(cset.compounds[0].is_catalyst)

    def test_catalyst_compound2(self):
        data = self._get_dict([None], ["CO"], [None], [None], [""])
        cset = build_compounds(data)
        self.assertEqual(1, len(cset))
        self.assertEqual(0, len(cset.compounds[0].boundaries))
        self.assertEqual("CO", rdmolfiles.MolToSmiles(cset.compounds[0].src_mol))
        self.assertTrue(cset.compounds[0].is_catalyst)

    def test_O_is_not_a_catalyst(self):
        data = self._get_dict([None], ["O"], [None], [None], [""])
        cset = build_compounds(data)
        self.assertEqual(1, len(cset))
        self.assertEqual(1, len(cset.compounds[0].boundaries))
        self.assertEqual("O", rdmolfiles.MolToSmiles(cset.compounds[0].src_mol))
        self.assertEqual("O", cset.compounds[0].boundaries[0].symbol)
        self.assertFalse(cset.compounds[0].is_catalyst)

    def test_with_none_compound(self):
        data = self._get_dict([None], ["O"], [None], [None], ["something"])
        compounds = build_compounds(data)
        self.assertEqual(0, len(compounds))


class TestImputeReaction(unittest.TestCase):
    def _reac_dict(
        self,
        old_reaction,
        smiles,
        boundaries,
        neighbors,
        sorted_reactants,
        mcs_results,
        carbon_balance_check="products",
        issue="",
    ):
        return {
            "old_reaction": old_reaction,
            "smiles": smiles,
            "boundary_atoms_products": boundaries,
            "nearest_neighbor_products": neighbors,
            "sorted_reactants": sorted_reactants,
            "mcs_results": mcs_results,
            "carbon_balance_check": carbon_balance_check,
            "issue": issue,
        }

    @mock.patch("synrbl.SynMCSImputer.mcs_based_method.is_carbon_balanced")
    @mock.patch("synrbl.SynMCSImputer.mcs_based_method.merge")
    @mock.patch("synrbl.SynMCSImputer.mcs_based_method.build_compounds")
    def test_successful_imputation(self, m_bc, m_merge, _):
        old_reaction = "A>>B"

        m_bc.return_value = ["Compound"]

        m_mergeresult = mock.MagicMock()
        m_mergeresult.smiles = "X"
        m_rule = mock.MagicMock()
        m_rule.name = "Mock Rule"
        m_mergeresult.rules = [m_rule]
        m_merge.return_value = m_mergeresult

        r = self._reac_dict(
            old_reaction,
            ["C", "D"],
            [[{"O": 0}], [{"C": 1}]],
            [[{"N": 1}], [{"O": 2}]],
            ["E", "F"],
            ["G", "H"],
        )

        result, rules = impute_reaction(
            r,
            reaction_col="old_reaction",
            issue_col="issue",
            carbon_balance_col="carbon_balance_check",
            mcs_data_col="mcs_results",
        )

        self.assertEqual("", r["issue"])
        self.assertEqual(old_reaction + ".X", result)
        self.assertEqual(1, len(rules))
        self.assertEqual(m_rule.name, rules[0])

    @mock.patch("synrbl.SynMCSImputer.mcs_based_method.is_carbon_balanced")
    @mock.patch("synrbl.SynMCSImputer.mcs_based_method.merge")
    @mock.patch("synrbl.SynMCSImputer.mcs_based_method.build_compounds")
    def test_carbon_check_fails(self, m_bc, m_merge, m_cec):
        old_reaction = "A>>B"

        m_bc.return_value = ["Compound"]

        m_mergeresult = mock.MagicMock()
        m_mergeresult.smiles = "X"
        m_rule = mock.MagicMock()
        m_rule.name = "Mock Rule"
        m_mergeresult.rules = [m_rule]
        m_merge.return_value = m_mergeresult

        m_cec.return_value = False

        r = self._reac_dict(
            old_reaction,
            ["C", "D"],
            [[{"O": 0}], [{"C": 1}]],
            [[{"N": 1}], [{"O": 2}]],
            ["E", "F"],
            ["G", "H"],
        )

        with self.assertRaises(RuntimeError) as e:
            impute_reaction(
                r,
                reaction_col="old_reaction",
                issue_col="issue",
                carbon_balance_col="carbon_balance_check",
                mcs_data_col="mcs_results",
            )

        self.assertTrue("carbon check", str(e.exception))
