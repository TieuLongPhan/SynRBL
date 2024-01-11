import unittest
import unittest.mock as mock
import rdkit.Chem.rdmolfiles as rdmolfiles

from SynRBL.SynMCSImputer.model import impute_reaction, build_compounds


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
        with self.assertRaises(ValueError) as e:
            build_compounds(data)

    def test_simple(self):
        data = self._get_dict(["O"], ["CO"], [[{"O": 0}]], [[{"C": 0}]], ["something"])
        compounds = build_compounds(data)
        self.assertEqual(1, len(compounds))
        self.assertEqual(1, len(compounds[0].boundaries))
        self.assertEqual("O", compounds[0].boundaries[0].get_atom().GetSymbol())
        self.assertEqual("CO", rdmolfiles.MolToSmiles(compounds[0].src_mol))

    def test_catalysis_compound(self):
        data = self._get_dict([None], ["N"], [None], [None], [""])
        compounds = build_compounds(data)
        self.assertEqual(1, len(compounds))
        self.assertEqual(0, len(compounds[0].boundaries))
        self.assertEqual("N", rdmolfiles.MolToSmiles(compounds[0].src_mol))

    def test_O_is_not_a_catalyst(self):
        data = self._get_dict([None], ["O"], [None], [None], [""])
        compounds = build_compounds(data)
        self.assertEqual(1, len(compounds))
        self.assertEqual(1, len(compounds[0].boundaries))
        self.assertEqual("O", rdmolfiles.MolToSmiles(compounds[0].src_mol))
        self.assertEqual("O", compounds[0].boundaries[0].symbol)

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

    @mock.patch("SynRBL.SynMCSImputer.model.is_carbon_balanced")
    @mock.patch("SynRBL.SynMCSImputer.model.merge")
    @mock.patch("SynRBL.SynMCSImputer.model.build_compounds")
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

        impute_reaction(r)

        self.assertEqual("", r["issue"])
        self.assertEqual(old_reaction + ".X", r["new_reaction"])
        self.assertEqual(1, len(r["rules"]))
        self.assertEqual(m_rule.name, r["rules"][0])

    @mock.patch("SynRBL.SynMCSImputer.model.is_carbon_balanced")
    @mock.patch("SynRBL.SynMCSImputer.model.merge")
    @mock.patch("SynRBL.SynMCSImputer.model.build_compounds")
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

        impute_reaction(r)

        self.assertEqual(old_reaction, r["new_reaction"])
        self.assertNotEqual("", r["issue"])
        self.assertEqual(0, len(r["rules"]))
