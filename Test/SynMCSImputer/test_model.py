import unittest
import unittest.mock as mock

from SynRBL.SynMCSImputer.model import impute_reaction


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

    @mock.patch("SynRBL.SynMCSImputer.model.carbon_equality_check")
    @mock.patch("SynRBL.SynMCSImputer.model.merge")
    @mock.patch("SynRBL.SynMCSImputer.model.build_compounds")
    def test_successful_imputation(self, m_bc, m_merge, m_cec):
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

        self.assertEqual('', r['issue'])
        self.assertEqual(old_reaction + ".X", r['new_reaction'])
        self.assertEqual(1, len(r['rules']))
        self.assertEqual(m_rule.name, r['rules'][0])

    @mock.patch("SynRBL.SynMCSImputer.model.carbon_equality_check")
    @mock.patch("SynRBL.SynMCSImputer.model.merge")
    @mock.patch("SynRBL.SynMCSImputer.model.build_compounds")
    def test_carbon_check_fails(self, m_bc, m_merge, m_cec):
        old_reaction = "A>>B"
        err_msg = "Mock Error"

        m_bc.return_value = ["Compound"]

        m_mergeresult = mock.MagicMock()
        m_mergeresult.smiles = "X"
        m_rule = mock.MagicMock()
        m_rule.name = "Mock Rule"
        m_mergeresult.rules = [m_rule]
        m_merge.return_value = m_mergeresult

        m_cec.side_effect = mock.Mock(side_effect=ValueError(err_msg))

        r = self._reac_dict(
            old_reaction,
            ["C", "D"],
            [[{"O": 0}], [{"C": 1}]],
            [[{"N": 1}], [{"O": 2}]],
            ["E", "F"],
            ["G", "H"],
        )

        impute_reaction(r)

        self.assertEqual(old_reaction, r['new_reaction'])
        self.assertEqual(err_msg, r['issue'])
        self.assertEqual(0, len(r['rules']))
