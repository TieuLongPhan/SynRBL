import unittest
from pathlib import Path

from unittest.mock import patch
from synrbl.SynMCSImputer.SubStructure.mcs_process import single_mcs_safe, ensemble_mcs


class TestMCSFunctions(unittest.TestCase):
    def setUp(self):
        # Example reaction data for testing
        self.sample_reaction_data = {
            "R-id": "example_id",
            "reactants": "CCO.C[O]",
            "products": "CCO",
            "carbon_balance_check": "balanced",
        }
        self.root_dir = Path(".")
        self.conditions = [
            {
                "RingMatchesRingOnly": True,
                "CompleteRingsOnly": True,
                "method": "MCIS",
                "sort": "MCIS",
                "ignore_bond_order": True,
            }
        ]

    @patch(
        "synrbl.SynMCSImputer.SubStructure.mcs_graph_detector.MCSMissingGraphAnalyzer.fit"
    )
    def test_single_mcs(self, mock_fit):
        # Mocking MCSMissingGraphAnalyzer.fit to return predefined values
        mock_fit.return_value = ([], [], [], None)

        result = single_mcs_safe(self.sample_reaction_data, id_col="R-id")
        self.assertEqual(result["R-id"], "example_id")
        self.assertIsInstance(result["mcs_results"], list)
        self.assertIsInstance(result["sorted_reactants"], list)


if __name__ == "__main__":
    unittest.main()


def test_timeout():
    conditions = [
        {
            "RingMatchesRingOnly": True,
            "CompleteRingsOnly": True,
            "method": "MCIS",
            "sort": "MCIS",
            "ignore_bond_order": True,
            "maxNodes": 1000,
        }
    ]
    data = [
        {
            "id": 0,
            "carbon_balance_check": "products",
            "reactants": "[Br]" + 100 * "[Si](C)(C)O" + "[Si][Br]",
            "products": "O" + 100 * "[Si](C)(C)O" + "[Si]O",
        }
    ]
    result = ensemble_mcs(data, conditions, n_jobs=2)
    assert "timeout" in result[0][0]["issue"]
    assert [] == result[0][0]["mcs_results"]
