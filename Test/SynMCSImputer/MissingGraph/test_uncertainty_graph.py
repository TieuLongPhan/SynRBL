import unittest

from synrbl.SynMCSImputer.MissingGraph.uncertainty_graph import GraphMissingUncertainty


class TestGraphMissingUncertainty(unittest.TestCase):
    def setUp(self):
        self.graph_data = [
            {"boundary_atoms_products": [None, None], "smiles": ["C1=CC=CC=C1", None]},
            {
                "boundary_atoms_products": [{"C": 0}],
                "smiles": ["C1=CC=CC=C1.C2=CC=CC=C2"],
            },
            {"boundary_atoms_products": [], "smiles": ["C1=CC=CC=C1.C2=CC=CC=C2.O"]},
            {"boundary_atoms_products": [{"O": 2}], "smiles": ["O.O.O"]},
            {"boundary_atoms_products": [{"O": 2}], "smiles": ["O"]},
        ]
        self.threshold = 2
        self.graph_missing_uncertainty = GraphMissingUncertainty(
            self.graph_data, self.threshold
        )

    def test_check_boundary(self):
        without_boundary = GraphMissingUncertainty.check_boundary(self.graph_data)
        self.assertEqual(without_boundary, [0, 2])

    def test_check_fragments(self):
        graph_uncertain = GraphMissingUncertainty.check_fragments(
            self.graph_data, self.threshold
        )
        self.assertEqual(graph_uncertain, [1, 2, 3])

    def test_fit(self):
        updated_graph_data = self.graph_missing_uncertainty.fit()
        self.assertFalse(updated_graph_data[0]["Certainty"])
        self.assertFalse(updated_graph_data[1]["Certainty"])
        self.assertFalse(updated_graph_data[2]["Certainty"])
        self.assertFalse(updated_graph_data[3]["Certainty"])
        self.assertTrue(updated_graph_data[4]["Certainty"])


if __name__ == "__main__":
    unittest.main()
