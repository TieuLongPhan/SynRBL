import unittest
from synrbl.SynChemImputer.post_process import PostProcess


class TestPostProcess(unittest.TestCase):
    def setUp(self):
        self.data = [
            {
                "R-id": 1,
                "reactions": "CC(=O)C.[H].[H]>>CC(O)C",
            },
            {
                "R-id": 2,
                "reactions": "CCO.[O]>>CC=O",
            },
            {
                "R-id": 3,
                "reactions": "C=CC>>C1CC1",
            },
        ]

    def test_label_reactions(self):
        post_process = PostProcess()
        result = post_process.label_reactions(self.data[0])
        self.assertEqual(result["label"], "Reduction")
        result = post_process.label_reactions(self.data[1])
        self.assertEqual(result["label"], "Oxidation")
        result = post_process.label_reactions(self.data[2])
        self.assertEqual(result["label"], "unspecified")

    def test_fit(self):
        post_process = PostProcess(n_jobs=4, verbose=1)
        result = post_process.fit(self.data)
        self.assertEqual(result[1]["curated_reaction"], "CC(=O)C.[H][H]>>CC(O)C")
        self.assertEqual(
            result[2]["curated_reaction"],
            (
                "CCO.O=[Cr](Cl)(-[O-])=O.c1cc[nH+]cc1>>"
                + "CC=O.O=[Cr](O)O.c1cc[nH+]cc1.[Cl-]"
            ),
        )


if __name__ == "__main__":
    unittest.main()
