import unittest
from synrbl.SynChemImputer.curate_reduction import CurationReduction


class TestCurationReduction(unittest.TestCase):

    def setUp(self):
        self.curate = CurationReduction()
        self.data = [
            {"R-id": "aldehyde", "reactions": "CC=O.[H].[H]>>CCO"},
            {"R-id": "ketone", "reactions": "CC(=O)C.[H].[H]>>CC(O)C"},
            {"R-id": "acid", "reactions": "CC(=O)O.[H].[H].[H].[H]>>CCO.O"},
            {"R-id": "ester", "reactions": "CC(=O)OC.[H].[H].[H].[H]>>CCO.CO"},
        ]

    def test_aldehyde(self):
        result_ion = self.curate.process_dict(
            self.data[0],
            "reactions",
            return_all=True,
            neutralize=False,
            compounds_template=self.curate.compounds_template,
            reaction_templates=self.curate.reaction_templates,
        )
        self.assertEqual(
            result_ion["curated_reaction"][1], "CC=O.[BH4-].[Na+].[H+]>>CCO.[BH3].[Na+]"
        )

        result_neutral = self.curate.process_dict(
            self.data[0],
            "reactions",
            return_all=True,
            neutralize=True,
            compounds_template=self.curate.compounds_template,
            reaction_templates=self.curate.reaction_templates,
        )
        self.assertEqual(
            result_neutral["curated_reaction"][1],
            "CC=O.[BH4-].[Na+].Cl>>CCO.[BH3].[Na][Cl]",
        )

    def test_ketone(self):
        result_ion = self.curate.process_dict(
            self.data[1],
            "reactions",
            return_all=True,
            neutralize=False,
            compounds_template=self.curate.compounds_template,
            reaction_templates=self.curate.reaction_templates,
        )
        self.assertEqual(
            result_ion["curated_reaction"][2],
            "CC(=O)C.[BH3-]C#N.[Na+].[H+]>>CC(O)C.[BH2]C#N.[Na+]",
        )

        result_neutral = self.curate.process_dict(
            self.data[1],
            "reactions",
            return_all=True,
            neutralize=True,
            compounds_template=self.curate.compounds_template,
            reaction_templates=self.curate.reaction_templates,
        )
        self.assertEqual(
            result_neutral["curated_reaction"][2],
            "CC(=O)C.[BH3-]C#N.[Na+].Cl>>CC(O)C.[BH2]C#N.[Na][Cl]",
        )

    def test_acid(self):
        result_ion = self.curate.process_dict(
            self.data[2],
            "reactions",
            return_all=True,
            neutralize=False,
            compounds_template=self.curate.compounds_template,
            reaction_templates=self.curate.reaction_templates,
        )
        self.assertEqual(
            result_ion["curated_reaction"][0],
            (
                "CC(=O)O.[AlH4-].[Li+].[H+].[AlH4-].[Li+].[H+]>>"
                + "CCO.O.[AlH3].[Li+].[AlH3].[Li+]"
            ),
        )

        result_neutral = self.curate.process_dict(
            self.data[2],
            "reactions",
            return_all=True,
            neutralize=True,
            compounds_template=self.curate.compounds_template,
            reaction_templates=self.curate.reaction_templates,
        )
        self.assertEqual(
            result_neutral["curated_reaction"][0],
            (
                "CC(=O)O.[AlH4-].[Li+].Cl.[AlH4-].[Li+].Cl>>"
                + "CCO.O.[AlH3].[Li][Cl].[AlH3].[Li][Cl]"
            ),
        )

    def test_ester(self):
        result_ion = self.curate.process_dict(
            self.data[3],
            "reactions",
            return_all=True,
            neutralize=False,
            compounds_template=self.curate.compounds_template,
            reaction_templates=self.curate.reaction_templates,
        )
        self.assertEqual(
            result_ion["curated_reaction"][0],
            (
                "CC(=O)OC.[BH4-].[Na+].[H+].[BH4-].[Na+].[H+]>>"
                + "CCO.CO.[BH3].[Na+].[BH3].[Na+]"
            ),
        )

        result_neutral = self.curate.process_dict(
            self.data[3],
            "reactions",
            return_all=True,
            neutralize=True,
            compounds_template=self.curate.compounds_template,
            reaction_templates=self.curate.reaction_templates,
        )
        print(result_neutral)
        self.assertEqual(
            result_neutral["curated_reaction"][0],
            (
                "CC(=O)OC.[BH4-].[Na+].Cl.[BH4-].[Na+].Cl>>"
                + "CCO.CO.[BH3].[Na][Cl].[BH3].[Na][Cl]"
            ),
        )

    def test_parallel_curate(self):
        result = self.curate.parallel_curate(
            self.data, n_jobs=2, verbose=2, return_all=True, neutralize=False
        )

        self.assertEqual(
            result[0]["curated_reaction"][1], "CC=O.[BH4-].[Na+].[H+]>>CCO.[BH3].[Na+]"
        )
        self.assertEqual(
            result[1]["curated_reaction"][2],
            "CC(=O)C.[BH3-]C#N.[Na+].[H+]>>CC(O)C.[BH2]C#N.[Na+]",
        )
        self.assertEqual(
            result[2]["curated_reaction"][0],
            (
                "CC(=O)O.[AlH4-].[Li+].[H+].[AlH4-].[Li+].[H+]>>"
                + "CCO.O.[AlH3].[Li+].[AlH3].[Li+]"
            ),
        )
        self.assertEqual(
            result[3]["curated_reaction"][0],
            (
                "CC(=O)OC.[BH4-].[Na+].[H+].[BH4-].[Na+].[H+]>>"
                + "CCO.CO.[BH3].[Na+].[BH3].[Na+]"
            ),
        )


if __name__ == "__main__":
    unittest.main()
