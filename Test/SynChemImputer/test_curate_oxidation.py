import unittest
from synrbl.SynChemImputer.curate_oxidation import CurationOxidation


class TestCurationOxidation(unittest.TestCase):

    def setUp(self):
        self.curate = CurationOxidation()
        self.data = [
            {"R-id": "alcohol_carbonyl", "reactions": "CCO.[O]>>CC=O"},
            {"R-id": "alcohol_acid", "reactions": "CCO.[O]>>CC(=O)O"},
            {"R-id": "carbonyl_acid", "reactions": "CC=O.[O]>>CC(=O)O"},
        ]

    def test_alcohol_carbonyl(self):
        result = self.curate.process_dict(
            self.data[0],
            "reactions",
            return_all=True,
            compounds_template=self.curate.compounds_template,
            reaction_templates=self.curate.reaction_templates,
        )
        self.assertEqual(
            result["curated_reaction"][0],
            "CCO.O=[Cr](Cl)(-[O-])=O.c1cc[nH+]cc1>>CC=O.O=[Cr](O)O.c1cc[nH+]cc1.[Cl-]",
        )
        self.assertEqual(result["stoichiometric"][0], [1])

    def test_alcohol_acid(self):
        result = self.curate.process_dict(
            self.data[1],
            "reactions",
            return_all=True,
            compounds_template=self.curate.compounds_template,
            reaction_templates=self.curate.reaction_templates,
        )
        print(result["curated_reaction"][0])
        self.assertEqual(
            result["curated_reaction"][0],
            (
                "CCO.[K][O][Mn](=O)(=O)=O.OS(=O)(=O)O>>"
                + "CC(=O)O.[K][O]S(=O)(=O)[O][K].[Mn]1[O]S(=O)(=O)[O]1"
            ),
        )
        self.assertEqual(result["stoichiometric"][0], [5, 4, 6, 5, 11, 2, 4])

    def test_aldehyde_acid(self):
        result = self.curate.process_dict(
            self.data[2],
            "reactions",
            return_all=True,
            compounds_template=self.curate.compounds_template,
            reaction_templates=self.curate.reaction_templates,
        )
        self.assertEqual(
            result["curated_reaction"][0],
            "CC=O.O=[Mn](=O)(=O)O[K].O>>CC(=O)O.O=[Mn]=O.O[K]",
        )
        self.assertEqual(result["stoichiometric"][0], [5, 4, 6, 5, 4, 1])


if __name__ == "__main__":
    unittest.main()
