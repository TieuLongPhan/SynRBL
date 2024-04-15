import unittest

from synrbl.SynChemImputer.peroxide_imputer import (
    PeroxidGroupImputer,
)


class TestPeroxidGroupImputer(unittest.TestCase):
    def setUp(self):
        self.pero_imputer = PeroxidGroupImputer()

    def test_fix_peroxid_group(self):
        # Test data and expected result for the first reaction
        reactions_dict = [{"reactions": "COOC.C>>C"}]
        expected_result = {
            "reactions": "COOC.C>>C.CO.CO"
        }  # Assuming this is the expected updated reaction

        # Applying the fix method
        self.pero_imputer.fix(reactions_dict[0], "reactions")

        # Assertion
        self.assertEqual(reactions_dict[0], expected_result)

    def test_fix_peracid_group(self):
        # Test data and expected result for the second reaction
        reactions_dict = [{"reactions": "C=C.C1=CC(=CC(=C1)Cl)C(=O)OO>>O1CC1"}]
        expected_result = {
            "reactions": "C=C.C1=CC(=CC(=C1)Cl)C(=O)OO>>O1CC1.O=C(O)c1cccc(Cl)c1"
        }  # Updated reaction

        # Applying the fix method
        self.pero_imputer.fix(reactions_dict[0], "reactions")

        # Assertion
        self.assertEqual(reactions_dict[0], expected_result)

    def test_fix_peracid_group_not_impute(self):
        # Test data and expected result for the third reaction
        reactions_dict = [
            {"reactions": "C=C.C1=CC(=CC(=C1)Cl)C(=O)OO>>O=C(O)c1cccc(Cl)c1.O1CC1"}
        ]
        expected_result = {
            "reactions": "C=C.C1=CC(=CC(=C1)Cl)C(=O)OO>>O=C(O)c1cccc(Cl)c1.O1CC1"
        }  # No change

        # Applying the fix method
        self.pero_imputer.fix(reactions_dict[0], "reactions")

        # Assertion
        self.assertEqual(reactions_dict[0], expected_result)


if __name__ == "__main__":
    unittest.main()
