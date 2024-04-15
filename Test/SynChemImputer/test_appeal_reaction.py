import unittest

from synrbl.SynChemImputer.appel_reaction import (
    AppelReaction,
)


class TestAppelReaction(unittest.TestCase):
    def test_check_alcohol_group(self):
        # Test with a molecule containing an alcohol group
        self.assertTrue(AppelReaction.check_alcohol_group("CCO"))  # Ethanol

        # Test with a molecule not containing an alcohol group
        self.assertFalse(AppelReaction.check_alcohol_group("CCC"))  # Propane

    def test_check_appel_reaction(self):
        # Test with a valid Appel reaction
        self.assertTrue(
            AppelReaction.check_appel_reaction("ClC(Cl)(Cl)Cl.CCO")
        )  # TCM with ethanol

        # Test with an invalid Appel reaction
        self.assertFalse(AppelReaction.check_appel_reaction("CCO"))  # Only ethanol

    def test_check_missing_reagent(self):
        # Test with missing Triphenylphosphine
        self.assertTrue(AppelReaction.check_missing_reagent("ClC(Cl)(Cl)Cl.CCO"))

        # Test with present Triphenylphosphine
        self.assertFalse(
            AppelReaction.check_missing_reagent(
                "ClC(Cl)(Cl)Cl.CCO.c1ccccc1P(c2ccccc2)c3ccccc3"
            )
        )

    def test_check_missing_products(self):
        # Test with missing Triphenylphosphine oxide
        self.assertTrue(AppelReaction.check_missing_products("CCCl"))

        # Test with present Triphenylphosphine oxide
        self.assertFalse(
            AppelReaction.check_missing_products("CCCl.O=P(c1ccccc1)(c2ccccc2)c3ccccc3")
        )

    def test_fit(self):
        # Test the fit function with various scenarios
        appel_reactor = AppelReaction()
        reaction_dict = {"rmsi_col": "CCO.ClC(Cl)(Cl)Cl>>CCCl"}
        updated_dict = appel_reactor.fit(reaction_dict, "rmsi_col")
        self.assertIn(
            AppelReaction.TPPO, updated_dict["rmsi_col"]
        )  # Check for TPPO in products
        self.assertIn(
            AppelReaction.TCM_product, updated_dict["rmsi_col"]
        )  # Check for TCM product

        # Add more test cases as necessary

    def test_fit_missing_tp_po_and_tcm_product(self):
        appel_reactor = AppelReaction()
        reaction_dict = {"rmsi_col": "CCO.ClC(Cl)(Cl)Cl>>CCCl"}
        updated_dict = appel_reactor.fit(reaction_dict, "rmsi_col")
        self.assertIn(AppelReaction.TPPO, updated_dict["rmsi_col"])
        self.assertIn(AppelReaction.TCM_product, updated_dict["rmsi_col"])

    def test_fit_with_tcm_missing_tp_and_po(self):
        appel_reactor = AppelReaction()
        reaction_dict = {"rmsi_col": "CO.ClC(Cl)(Cl)Cl>>CCl"}
        updated_dict = appel_reactor.fit(reaction_dict, "rmsi_col")
        self.assertIn(AppelReaction.TPPO, updated_dict["rmsi_col"])

    def test_fit_with_existing_tcm_product(self):
        appel_reactor = AppelReaction()
        reaction_dict = {"rmsi_col": "CCO.ClC(Cl)(Cl)Cl>>CCCl.'ClC(Cl)Cl'"}
        updated_dict = appel_reactor.fit(reaction_dict, "rmsi_col")
        self.assertIn(AppelReaction.TPPO, updated_dict["rmsi_col"])

    def test_fit_with_existing_tp_po(self):
        appel_reactor = AppelReaction()
        reaction_dict = {
            "rmsi_col": "CCO.ClC(Cl)(Cl)Cl>>CCCl.'O=P(c1ccccc1)(c2ccccc2)c3ccccc3'"
        }
        updated_dict = appel_reactor.fit(reaction_dict, "rmsi_col")
        self.assertIn(AppelReaction.TPP, updated_dict["rmsi_col"])

    def test_fit_with_existing_tpp(self):
        appel_reactor = AppelReaction()
        reaction_dict = {
            "rmsi_col": "CCO.ClC(Cl)(Cl)Cl.c1ccccc1P(c2ccccc2)c3ccccc3>>CCCl'"
        }
        updated_dict = appel_reactor.fit(reaction_dict, "rmsi_col")
        self.assertIn(AppelReaction.TPPO, updated_dict["rmsi_col"])
        self.assertIn(AppelReaction.TCM_product, updated_dict["rmsi_col"])


if __name__ == "__main__":
    unittest.main()
