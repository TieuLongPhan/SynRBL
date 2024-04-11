import unittest

from synrbl.SynRuleImputer import RuleImputeManager


class TestRuleImputeManager(unittest.TestCase):
    def setUp(self):
        self.db = RuleImputeManager()

    def test_add_entry_valid(self):
        # Test adding a valid entry
        formula = "H2O"
        smiles = "O"
        self.db.add_entry(formula, smiles)
        self.assertEqual(len(self.db.database), 1)
        self.assertEqual(self.db.database[0]["formula"], formula)
        self.assertEqual(self.db.database[0]["smiles"], smiles)

    def test_add_entry_duplicate_formula(self):
        # Test adding an entry with a duplicate formula
        self.db.add_entry("H2O", "O")
        with self.assertRaises(ValueError):
            self.db.add_entry("H2O", "O")

    def test_add_entry_duplicate_smiles(self):
        # Test adding an entry with a duplicate SMILES string
        self.db.add_entry("H2O", "O")
        with self.assertRaises(ValueError):
            self.db.add_entry("Water", "O")

    def test_add_entry_invalid_smiles(self):
        # Test adding an entry with an invalid SMILES string
        with self.assertRaises(ValueError):
            self.db.add_entry("Invalid", "InvalidString")

    def test_add_entries_mixed_validity(self):
        # Test adding multiple entries with mixed validity
        entries = [
            {"formula": "CO2", "smiles": "C=O"},
            {"formula": "Invalid", "smiles": "InvalidString"},
        ]
        invalid_entries = self.db.add_entries(entries)
        self.assertEqual(len(invalid_entries), 1)
        self.assertEqual(invalid_entries[0]["formula"], "Invalid")

    def test_remove_entry(self):
        # Test removing an entry
        self.db.add_entry("H2O", "O")
        self.db.remove_entry("H2O")
        self.assertEqual(len(self.db.database), 0)

    def test_canonicalize_smiles(self):
        # Test canonicalizing a SMILES string
        canonical_smiles = RuleImputeManager.canonicalize_smiles("C1=CC=CC=C1")
        self.assertEqual(canonical_smiles, "c1ccccc1")

    def test_is_valid_smiles(self):
        # Test validating a SMILES string
        self.assertTrue(RuleImputeManager.is_valid_smiles("C1=CC=CC=C1"))
        self.assertFalse(RuleImputeManager.is_valid_smiles("InvalidString"))


if __name__ == "__main__":
    unittest.main()
