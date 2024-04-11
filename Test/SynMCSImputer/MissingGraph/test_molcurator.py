import unittest
from rdkit import Chem

from synrbl.SynMCSImputer.MissingGraph.molcurator import MoleculeCurator


class TestMoleculeCurator(unittest.TestCase):
    def test_add_hydrogens_to_radicals(self):
        mol = Chem.MolFromSmiles("CC[O]")  # Ethanol with unpaired electron on oxygen
        curated_mol = MoleculeCurator.add_hydrogens_to_radicals(mol)
        self.assertEqual(Chem.MolToSmiles(curated_mol), "CCO")

    def test_standardize_diazo_charge(self):
        mol = Chem.MolFromSmiles("[N-]=[NH2+]")  # Charged diazo compound
        neutralized_mol = MoleculeCurator.standardize_diazo_charge(mol)
        self.assertEqual(Chem.MolToSmiles(neutralized_mol), "N#N")

    def test_manual_kekulize(self):
        smiles = "c1cccn1"  # Benzene with two water molecules
        kekulized_mol = MoleculeCurator.manual_kekulize(smiles)
        self.assertIsNotNone(kekulized_mol)
        self.assertIn("c1cc[nH]c1", Chem.MolToSmiles(kekulized_mol))


if __name__ == "__main__":
    unittest.main()
