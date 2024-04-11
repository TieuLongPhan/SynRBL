import unittest

from rdkit import Chem

from synrbl.SynMCSImputer.SubStructure.mcs_graph_detector import MCSMissingGraphAnalyzer


class TestMCSMissingGraphAnalyzer(unittest.TestCase):
    def setUp(self):
        self.reaction_dict = {
            "reactants": "CC",  # Example reactant SMILES
            "products": "CCO",  # Example product SMILES
            "carbon_balance_check": "balanced",
        }
        self.reaction_dict_unbalanced = {
            "reactants": "CC(=O)NC",  # Example reactant SMILES
            "products": "CC(=O)O",  # Example product SMILES with imbalance
            "carbon_balance_check": "products",
        }
        self.reaction_dict_unbalanced_reactants = {
            "reactants": "O=Cc1ccccc1",  # Example reactant SMILES
            # Example product SMILES with imbalance
            "products": "O=C(c1ccccc1)C(O)c2ccccc2 c1ccc(cc1)C(C(=O)c2ccccc2)O",
            "carbon_balance_check": "reactants",
        }
        self.params_MCIS = Chem.rdFMCS.MCSParameters()
        self.params_MCES = Chem.rdRascalMCES.RascalOptions()

    def test_get_smiles(self):
        reactant_smiles, product_smiles = MCSMissingGraphAnalyzer.get_smiles(
            self.reaction_dict
        )
        self.assertEqual(reactant_smiles, "CC")
        self.assertEqual(product_smiles, "CCO")

    def test_convert_smiles_to_molecule(self):
        mol = MCSMissingGraphAnalyzer.convert_smiles_to_molecule("CCO")
        self.assertIsInstance(mol, Chem.Mol)

    def test_fit_balanced(self):
        # Testing 'fit' function with balanced reaction
        (
            mcs_list,
            sorted_parents,
            reactant_mol_list,
            product_mol,
        ) = MCSMissingGraphAnalyzer.fit(self.reaction_dict)
        self.assertTrue(all(isinstance(mol, Chem.Mol) for mol in reactant_mol_list))
        self.assertIsInstance(product_mol, Chem.Mol)
        self.assertTrue(len(mcs_list) > 0)

    def test_fit_unbalanced_products(self):
        (
            mcs_list,
            sorted_parents,
            reactant_mol_list,
            product_mol,
        ) = MCSMissingGraphAnalyzer.fit(self.reaction_dict_unbalanced)

        self.assertTrue(len(mcs_list) > 0)
        self.assertEqual(Chem.MolToSmarts(mcs_list[0]), "[#6]-&!@[#6]=&!@[#8]")
        self.assertTrue(all(isinstance(mol, Chem.Mol) for mol in reactant_mol_list))
        self.assertIsInstance(product_mol, Chem.Mol)

    def test_fit_unbalanced_reactants(self):
        (
            mcs_list_unbalanced,
            sorted_parents_unbalanced,
            product_mol_list_unbalanced,
            reactant_mol_unbalanced,
        ) = MCSMissingGraphAnalyzer.fit(self.reaction_dict_unbalanced_reactants)

        self.assertTrue(
            all(isinstance(mol, Chem.Mol) for mol in product_mol_list_unbalanced)
        )
        self.assertIsInstance(reactant_mol_unbalanced, Chem.Mol)
        self.assertTrue(len(mcs_list_unbalanced) == 1)
        self.assertEqual(
            Chem.MolToSmarts(mcs_list_unbalanced[0]),
            "[#8]=&!@[#6]-&!@[#6]1:&@[#6]:&@[#6]:&@[#6]:&@[#6]:&@[#6]:&@1",
        )

    def test_IterativeMCSReactionPairs(self):
        reactant_smiles, product_smiles = MCSMissingGraphAnalyzer.get_smiles(
            self.reaction_dict_unbalanced
        )
        reactant_mols = [
            MCSMissingGraphAnalyzer.convert_smiles_to_molecule(smiles)
            for smiles in reactant_smiles.split(".")
        ]
        product_mol = MCSMissingGraphAnalyzer.convert_smiles_to_molecule(product_smiles)

        # Test with MCIS
        (
            mcs_list_MCIS,
            sorted_reactants_MCIS,
        ) = MCSMissingGraphAnalyzer.IterativeMCSReactionPairs(
            reactant_mols,
            product_mol,
            self.params_MCIS,
            method="MCIS",
            sort="MCIS",
            remove_substructure=True,
        )
        self.assertTrue(all(isinstance(mol, Chem.Mol) for mol in mcs_list_MCIS))

        # Test with MCES
        reactant_mols = [
            Chem.MolFromSmiles("CN(C)c1ccc(CC(=O)NCCCCCCCCCCNC23CC4CC(C2)CC(C3)C4)cc1")
        ]
        product_mol = Chem.MolFromSmiles(
            "CN(C)c1ccc(CC(=O)NCCCCCCCCCCCCNC23CC4CC(C2)CC(C3)C4)cc1"
        )

        (
            mcs_list_MCES,
            sorted_reactants_MCES,
        ) = MCSMissingGraphAnalyzer.IterativeMCSReactionPairs(
            reactant_mols,
            product_mol,
            self.params_MCES,
            method="MCES",
            sort="MCES",
            remove_substructure=True,
        )
        self.assertTrue(all(isinstance(mol, Chem.Mol) for mol in mcs_list_MCES))


if __name__ == "__main__":
    unittest.main()
