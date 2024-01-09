from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

class MolStandardizer:
    def __init__(self):
        pass

    @staticmethod
    def functional_group_check(mol):
        # add your functional group check here
        if FunctionalGroupChecker.check_vicinal_diol(mol):
            return 'vicinal_diol'
        elif FunctionalGroupChecker.check_hemiacetal(mol):
            return 'hemiacetal'
        elif FunctionalGroupChecker.check_carbonate(mol):
            return 'carbonate'
        else:
            return None

    @staticmethod
    def standardize_mol(mol):
        try:
            # Standardize tautomers
            enumerator = rdMolStandardize.TautomerEnumerator()
            standardized_mol = enumerator.Canonicalize(mol)

            # Convert to SMILES for further processing
            smiles = Chem.MolToSmiles(standardized_mol)
            functional_group = UnstableMol.functional_group_check(Chem.MolFromSmiles(smiles))

            # Apply specific transformations based on functional groups
            if functional_group == 'vicinal_diol':
                smiles = smiles.replace('OCO', 'C=O')
            elif functional_group == 'hemiacetal':
                smiles = smiles.replace('COCO', 'CO.C=O')
            elif functional_group == 'carbonate':
                smiles = smiles.replace('COC(=O)O', 'CO.OC(=O)O').replace('OC(=O)OC', 'COC(=O)O.OC')

            # Convert back to RDKit Mol object
            return Chem.MolFromSmiles(smiles)
        except Exception as e:
            #print(f"An error occurred: {e}")
            return mol


