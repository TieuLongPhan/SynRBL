from rdkit import Chem
from rdkit.Chem import Draw, rdChemReactions, MolStandardize
from rdkit.Chem.SaltRemover import SaltRemover

class SMILESStandardizer:
    """
    Extended class to standardize SMILES strings for chemical molecules and reactions.
    Provides options for normalization, tautomerization, salt removal, charge handling, stereochemistry handling, radical cleaning, and aromatization/dearomatization.
    """

    def __init__(self):
        """
        Initialize the SMILESStandardizer object with normalizer, tautomer canonicalizer, and salt remover.
        """
        self.normalizer = MolStandardize.normalize.Normalizer()
        self.tautomer = MolStandardize.tautomer.TautomerCanonicalizer()
        self.salt_remover = SaltRemover()

    def standardize_smiles(self, smiles, visualize=False, **kwargs):
        """
        Standardize a single SMILES string with an option to visualize the molecule before and after standardization.

        Parameters:
        smiles (str): Original SMILES string.
        visualize (bool): If True, visualize the molecule before and after standardization.
        **kwargs: Additional arguments for standardization options such as normalize, tautomerize, etc.

        Returns:
        str: Standardized SMILES string or an error message if sanitization fails.
        """
        original_mol = Chem.MolFromSmiles(smiles)
        if not original_mol:
            return None

        try:
            standardized_mol = self._apply_standardization(original_mol, **kwargs)
            standardized_smiles = Chem.MolToSmiles(standardized_mol, isomericSmiles=True, canonical=True)

            if visualize:
                self._visualize_molecules(original_mol, standardized_mol)

            return standardized_smiles

        except Chem.MolSanitizeException:
            return "Sanitization failed for SMILES: " + smiles

    def _apply_standardization(self, mol, normalize=True, tautomerize=True, remove_salts=False, 
                               handle_charges=False, handle_stereo=True, clean_radicals=False, 
                               dearomatize=True, aromatize=True):
        """
        Apply the standardization steps to the molecule.

        Parameters:
        mol (Mol): RDKit Mol object.
        **kwargs: Options for normalization, tautomerization, etc.

        Returns:
        Mol: Standardized RDKit Mol object.
        """
        # Apply normalization
        if normalize:
            mol = self.normalizer.normalize(mol)

        # Apply tautomerization
        if tautomerize:
            try: 
                
                fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
                tautomerized_fragments = [self.tautomer.canonicalize(frag) for frag in fragments]
                mol = Chem.MolFromSmiles('.'.join(Chem.MolToSmiles(frag, isomericSmiles=True) for frag in tautomerized_fragments))
            
            except:
                mol = mol

        # Remove salts
        if remove_salts:
            mol = self.salt_remover.StripMol(mol)

        # Handle charges
        if handle_charges:
            mol = MolStandardize.charge.Reionizer().reionize(mol)

        # Apply dearomatization
        if dearomatize:
            Chem.Kekulize(mol, clearAromaticFlags=True)

        # Apply aromatization
        if aromatize:
            Chem.SanitizeMol(mol, Chem.SanitizeFlags.SANITIZE_SETAROMATICITY)

        # Handle stereochemistry
        if handle_stereo:
            Chem.AssignStereochemistry(mol, cleanIt=True, force=True)

        # Handle radicals (custom logic required)
        # Add your custom logic here to handle radicals => working

        # Remove explicit hydrogens and sanitize molecule
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)

        return mol

    def _visualize_molecules(self, before_mol, after_mol):
        """
        Visualize the molecules before and after standardization.

        Parameters:
        before_mol (Mol): Original RDKit Mol object.
        after_mol (Mol): Standardized RDKit Mol object.
        """
        img = Draw.MolsToGridImage([before_mol, after_mol], legends=['Before', 'After'], useSVG=False)
        display(img)

    def standardize_reaction(self, reaction_smiles, **kwargs):
        """
        Standardize a reaction represented by a SMILES string.

        Parameters:
        reaction_smiles (str): Reaction SMILES string.
        **kwargs: Additional arguments for standardization options.

        Returns:
        str: Standardized reaction SMILES string or an error message if conversion fails.
        """
        rxn = rdChemReactions.ReactionFromSmarts(reaction_smiles)
        if not rxn:
            return None

        try:
            reactants = [self.standardize_smiles(Chem.MolToSmiles(reactant), **kwargs) for reactant in rxn.GetReactants()]
            products = [self.standardize_smiles(Chem.MolToSmiles(product), **kwargs) for product in rxn.GetProducts()]
            return '>>'.join(['.'.join(reactants), '.'.join(products)])
        except:
            return "Standardization failed for reaction SMILES: " + reaction_smiles
