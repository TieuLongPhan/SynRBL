from rdkit import Chem
from rdkit.Chem import Draw, MolStandardize
from rdkit.Chem.SaltRemover import SaltRemover
from joblib import Parallel, delayed

class SMILESStandardizer:
    """
    A class to standardize SMILES strings for chemical molecules and reactions.

    Parameters
    ----------
    None

    Attributes
    ----------
    normalizer : MolStandardize.normalize.Normalizer
        Normalization object.
    tautomer : MolStandardize.tautomer.TautomerCanonicalizer
        Tautomer canonicalization object.
    salt_remover : SaltRemover
        Salt removal object.

    Methods
    -------
    standardize_smiles(smiles, visualize=False, **kwargs)
        Standardize a single SMILES string with an option to visualize the molecule before and after standardization.

    _apply_standardization(mol, normalize=True, tautomerize=True, remove_salts=False,
                           handle_charges=False, handle_stereo=True, clean_radicals=False,
                           dearomatize=False, aromatize=False)
        Apply the standardization steps to the molecule.

    _visualize_molecules(before_mol, after_mol)
        Visualize the molecules before and after standardization.

    standardize_dict_smiles(data_input, key='reactants', visualize=False, parallel=True, n_jobs=4, **kwargs)
        Process a list of reaction data and standardize the SMILES strings based on the specified key.

    Examples
    --------
    Example usage of SMILESStandardizer:

    >>> from rdkit import Chem
    >>> from rdkit.Chem import Draw, MolStandardize
    >>> from rdkit.Chem.SaltRemover import SaltRemover
    >>> from joblib import Parallel, delayed
    >>> smiles = 'C1=CC=CC=C1'
    >>> standardizer = SMILESStandardizer()
    >>> standardized_smiles = standardizer.standardize_smiles(smiles)
    >>> print(standardized_smiles)
    """
    
    def __init__(self):
        """
        Initialize the SMILESStandardizer object with normalizer, tautomer canonicalizer, and salt remover.
        """
        self.normalizer = MolStandardize.normalize.Normalizer()
        self.tautomer = MolStandardize.tautomer.TautomerCanonicalizer()
        self.salt_remover = SaltRemover()

    @staticmethod
    def standardize_smiles(smiles,normalizer, tautomer, salt_remover, visualize=False, **kwargs):
        """
        Standardize a single SMILES string with an option to visualize the molecule before and after standardization.

        Parameters
        ----------
        smiles : str
            Original SMILES string.
        visualize : bool, optional
            If True, visualize the molecule before and after standardization.
        **kwargs
            Additional arguments for standardization options such as normalize, tautomerize, etc.

        Returns
        -------
        str
            Standardized SMILES string or an error message if sanitization fails.
        """
        original_mol = Chem.MolFromSmiles(smiles)

        if not original_mol:
            return None

        try:
            standardized_mol = SMILESStandardizer._apply_standardization(original_mol,normalizer, tautomer, salt_remover, **kwargs)
            standardized_smiles = Chem.MolToSmiles(standardized_mol, isomericSmiles=True, canonical=True)

            if visualize:
                SMILESStandardizer._visualize_molecules(original_mol, standardized_mol)

            return standardized_smiles

        except Chem.MolSanitizeException:
            return "Sanitization failed for SMILES: " + smiles

    @staticmethod
    def _apply_standardization(mol, normalizer, tautomer, salt_remover,
                               normalize=True, tautomerize=True, remove_salts=False,
                               handle_charges=False, handle_stereo=True, clean_radicals=False,
                               dearomatize=False, aromatize=False):
        """
        Apply the standardization steps to the molecule.

        Parameters
        ----------
        mol : Mol
            RDKit Mol object.
        normalize : bool, optional
            Perform normalization.
        tautomerize : bool, optional
            Perform tautomerization.
        remove_salts : bool, optional
            Remove salts.
        handle_charges : bool, optional
            Handle charges.
        handle_stereo : bool, optional
            Handle stereochemistry.
        clean_radicals : bool, optional
            Clean radicals (custom logic required).
        dearomatize : bool, optional
            Apply dearomatization.
        aromatize : bool, optional
            Apply aromatization.

        Returns
        -------
        Mol
            Standardized RDKit Mol object.
        """

        #  Check and normalize the molecule
        if normalize:
            mol = normalizer.normalize(mol)

        # Apply tautomerization
        if tautomerize:
            fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            if fragments:  # Check if fragments list is not empty
                try:
                    tautomerized_fragments = [tautomer.canonicalize(frag) for frag in fragments]
                    mol = Chem.MolFromSmiles('.'.join(Chem.MolToSmiles(frag, isomericSmiles=True) for frag in tautomerized_fragments))
                except:
                    mol = mol
            else:
                mol = mol  # Handle the case where fragments list is empty
        
        # Remove salts
        if remove_salts:
            mol = salt_remover.StripMol(mol)

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

    @staticmethod
    def _visualize_molecules(before_mol, after_mol):
        """
        Visualize the molecules before and after standardization.

        Parameters
        ----------
        before_mol : Mol
            Original RDKit Mol object.
        after_mol : Mol
            Standardized RDKit Mol object.
        """
        img = Draw.MolsToGridImage([before_mol, after_mol], legends=['Before', 'After'], useSVG=False)
        display(img)

    def standardize_dict_smiles(self, data_input, key=['reactants', 'products'], visualize=False, parallel=True, n_jobs=4, **kwargs):
        """
        Process a list of reaction data and standardize the SMILES strings based on the specified key.

        Parameters
        ----------
        data_input : list of dict
            List containing reaction data.
        key : str, optional
            The key from which to extract SMILES strings ('reactants' or 'products').
        visualize : bool, optional
            If True, visualize the molecules before and after standardization.
        parallel : bool, optional
            If True, run in parallel.
        n_jobs : int, optional
            Number of jobs to run in parallel.
        **kwargs
            Additional arguments for standardization options.

        Returns
        -------
        list of dict
            Processed list with standardized SMILES strings.
        """
        if parallel:
            # Correctly use Parallel with delayed

            # reactants
            standardized_smiles = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(self.standardize_smiles)(
                    reaction_data.get(key[0], ''), visualize=visualize, **kwargs
                ) for reaction_data in data_input
            )

            # Update data_input with standardized SMILES strings
            for i, reaction_data in enumerate(data_input):
                reaction_data['standardized_' + key[0]] = standardized_smiles[i]

             # products
            standardized_smiles = Parallel(n_jobs=n_jobs, verbose=1)(
                delayed(self.standardize_smiles)(
                    reaction_data.get(key[1], ''), visualize=visualize, **kwargs
                ) for reaction_data in data_input
            )

            # Update data_input with standardized SMILES strings
            for i, reaction_data in enumerate(data_input):
                reaction_data['standardized_' + key[1]] = standardized_smiles[i]


        else:
            for reaction_data in data_input:
                try:
                    for i in key:
                        smiles_string = reaction_data.get(i, '')
                        standardized_smiles = SMILESStandardizer.standardize_smiles(smiles_string, visualize=visualize)
                        reaction_data['standardized_' + i] = standardized_smiles
                        
                except Exception as e:
                    reaction_data['error'] = str(e)
        return data_input
