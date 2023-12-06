from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys, rdFingerprintGenerator
from rdkit.Avalon import pyAvalonTools as Avalon
from rdkit.DataStructs import cDataStructs
from map4 import MAP4Calculator
import numpy as np

class FormulaSimilarityFinder:
    """
    A class to calculate the similarity between a reference molecule and a list of hit molecules using various fingerprint methods.
    It supports 'hard' and 'soft' ensemble voting strategies to aggregate similarity scores from multiple fingerprint types.

    Attributes
    ----------
    ref_molecule_smiles : str
        SMILES representation of the reference molecule.
    hit_molecules : list of dicts
        List of dictionaries containing 'formula' and 'smiles' for hit molecules.

    Methods
    -------
    calculate_fingerprint(mol, fingerprint_type)
        Calculates the specified type of fingerprint for a given molecule.
    calculate_similarity(fingerprint_type)
        Calculates the similarity scores for all hit molecules using a specified fingerprint type.
    ensemble_similarity(fingerprint_types, voting='hard')
        Calculates ensemble similarity scores using multiple fingerprint types.
    get_most_similar_molecule(fingerprint_types, voting='hard')
        Returns the most similar molecule based on the calculated ensemble similarity scores.

    Example
    -------
    >>> pen = {'formula': 'C5H12O', 'smiles': 'CCCCCO'}
    >>> spen = {'formula': 'C5H12O', 'smiles': 'CCCC(O)C'}
    >>> ref_smiles = 'CC(C)CCOC(C)=O'
    >>> hit_molecules = [pen, spen]
    >>> fingerprint_types = ['ecfp4', 'rdk5', 'maccs', 'map4', 'avalon']
    >>> calculator = SimilarityCalculator(ref_smiles, hit_molecules)
    >>> most_similar_molecule = calculator.get_most_similar_molecule(fingerprint_types, 'hard')
    >>> print(most_similar_molecule)
    """

    def __init__(self, ref_molecule_smiles, hit_molecules):
        """
        Initializes the SimilarityCalculator with a reference molecule and a list of hit molecules.

        Parameters
        ----------
        ref_molecule_smiles : str
            SMILES representation of the reference molecule.
        hit_molecules : list of dicts
            List of dictionaries containing 'formula' and 'smiles' for hit molecules.
        """
        self.ref_molecule_smiles = ref_molecule_smiles
        self.hit_molecules = hit_molecules
        self.ref_molecule = Chem.MolFromSmiles(ref_molecule_smiles)
        self.map4_calculator = MAP4Calculator(is_folded=True)

    def convert_arr2vec(self, arr):
        """
        Converts a numpy array to a RDKit ExplicitBitVect.

        Parameters
        ----------
        arr : numpy.ndarray
            A numpy array representing the binary fingerprint.

        Returns
        -------
        ExplicitBitVect
            A RDKit ExplicitBitVect object representing the fingerprint.
        """
        arr_tostring = "".join(arr.astype(str))
        EBitVect2 = cDataStructs.CreateFromBitString(arr_tostring)
        return EBitVect2

    def calculate_map4(self, mol):
        """
        Calculates the MAP4 fingerprint for a given molecule.

        Parameters
        ----------
        mol : Mol
            A RDKit Mol object.

        Returns
        -------
        ExplicitBitVect
            The MAP4 fingerprint as a RDKit ExplicitBitVect object.
        """
        map4 = self.map4_calculator.calculate(mol)
        return self.convert_arr2vec(map4)

    def calculate_fingerprint(self, mol, fingerprint_type):
        """
        Calculates the fingerprint of the specified type for a given molecule.

        Parameters
        ----------
        mol : Mol
            A RDKit Mol object.
        fingerprint_type : str
            The type of fingerprint to calculate ('ecfp4', 'rdk5', 'maccs', 'map4', 'avalon').

        Returns
        -------
        ExplicitBitVect
            The calculated fingerprint as a RDKit ExplicitBitVect object.

        Raises
        ------
        ValueError
            If an invalid fingerprint type is specified.
        """
        if fingerprint_type == 'ecfp4':
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        elif fingerprint_type == 'rdk5':
            return Chem.RDKFingerprint(mol, maxPath=5, fpSize=2048, nBitsPerHash=2)
        elif fingerprint_type == 'maccs':
            return MACCSkeys.GenMACCSKeys(mol)
        elif fingerprint_type == 'map4':
            return self.calculate_map4(mol)
        elif fingerprint_type == 'avalon':
            return Avalon.GetAvalonFP(mol, 1024)
        else:
            raise ValueError("Invalid fingerprint_type.")

    def calculate_similarity(self, fingerprint_type):
        """
        Calculates similarity scores for all hit molecules using a specified fingerprint type.

        Parameters
        ----------
        fingerprint_type : str
            The type of fingerprint to use for similarity calculation.

        Returns
        -------
        list of float
            A list of similarity scores between the reference molecule and each hit molecule.
        """
        ref_fp = self.calculate_fingerprint(self.ref_molecule, fingerprint_type)
        hit_fps = [self.calculate_fingerprint(Chem.MolFromSmiles(mol['smiles']), fingerprint_type) for mol in self.hit_molecules]

        similarities = [DataStructs.TanimotoSimilarity(ref_fp, fp) for fp in hit_fps]
        return similarities

    def ensemble_similarity(self, fingerprint_types, voting='hard'):
        """
        Calculates the ensemble similarity scores using multiple fingerprint types.

        Parameters
        ----------
        fingerprint_types : list of str
            A list of fingerprint types to use for similarity calculation.
        voting : str, optional
            The voting strategy to use ('soft' or 'hard'). Default is 'hard'.

        Returns
        -------
        list of float
            Ensemble similarity scores for each hit molecule.

        Raises
        ------
        ValueError
            If an invalid voting strategy is specified or if fingerprint_types is not a list.
        """
        if not isinstance(fingerprint_types, list):
            raise ValueError("fingerprint_types must be a list of fingerprint names.")

        total_similarities = [0] * len(self.hit_molecules)

        for fp_type in fingerprint_types:
            similarities = self.calculate_similarity(fp_type)
            if voting == 'hard':
                total_similarities = [sum(x) for x in zip(total_similarities, similarities)]
            elif voting == 'soft':
                total_similarities = [sum(x) for x in zip(total_similarities, [s/len(fingerprint_types) for s in similarities])]
            else:
                raise ValueError("Invalid voting strategy. Supported values: 'soft', 'hard'.")

        return total_similarities

    def get_most_similar_molecule(self, fingerprint_types, voting='hard'):
        """
        Returns the most similar molecule based on the calculated ensemble similarity scores.

        Parameters
        ----------
        fingerprint_types : list of str
            A list of fingerprint types to use for calculating ensemble similarity.
        voting : str, optional
            The voting strategy to use ('soft' or 'hard'). Default is 'hard'.

        Returns
        -------
        dict
            The most similar molecule, represented as a dictionary containing 'formula' and 'smiles'.
        """
        ensemble_similarities = self.ensemble_similarity(fingerprint_types, voting)
        highest_similarity_index = np.argmax(ensemble_similarities)
        return self.hit_molecules[highest_similarity_index]