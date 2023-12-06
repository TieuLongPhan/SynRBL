import networkx as nx
from rdkit import Chem
from rdkit.Chem import rdFMCS
from pysmiles import read_smiles
import matplotlib.pyplot as plt

class MCSFinder:
    """
    A class to find the Maximum Common Substructure (MCS) between a reference molecule and hit molecules.
    Supports RDKit's native functionality, graph-based approach using RDKit, and graph-based approach using pysmiles.
    """

    def __init__(self, ref_smiles):
        """
        Initializes the MCSFinder with a reference molecule.
        
        Parameters:
        ref_smiles (str): SMILES string of the reference molecule.
        """
        self.ref_molecule = Chem.MolFromSmiles(ref_smiles)
        self.ref_graph = read_smiles(ref_smiles, explicit_hydrogen=False)

    def mol_to_nx(self, mol):
        """
        Convert an RDKit molecule to a NetworkX graph.
        
        Parameters:
        mol (Mol): An RDKit Mol object.

        Returns:
        Graph: A NetworkX graph.
        """
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(atom.GetIdx(), element=atom.GetSymbol())
        for bond in mol.GetBonds():
            G.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), order=bond.GetBondType())
        return G

    def find_mcs(self, hit_smiles, use_graph=False, use_pysmiles=False, ignore_atom_type=False, ignore_bond_type=False):
        """
        Find the MCS between the reference molecule and a hit molecule.
        
        Parameters:
        hit_smiles (str): SMILES string of the hit molecule.
        use_graph (bool): Whether to use the graph-based approach for MCS.
        use_pysmiles (bool): Whether to use pysmiles for graph conversion.
        ignore_atom_type (bool): Whether to ignore atom types in matching.
        ignore_bond_type (bool): Whether to ignore bond types in matching.

        Returns:
        MCSResult or Graph: RDKit MCSResult object or a NetworkX Graph depending on the approach.
        """
        hit_mol = Chem.MolFromSmiles(hit_smiles)

        if use_graph:
            ref_graph = self.ref_graph if use_pysmiles else self.mol_to_nx(self.ref_molecule)
            hit_graph = read_smiles(hit_smiles, explicit_hydrogen=False) if use_pysmiles else self.mol_to_nx(hit_mol)

            # Define node and edge match functions based on the options
            node_match_func = lambda a, b: a['element'] == b['element'] or ignore_atom_type
            edge_match_func = lambda a, b: a['order'] == b['order'] or ignore_bond_type

            GM = nx.algorithms.isomorphism.GraphMatcher(ref_graph, hit_graph, node_match=node_match_func, edge_match=edge_match_func)

            try:
                mcs = max(GM.subgraph_isomorphisms_iter(), key=len)
                mcs_graph = ref_graph.subgraph(mcs)
                return mcs_graph
            except ValueError:
                return None  # No common subgraph found
        else:
            # Use RDKit's native MCS method
            atom_compare = rdFMCS.AtomCompare.CompareAny if ignore_atom_type else rdFMCS.AtomCompare.CompareElements
            bond_compare = rdFMCS.BondCompare.CompareAny if ignore_bond_type else rdFMCS.BondCompare.CompareOrder
            mcs_result = rdFMCS.FindMCS([self.ref_molecule, hit_mol], atomCompare=atom_compare, bondCompare=bond_compare)
            return mcs_result
    
    def get_mcs_info(self, hit_smiles, use_graph=False, use_pysmiles=False, ignore_atom_type=False, ignore_bond_type=False):
        """
        Get information about the MCS as a dictionary.

        Parameters:
        hit_smiles (str): SMILES string of the hit molecule.
        use_graph (bool): Whether to use the graph-based approach for MCS.
        use_pysmiles (bool): Whether to use pysmiles for graph conversion.
        ignore_atom_type (bool): Whether to ignore atom types in matching.
        ignore_bond_type (bool): Whether to ignore bond types in matching.

        Returns:
        dict: Information about the MCS including size and SMILES representation.
        """
        mcs_result = self.find_mcs(hit_smiles, use_graph, use_pysmiles, ignore_atom_type, ignore_bond_type)
        if mcs_result is None:
            return {"size": 0, "smiles": None}

        if use_graph:
            size = mcs_result.number_of_nodes() if mcs_result else 0
            smiles = None  # Graph-based approach does not directly provide SMILES
        else:
            size = mcs_result.numAtoms if mcs_result else 0
            smiles = mcs_result.smartsString if mcs_result else None

        return {"size": size, "smiles": smiles}


    def draw_mol_graph(self, G):
        """
        Draw a molecular graph if it's not None.

        Parameters:
        G (Graph): A NetworkX graph.
        """
        if G is None:
            print("No common substructure to draw.")
            return

        pos = nx.spring_layout(G)
        labels = nx.get_node_attributes(G, 'element')
        nx.draw(G, pos, labels=labels, with_labels=True, node_size=700, node_color='lightblue')
        plt.show()