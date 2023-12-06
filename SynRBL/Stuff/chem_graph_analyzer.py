import networkx as nx
from pysmiles import read_smiles
import matplotlib.pyplot as plt
from rdkit.Chem import rdmolops
from rdkit import Chem

class ChemicalGraphAnalyzer:
    """
    A class for analyzing chemical graphs based on SMILES strings.

    Provides functionality to convert SMILES strings to graphs, find the Maximum Common Subgraph (MCS) between graphs,
    subtract one graph from another, identify boundary nodes in a graph, display graphs using matplotlib, and convert
    graphs to molecule representations using RDKit.

    Methods
    -------
    convert_smiles_to_graph(smiles)
        Converts a SMILES string to a NetworkX graph representation of the chemical compound.
        Args:
            smiles (str): A SMILES string representing a chemical compound.
        Returns:
            networkx.Graph: A graph representation of the given SMILES string.

    find_maximum_common_subgraph(graph1, graph2, ignore_atom_type=False, ignore_bond_type=True)
        Finds the Maximum Common Subgraph (MCS) between two graphs.
        Args:
            graph1, graph2 (networkx.Graph): Graphs to compare.
            ignore_atom_type (bool): If True, ignores atom types in comparison.
            ignore_bond_type (bool): If True, ignores bond types in comparison.
        Returns:
            networkx.Graph: The MCS of the two graphs, or None if no MCS found.

    subtract_graphs(main_graph, sub_graph)
        Subtracts one graph from another.
        Args:
            main_graph (networkx.Graph): The graph from which to subtract.
            sub_graph (networkx.Graph): The graph to be subtracted.
        Returns:
            networkx.Graph: The resulting graph after subtraction.

    identify_boundary_nodes(mcs_graph, main_graph)
        Identifies boundary nodes in the MCS that have edges to nodes not in MCS.
        Args:
            mcs_graph (networkx.Graph): The Maximum Common Subgraph.
            main_graph (networkx.Graph): The main graph to compare with.
        Returns:
            set: A set of boundary nodes.

    display_graph(graph, title='Graph', highlight_nodes=None)
        Displays a graph using matplotlib.
        Args:
            graph (networkx.Graph): The graph to be displayed.
            title (str): Title of the graph.
            highlight_nodes (set): Nodes to be highlighted.

    graph_to_molecule(graph, return_smiles=True, return_mol=False)
        Converts a NetworkX graph to a molecule using RDKit.
        Args:
            graph (networkx.Graph): A graph representation of a molecule.
        Returns:
            str: The SMILES representation of the molecule, or RDKit molecule object, or both.

    analyze_reactant_product(reactant_smiles, product_smiles, display=True)
        Analyzes the reactant and product of a chemical reaction represented by SMILES strings.
        Args:
            reactant_smiles, product_smiles (str): SMILES strings of the reactant and product.
            display (bool): If True, displays the graphs.
        Returns:
            tuple: A tuple containing the MCS graph, the missing graph, and the set of boundary nodes.

    Examples
    --------
    Example usage of ChemicalGraphAnalyzer:

    >>> analyzer = ChemicalGraphAnalyzer()
    >>> reactant_smiles = 'C1=CC=CC=C1'
    >>> product_smiles = 'C1=CC=CC=C1O'
    >>> mcs_graph, missing_graph, boundary_nodes = analyzer.analyze_reactant_product(reactant_smiles, product_smiles, display=True)
    >>> print("Boundary nodes:", boundary_nodes)
    """

    @staticmethod
    def convert_smiles_to_graph(smiles):
        """
        Converts a SMILES string to a NetworkX graph.

        Args:
            smiles (str): A SMILES string representing a chemical compound.

        Returns:
            networkx.Graph: A graph representation of the given SMILES string.
        """
        return read_smiles(Chem.CanonSmiles(smiles), explicit_hydrogen=False, reinterpret_aromatic=False)

    @staticmethod
    def find_maximum_common_subgraph(graph1, graph2, ignore_atom_type=False, ignore_bond_type=True):
        """
        Finds the Maximum Common Subgraph (MCS) between two graphs.

        Args:
            graph1, graph2 (networkx.Graph): Graphs to compare.
            ignore_atom_type (bool): If True, ignores atom types in comparison.
            ignore_bond_type (bool): If True, ignores bond types in comparison.

        Returns:
            networkx.Graph: The MCS of the two graphs, or None if no MCS found.
        """
        # Custom matching functions for nodes and edges
        node_match_func = lambda a, b: ((a['element'] == b['element']) and (a['charge'] == b['charge']) and (a['aromatic'] == b['aromatic'])) 
        edge_match_func = lambda a, b: (a['order'] == b['order']) 
        

        if ignore_bond_type and not ignore_atom_type:
            # Only node matching criteria are considered.
            graph_matcher = nx.algorithms.isomorphism.GraphMatcher(graph1, graph2, node_match=node_match_func)
        elif not ignore_bond_type and ignore_atom_type:
            # Only edge matching criteria are considered.
            graph_matcher = nx.algorithms.isomorphism.GraphMatcher(graph1, graph2, edge_match=edge_match_func)
        elif ignore_bond_type and ignore_atom_type:
            # Neither node nor edge matching criteria are considered.
            graph_matcher = nx.algorithms.isomorphism.GraphMatcher(graph1, graph2)
        else:
            # Both node and edge matching criteria are considered.
            graph_matcher = nx.algorithms.isomorphism.GraphMatcher(graph1, graph2, node_match=node_match_func, edge_match=edge_match_func)


        try:
            mcs = max(graph_matcher.subgraph_isomorphisms_iter(), key=len)
            return graph1.subgraph(mcs)
        except ValueError:
            return None

    @staticmethod
    def subtract_graphs(main_graph, sub_graph):
        """
        Subtracts one graph from another.

        Args:
            main_graph (networkx.Graph): The graph from which to subtract.
            sub_graph (networkx.Graph): The graph to be subtracted.

        Returns:
            networkx.Graph: The resulting graph after subtraction.
        """
        result_graph = main_graph.copy()
        for node in sub_graph.nodes():
            if node in result_graph:
                result_graph.remove_node(node)
        return result_graph

    @staticmethod
    def identify_boundary_nodes(mcs_graph, main_graph):
        """
        Identifies boundary nodes in the MCS that have edges to nodes not in MCS.

        Args:
            mcs_graph (networkx.Graph): The Maximum Common Subgraph.
            main_graph (networkx.Graph): The main graph to compare with.

        Returns:
            set: A set of boundary nodes.
        """
        boundary_nodes = set()
        for node in mcs_graph.nodes():
            for neighbor in main_graph.neighbors(node):
                if neighbor not in mcs_graph:
                    boundary_nodes.add(node)
        return boundary_nodes

    @staticmethod
    def display_graph(graph, title='Graph', highlight_nodes=None):
        """
        Displays a graph using matplotlib.

        Args:
            graph (networkx.Graph): The graph to be displayed.
            title (str): Title of the graph.
            highlight_nodes (set): Nodes to be highlighted.
        """
        positions = nx.spring_layout(graph)
        node_labels = {node: data.get('element', node) for node, data in graph.nodes(data=True)}
        node_color_map = ['lightblue' if node not in highlight_nodes else 'lightgreen' for node in graph.nodes()] if highlight_nodes else 'lightblue'
        nx.draw(graph, positions, with_labels=True, labels=node_labels, node_size=200, node_color=node_color_map, font_size=10, font_color='black', font_weight='bold')
        edge_labels = {(u, v): data.get('order', '') for u, v, data in graph.edges(data=True)}
        nx.draw_networkx_edge_labels(graph, positions, edge_labels=edge_labels, font_color='red')
        plt.title(title)
        plt.show()
    @staticmethod
    def graph_to_molecule(graph, return_smiles = True, return_mol = False ):
        """
        Converts a NetworkX graph to a molecule using RDKit.

        Args:
            graph (networkx.Graph): A graph representation of a molecule, 
                                    with nodes having 'element' attributes 
                                    and edges having 'order' attributes.

        Returns:
            str: The SMILES representation of the molecule.
        """
        mol = Chem.RWMol()

        # Map for bond types
        bond_type_map = {
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
            1.5: Chem.rdchem.BondType.AROMATIC
        }

        # Add atoms to the molecule
        node_to_idx = {}
        for node, data in graph.nodes(data=True):
            atom = Chem.Atom(data['element'])
            mol_idx = mol.AddAtom(atom)
            node_to_idx[node] = mol_idx

        # Add bonds to the molecule
        for u, v, data in graph.edges(data=True):
            bond_type = data.get('order')
            rdkit_bond_type = bond_type_map.get(bond_type)
            if rdkit_bond_type:
                mol.AddBond(node_to_idx[u], node_to_idx[v], rdkit_bond_type)

        # Sanitize the molecule
        rdmolops.SanitizeMol(mol)
        if return_mol:
            return mol
        elif return_smiles:
            # Convert to SMILES
            smiles = Chem.MolToSmiles(mol)
            return smiles
        else:
            return mol, Chem.MolToSmiles(mol)
    
    @staticmethod
    def analyze_reactant_product(reactant_smiles, product_smiles, display=True):
        """
        Analyzes the reactant and product of a chemical reaction represented by SMILES strings.

        Args:
            reactant_smiles, product_smiles (str): SMILES strings of the reactant and product.
            display (bool): If True, displays the graphs.

        Returns:
            tuple: A tuple containing the MCS graph, the missing graph, and the set of boundary nodes.
        """
        reactant_graph = ChemicalGraphAnalyzer.convert_smiles_to_graph(reactant_smiles)
        product_graph = ChemicalGraphAnalyzer.convert_smiles_to_graph(product_smiles)
        mcs_graph = ChemicalGraphAnalyzer.find_maximum_common_subgraph(reactant_graph, product_graph, ignore_atom_type=False, ignore_bond_type=True)
        if mcs_graph:

            if len(reactant_graph.nodes) >= len(product_graph.nodes):
                missing_graph = ChemicalGraphAnalyzer.subtract_graphs(reactant_graph, mcs_graph)
                main_graph = reactant_graph
            else:
                missing_graph = ChemicalGraphAnalyzer.subtract_graphs(product_graph, mcs_graph)
                main_graph = product_graph

            boundary_nodes = ChemicalGraphAnalyzer.identify_boundary_nodes(missing_graph, main_graph)

            if display:
                if mcs_graph:
                    ChemicalGraphAnalyzer.display_graph(nx.Graph(mcs_graph), 'Maximum Common Subgraph')
                else:
                    print("No Maximum Common Subgraph found.")

                if missing_graph:
                    ChemicalGraphAnalyzer.display_graph(nx.Graph(missing_graph), 'Missing Graph', highlight_nodes=boundary_nodes)
                else:
                    print("No missing part found.")

            return mcs_graph, missing_graph, boundary_nodes
        else:
            print('NO MCS')