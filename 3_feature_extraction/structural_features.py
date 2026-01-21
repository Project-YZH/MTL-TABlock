"""
MTL-TABlock:  Structural Feature Extraction Module

It extracts structural features from the function-level behavior graph. 

"""

import networkx as nx
import numpy as np
from typing import Dict, List, Any, Set, Optional, Tuple


class StructuralFeatureExtractor:
    """
    Extracts structural features from the function-level behavior graph.
    
    This class analyzes the graph topology to compute features that describe
    how function nodes relate to each other and to other node types (Network,
    Storage, Script, WebAPI).
    """
    
    def __init__(self, graph:  nx.DiGraph, node_metadata: Dict[str, Dict[str, Any]]):
        """
        Initialize the structural feature extractor.
        
        Args: 
            graph: NetworkX directed graph representation of website behavior
            node_metadata: Dictionary mapping node IDs to their metadata including:
                - node_type: Type of node (ScriptMethod, Script, Network, Storage, WebAPI)
                - script_url: URL of the parent script (for ScriptMethod nodes)
                - method_name: Name of the function (for ScriptMethod nodes)
                - tracking_count: Number of tracking requests initiated
                - non_tracking_count:  Number of non-tracking requests initiated
        """
        self. graph = graph
        self.node_metadata = node_metadata
        
        # Pre-compute node type sets for efficient lookup
        self._script_method_ids:  Set[str] = set()
        self._script_ids: Set[str] = set()
        self._network_ids: Set[str] = set()
        self._storage_ids: Set[str] = set()
        self._webapi_ids: Set[str] = set()
        self._eval_or_external_function_ids: Set[str] = set()
        self._fingerprinting_ids: Set[str] = set()
        
        self._classify_nodes()
        
        # Pre-compute graph-level metrics
        self._num_nodes = graph. number_of_nodes()
        self._num_edges = graph.number_of_edges()
        self._nodes_div_by_edges = self._num_nodes / self._num_edges if self._num_edges > 0 else 0.0
        self._edges_div_by_nodes = self._num_edges / self._num_nodes if self._num_nodes > 0 else 0.0
        
        # Pre-compute centrality metrics (expensive, so cache them)
        self._in_degree_centrality:  Dict[str, float] = {}
        self._out_degree_centrality: Dict[str, float] = {}
        self._closeness_centrality: Dict[str, float] = {}
        self._precompute_centrality_metrics()
    
    def _classify_nodes(self) -> None:
        """Classify all nodes by their types for efficient lookup."""
        for node_id, metadata in self.node_metadata.items():
            node_type = metadata.get('node_type', '')
            
            if node_type == 'ScriptMethod': 
                # Filter out invalid method nodes
                script_url = metadata.get('script_url', '')
                method_name = metadata.get('method_name', '')
                
                if not self._is_valid_method_node(script_url, method_name, node_id):
                    continue
                    
                self._script_method_ids. add(str(node_id))
                
                # Check if it's an eval or external function (no script URL)
                if script_url == '': 
                    self._eval_or_external_function_ids. add(str(node_id))
                    
            elif node_type == 'Script': 
                self._script_ids.add(str(node_id))
                # Check if script has no method name (eval or external)
                if metadata.get('method_name', '') == '':
                    self._eval_or_external_function_ids.add(str(node_id))
                    
            elif node_type == 'Network':
                if 'chrome-extension' not in str(metadata.get('url', '')):
                    self._network_ids.add(str(node_id))
                    
            elif node_type == 'Storage': 
                self._storage_ids.add(str(node_id))
                
            elif node_type == 'WebAPI': 
                self._webapi_ids.add(str(node_id))
    
    def _is_valid_method_node(self, script_url: str, method_name: str, node_id:  str) -> bool:
        """Check if a method node is valid for feature extraction."""
        # Filter out empty nodes and chrome extension nodes
        if str(node_id) == '' or method_name == '': 
            return False
        if 'chrome-extension' in script_url:
            return False
        return True
    
    def _precompute_centrality_metrics(self) -> None:
        """Pre-compute centrality metrics for all nodes."""
        try:
            self._in_degree_centrality = nx. in_degree_centrality(self.graph)
        except Exception:
            self._in_degree_centrality = {}
            
        try: 
            self._out_degree_centrality = nx.out_degree_centrality(self.graph)
        except Exception:
            self._out_degree_centrality = {}
    
    def set_fingerprinting_ids(self, fingerprinting_ids:  Set[str]) -> None:
        """
        Set the IDs of nodes that have fingerprinting-related event listeners.
        
        Args:
            fingerprinting_ids:  Set of node IDs with fingerprinting behaviors
        """
        self._fingerprinting_ids = fingerprinting_ids
    
    def extract_features(self, node_id: str) -> Dict[str, Any]: 
        """
        Extract all structural features for a given function node.
        
        Args:
            node_id: The node ID of the function to extract features for
            
        Returns:
            Dictionary containing all structural features
        """
        node_id_str = str(node_id)
        
        features = {
            # Graph-level features
            'num_nodes': self._num_nodes,
            'num_edges':  self._num_edges,
            'nodes_div_by_edges': self._nodes_div_by_edges,
            'edges_div_by_nodes': self._edges_div_by_nodes,
            
            # Node degree features
            'in_degree':  0,
            'out_degree': 0,
            'in_out_degree':  0,
            
            # Ancestor/Descendant features
            'ancestor_count': 0,
            'descendant_count': 0,
            
            # Centrality features
            'closeness_centrality':  0.0,
            'in_degree_centrality': 0.0,
            'out_degree_centrality': 0.0,
            
            # Binary flags for special relationships
            'is_anonymous': 0,
            'is_eval_or_external_function': 0,
            'descendant_of_eval_or_function':  0,
            'ascendant_script_has_eval_or_function':  0,
            
            # Successor/Predecessor counts by type
            'num_script_successors': 0,
            'num_script_predecessors':  0,
            'num_method_successors': 0,
            'num_method_predecessors': 0,
            
            # Storage relationship features
            'descendant_of_storage_node': 0,
            'ascendant_of_storage_node': 0,
            
            # Network relationship features
            'is_initiator': 0,  # Has immediate network child
            'immediate_method': 0,  # Number of immediate method children
            
            # Fingerprinting relationship features
            'descendant_of_fingerprinting': 0,
            'ascendant_of_fingerprinting': 0,
        }
        
        # Check if node exists in graph
        if node_id_str not in self.graph:
            return features
        
        # Extract degree features
        features['in_degree'] = self. graph.in_degree(node_id_str)
        features['out_degree'] = self.graph. out_degree(node_id_str)
        features['in_out_degree'] = features['in_degree'] + features['out_degree']
        
        # Extract ancestor/descendant counts
        try:
            ancestors = nx.ancestors(self. graph, node_id_str)
            descendants = nx.descendants(self.graph, node_id_str)
            features['ancestor_count'] = len(ancestors)
            features['descendant_count'] = len(descendants)
        except Exception: 
            ancestors = set()
            descendants = set()
        
        # Extract centrality features
        try:
            features['closeness_centrality'] = nx.closeness_centrality(self.graph, node_id_str)
        except Exception:
            features['closeness_centrality'] = 0.0
            
        features['in_degree_centrality'] = self._in_degree_centrality.get(node_id_str, 0.0)
        features['out_degree_centrality'] = self._out_degree_centrality.get(node_id_str, 0.0)
        
        # Check if node is anonymous (no method name)
        metadata = self.node_metadata.get(node_id, self.node_metadata. get(node_id_str, {}))
        method_name = metadata. get('method_name', '')
        script_url = metadata. get('script_url', '')
        
        features['is_anonymous'] = 1 if method_name == '' else 0
        features['is_eval_or_external_function'] = 1 if script_url == '' else 0
        
        # Analyze descendants
        for desc_id in descendants: 
            desc_id_int = int(desc_id) if desc_id. isdigit() else desc_id
            
            # Check for eval/external function in descendants
            if desc_id in self._eval_or_external_function_ids:
                features['descendant_of_eval_or_function'] = 1
                
            # Count script successors
            if desc_id in self._script_ids:
                features['num_script_successors'] += 1
                
            # Count method successors
            if desc_id in self._script_method_ids: 
                features['num_method_successors'] += 1
                
            # Check for storage descendants
            if desc_id in self._storage_ids:
                features['descendant_of_storage_node'] += 1
                
            # Check for fingerprinting descendants
            if desc_id in self._fingerprinting_ids:
                features['descendant_of_fingerprinting'] += 1
        
        # Analyze ancestors
        for anc_id in ancestors:
            anc_id_int = int(anc_id) if anc_id.isdigit() else anc_id
            
            # Check for eval/external function in ancestors
            if anc_id in self._eval_or_external_function_ids:
                features['ascendant_script_has_eval_or_function'] = 1
                
            # Count script predecessors
            if anc_id in self._script_ids:
                features['num_script_predecessors'] += 1
                
            # Count method predecessors
            if anc_id in self._script_method_ids: 
                features['num_method_predecessors'] += 1
                
            # Check for storage ancestors
            if anc_id in self._storage_ids:
                features['ascendant_of_storage_node'] += 1
                
            # Check for fingerprinting ancestors
            if anc_id in self._fingerprinting_ids: 
                features['ascendant_of_fingerprinting'] += 1
        
        # Analyze immediate successors
        try:
            immediate_children = list(self.graph. successors(node_id_str))
            for child_id in immediate_children:
                # Check if immediate child is a network node (is_initiator)
                if child_id in self._network_ids:
                    features['is_initiator'] = 1
                    
                # Count immediate method children
                if child_id in self._script_method_ids: 
                    features['immediate_method'] += 1
        except Exception:
            pass
        
        return features
    
    def batch_extract_features(
        self,
        function_node_ids: List[str]
    ) -> Dict[str, Dict[str, Any]]: 
        """
        Extract structural features for multiple function nodes efficiently.
        
        Args:
            function_node_ids:  List of function node IDs
            
        Returns: 
            Dictionary mapping node IDs to their structural features
        """
        results = {}
        for node_id in function_node_ids: 
            results[node_id] = self.extract_features(node_id)
        return results
    
    def get_graph_level_features(self) -> Dict[str, Any]:
        """
        Get graph-level structural features that apply to all nodes. 
        
        Returns: 
            Dictionary of graph-level features
        """
        return {
            'num_nodes': self._num_nodes,
            'num_edges':  self._num_edges,
            'nodes_div_by_edges': self._nodes_div_by_edges,
            'edges_div_by_nodes': self._edges_div_by_nodes,
        }


def extract_structural_features(graph: nx.DiGraph, node_id: str) -> Dict[str, Any]: 
    """
    Extract structural features for a given function node. 
    Standalone function for backward compatibility.
    
    Args:
        graph: NetworkX directed graph representation of the website behavior
        node_id: The node ID of the function to extract features for
        
    Returns: 
        Dictionary containing structural features
    """
    # Create a basic metadata dict from graph node attributes
    node_metadata = {}
    for n in graph.nodes():
        node_data = graph.nodes[n]
        node_metadata[n] = {
            'node_type': node_data.get('type', 'ScriptMethod'),
            'script_url':  node_data.get('script_url', ''),
            'method_name': node_data.get('method_name', ''),
        }
    
    extractor = StructuralFeatureExtractor(graph, node_metadata)
    return extractor.extract_features(node_id)


def batch_extract_structural_features(
    graph: nx.DiGraph, 
    function_nodes: List[str]
) -> Dict[str, Dict[str, Any]]: 
    """
    Extract structural features for all function nodes in batch.
    Standalone function for backward compatibility.
    
    Args:
        graph: NetworkX directed graph
        function_nodes:  List of function node IDs
        
    Returns:
        Dictionary mapping node IDs to their structural features
    """
    # Create a basic metadata dict from graph node attributes
    node_metadata = {}
    for n in graph.nodes():
        node_data = graph.nodes[n]
        node_metadata[n] = {
            'node_type': node_data.get('type', 'ScriptMethod'),
            'script_url':  node_data.get('script_url', ''),
            'method_name': node_data.get('method_name', ''),
        }
    
    extractor = StructuralFeatureExtractor(graph, node_metadata)
    return extractor.batch_extract_features(function_nodes)


def compute_eccentricity(graph: nx.DiGraph, node_id: str) -> int:
    """
    Compute the eccentricity of a node in the graph.
    
    Eccentricity is the maximum distance from the node to any other node.
    
    Args:
        graph: NetworkX directed graph
        node_id:  The node ID to compute eccentricity for
        
    Returns:
        Eccentricity value (or 0 if not computable)
    """
    try:
        # For directed graphs, we consider the underlying undirected graph
        undirected = graph.to_undirected()
        if nx.is_connected(undirected):
            return nx. eccentricity(undirected, node_id)
        else:
            # For disconnected graphs, compute within component
            for component in nx.connected_components(undirected):
                if node_id in component:
                    subgraph = undirected.subgraph(component)
                    return nx.eccentricity(subgraph, node_id)
            return 0
    except Exception:
        return 0


def compute_average_degree_connectivity(graph: nx.DiGraph, node_id: str) -> float:
    """
    Compute the average degree connectivity for neighbors of a node. 
    
    Args: 
        graph: NetworkX directed graph
        node_id: The node ID
        
    Returns: 
        Average degree of neighbors
    """
    try:
        neighbors = list(graph. predecessors(node_id)) + list(graph. successors(node_id))
        if not neighbors:
            return 0.0
        
        degrees = [graph.degree(n) for n in neighbors]
        return np.mean(degrees)
    except Exception: 
        return 0.0