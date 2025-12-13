"""
Graph Builder for Shell Scripts
Converts AST to NetworkX graph for GNN processing
"""

import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.ast_parser import ShellScriptParser


class ScriptGraphBuilder:
    """Convert shell scripts to graph representation"""

    def __init__(self):
        self.parser = ShellScriptParser()

        # Command categories (for one-hot encoding)
        self.command_categories = {
            'network': ['curl', 'wget', 'nc', 'netcat', 'telnet', 'ssh', 'scp', 'ftp'],
            'execution': ['bash', 'sh', 'eval', 'exec', 'source', '.'],
            'file_ops': ['chmod', 'chown', 'rm', 'dd', 'mv', 'cp', 'shred'],
            'system': ['systemctl', 'service', 'crontab', 'kill', 'pkill'],
            'user': ['useradd', 'usermod', 'passwd', 'su', 'sudo'],
            'process': ['ps', 'top', 'nohup', 'bg', 'fg', 'jobs', 'setsid'],
            'info': ['whoami', 'id', 'uname', 'hostname', 'ifconfig', 'ip'],
            'text': ['echo', 'cat', 'grep', 'awk', 'sed', 'head', 'tail']
        }

    def build_graph(self, script_content: str) -> Tuple[nx.DiGraph, Dict]:
        """
        Build graph from shell script

        Args:
            script_content: Shell script as string

        Returns:
            (graph, metadata) tuple
        """
        # Parse script to AST
        ast_nodes = self.parser.parse(script_content)

        if not ast_nodes:
            # Return empty graph if parsing fails
            return self._create_empty_graph(), {'error': 'parsing_failed'}

        # Create directed graph
        G = nx.DiGraph()

        # Add nodes with features
        for idx, node in enumerate(ast_nodes):
            node_features = self._extract_node_features(node, idx)
            G.add_node(idx, **node_features)

        # Add edges (control flow)
        self._add_edges(G, ast_nodes)

        # Calculate graph-level features
        metadata = self._extract_graph_metadata(G, ast_nodes)

        return G, metadata

    def _extract_node_features(self, node: Dict, node_id: int) -> Dict:
        """Extract features for a single node"""
        features = {
            'id': node_id,
            'type': node.get('type', 'unknown'),
            'risk': node.get('risk', 0.3),
            'depth': node.get('depth', 0)
        }

        # For command nodes, add more features
        if node.get('type') == 'command':
            command = node.get('command', '')
            features['command'] = command
            features['category'] = self._get_command_category(command)
            features['arg_count'] = len(node.get('args', []))

        # For control flow nodes
        elif node.get('type') in ['conditional', 'loop']:
            features['control_type'] = node.get('kind', 'unknown')

        return features

    def _get_command_category(self, command: str) -> str:
        """Get category for a command"""
        for category, commands in self.command_categories.items():
            if command in commands:
                return category
        return 'other'

    def _add_edges(self, G: nx.DiGraph, ast_nodes: List[Dict]):
        """Add edges representing control flow"""
        # Sequential execution edges
        for i in range(len(ast_nodes) - 1):
            current = ast_nodes[i]
            next_node = ast_nodes[i + 1]

            # Add edge from current to next (sequential flow)
            G.add_edge(i, i + 1, edge_type='sequential')

            # Add additional edges for control structures
            if current.get('type') in ['conditional', 'loop']:
                # Control structure can jump
                G.add_edge(i, i + 1, edge_type='control_flow')

                # Find nested commands (based on depth)
                current_depth = current.get('depth', 0)
                for j in range(i + 1, len(ast_nodes)):
                    if ast_nodes[j].get('depth', 0) == current_depth + 1:
                        G.add_edge(i, j, edge_type='nested')
                    elif ast_nodes[j].get('depth', 0) <= current_depth:
                        break  # Out of nested block

    def _extract_graph_metadata(self, G: nx.DiGraph, ast_nodes: List[Dict]) -> Dict:
        """Extract graph-level features"""
        metadata = {
            'num_nodes': G.number_of_nodes(),
            'num_edges': G.number_of_edges(),
            'num_commands': sum(1 for n in ast_nodes if n.get('type') == 'command'),
            'num_conditionals': sum(1 for n in ast_nodes if n.get('type') == 'conditional'),
            'num_loops': sum(1 for n in ast_nodes if n.get('type') == 'loop'),
            'avg_risk': np.mean([n.get('risk', 0.3) for n in ast_nodes]) if ast_nodes else 0.0,
            'max_risk': max([n.get('risk', 0.3) for n in ast_nodes]) if ast_nodes else 0.0,
            'has_cycles': len(list(nx.simple_cycles(G))) > 0 if G.number_of_nodes() > 0 else False
        }

        # Network analysis metrics (if graph is not empty)
        if G.number_of_nodes() > 0:
            try:
                metadata['avg_degree'] = sum(dict(G.degree()).values()) / G.number_of_nodes()
                metadata['density'] = nx.density(G)

                # Longest path length
                if nx.is_directed_acyclic_graph(G):
                    metadata['longest_path'] = nx.dag_longest_path_length(G)
                else:
                    metadata['longest_path'] = -1  # Has cycles

            except:
                metadata['avg_degree'] = 0
                metadata['density'] = 0
                metadata['longest_path'] = 0

        # Command category distribution
        categories = {}
        for node in ast_nodes:
            if node.get('type') == 'command':
                cat = self._get_command_category(node.get('command', ''))
                categories[cat] = categories.get(cat, 0) + 1

        metadata['command_categories'] = categories

        return metadata

    def _create_empty_graph(self) -> nx.DiGraph:
        """Create empty graph for failed parsing"""
        G = nx.DiGraph()
        G.add_node(0, type='error', risk=0.0, depth=0)
        return G

    def graph_to_pyg_data(self, G: nx.DiGraph):
        """
        Convert NetworkX graph to PyTorch Geometric Data object

        Returns:
            torch_geometric.data.Data object
        """
        try:
            import torch
            from torch_geometric.data import Data
        except ImportError:
            raise ImportError("PyTorch and PyTorch Geometric are required. Run: pip install torch torch-geometric")

        # Extract node features
        node_features = []
        for node_id in range(G.number_of_nodes()):
            node = G.nodes[node_id]

            # Create feature vector: [risk, depth, is_command, is_control, category_onehot(8)]
            features = [
                node.get('risk', 0.3),
                node.get('depth', 0) / 10.0,  # Normalize depth
                1.0 if node.get('type') == 'command' else 0.0,
                1.0 if node.get('type') in ['conditional', 'loop'] else 0.0,
            ]

            # One-hot encode command category
            category = node.get('category', 'other')
            category_onehot = [0.0] * 8
            categories_list = list(self.command_categories.keys())
            if category in categories_list:
                category_onehot[categories_list.index(category)] = 1.0

            features.extend(category_onehot)

            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        # Extract edges
        edge_list = list(G.edges())
        if edge_list:
            edge_index = torch.tensor([[e[0] for e in edge_list],
                                      [e[1] for e in edge_list]], dtype=torch.long)
        else:
            # Empty graph case
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create PyG Data object
        data = Data(x=x, edge_index=edge_index)

        return data

    def visualize_graph(self, G: nx.DiGraph, output_path: str = None):
        """
        Visualize graph using matplotlib

        Args:
            G: NetworkX graph
            output_path: Path to save image (optional)
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 8))

            # Color nodes by type
            node_colors = []
            for node_id in G.nodes():
                node = G.nodes[node_id]
                risk = node.get('risk', 0.3)
                if risk > 0.7:
                    node_colors.append('red')
                elif risk > 0.5:
                    node_colors.append('orange')
                else:
                    node_colors.append('lightblue')

            # Draw graph
            pos = nx.spring_layout(G, seed=42)
            nx.draw(G, pos,
                   node_color=node_colors,
                   with_labels=True,
                   node_size=500,
                   font_size=8,
                   arrows=True,
                   edge_color='gray',
                   arrowsize=10)

            # Add node labels with command names
            labels = {}
            for node_id in G.nodes():
                node = G.nodes[node_id]
                if node.get('type') == 'command':
                    labels[node_id] = node.get('command', str(node_id))
                else:
                    labels[node_id] = str(node_id)

            nx.draw_networkx_labels(G, pos, labels, font_size=6)

            plt.title("Control Flow Graph")
            plt.axis('off')

            if output_path:
                plt.savefig(output_path, bbox_inches='tight', dpi=150)
                print(f"Graph saved to: {output_path}")
            else:
                plt.show()

            plt.close()

        except ImportError:
            print("Matplotlib not installed. Skipping visualization.")
