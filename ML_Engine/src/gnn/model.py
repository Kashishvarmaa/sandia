"""
GNN Model Architecture for Malware Detection
3-layer Graph Convolutional Network (GCN) for graph-level classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


class MalwareGNN(nn.Module):
    """
    Graph Neural Network for Malware Detection

    Architecture:
        - 3 GCN layers with increasing then decreasing hidden dimensions
        - Global mean + max pooling for graph-level representation
        - Dropout for regularization
        - Binary classification (malicious/benign)
    """

    def __init__(self,
                 input_dim=12,  # Node feature dimension
                 hidden_dim=64,
                 num_layers=3,
                 dropout=0.3,
                 num_classes=2):
        super(MalwareGNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        # GCN Layers
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim * 2)
        self.conv3 = GCNConv(hidden_dim * 2, hidden_dim)

        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim * 2)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Fully connected layers for classification
        # After pooling, we concatenate mean and max: hidden_dim * 2
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, data):
        """
        Forward pass

        Args:
            data: PyG Data object with attributes:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge list [2, num_edges]
                - batch: Batch assignment vector [num_nodes]

        Returns:
            - logits: Class logits [batch_size, num_classes]
            - embeddings: Graph embeddings [batch_size, hidden_dim]
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # GCN Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # GCN Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout_layer(x)

        # GCN Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Graph-level pooling (combine mean and max)
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)

        # Save graph embedding for analysis
        graph_embedding = x.clone()

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout_layer(x)

        x = F.relu(self.fc2(x))
        x = self.dropout_layer(x)

        x = self.fc3(x)

        return x, graph_embedding

    def predict(self, data):
        """
        Make prediction with confidence score

        Returns:
            - predicted_class: 0 (benign) or 1 (malicious)
            - confidence: Probability of predicted class
            - probabilities: [prob_benign, prob_malicious]
        """
        self.eval()
        with torch.no_grad():
            logits, embedding = self.forward(data)
            probabilities = F.softmax(logits, dim=1)

            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, predicted_class].item()

            return {
                'predicted_class': predicted_class,
                'is_malicious': bool(predicted_class == 1),
                'confidence': confidence,
                'prob_benign': probabilities[0, 0].item(),
                'prob_malicious': probabilities[0, 1].item(),
                'embedding': embedding.cpu().numpy()
            }


class EarlyStopping:
    """Early stopping to prevent overfitting"""

    def __init__(self, patience=10, min_delta=0.001, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                if self.verbose:
                    print('Early stopping triggered')
        else:
            self.best_loss = val_loss
            self.best_model_state = model.state_dict()
            self.counter = 0

        return self.early_stop


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_info(model):
    """Get model information"""
    return {
        'architecture': 'MalwareGNN',
        'input_dim': model.input_dim,
        'hidden_dim': model.hidden_dim,
        'num_layers': model.num_layers,
        'dropout': model.dropout,
        'num_classes': model.num_classes,
        'total_parameters': count_parameters(model),
        'trainable_parameters': count_parameters(model)
    }
