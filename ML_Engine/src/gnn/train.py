"""
GNN Training Script
Train malware detection GNN on shell script dataset
"""

import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import json
import os
from datetime import datetime
from tqdm import tqdm
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gnn.model import MalwareGNN, EarlyStopping, get_model_info
from gnn.graph_builder import ScriptGraphBuilder


def load_dataset(dataset_dir='../../datasets/raw'):
    """Load and convert shell scripts to graphs"""
    builder = ScriptGraphBuilder()
    graphs = []
    labels = []

    malicious_dir = os.path.join(dataset_dir, 'malicious')
    benign_dir = os.path.join(dataset_dir, 'benign')

    # Load malicious samples
    if os.path.exists(malicious_dir):
        for filename in os.listdir(malicious_dir):
            if filename.endswith('.sh'):
                filepath = os.path.join(malicious_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        script = f.read()

                    G, metadata = builder.build_graph(script)
                    pyg_data = builder.graph_to_pyg_data(G)
                    pyg_data.y = torch.tensor([1], dtype=torch.long)  # Malicious label
                    pyg_data.filename = filename

                    graphs.append(pyg_data)
                    labels.append(1)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    # Load benign samples
    if os.path.exists(benign_dir):
        for filename in os.listdir(benign_dir):
            if filename.endswith('.sh'):
                filepath = os.path.join(benign_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        script = f.read()

                    G, metadata = builder.build_graph(script)
                    pyg_data = builder.graph_to_pyg_data(G)
                    pyg_data.y = torch.tensor([0], dtype=torch.long)  # Benign label
                    pyg_data.filename = filename

                    graphs.append(pyg_data)
                    labels.append(0)
                except Exception as e:
                    print(f"Error processing {filename}: {e}")

    print(f"Loaded {len(graphs)} graphs ({sum(labels)} malicious, {len(labels) - sum(labels)} benign)")
    return graphs


def train_epoch(model, loader, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()

        logits, _ = model(data)
        loss = F.cross_entropy(logits, data.y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        pred = logits.argmax(dim=1)
        correct += (pred == data.y).sum().item()
        total += data.num_graphs

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def evaluate(model, loader, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            logits, _ = model(data)
            loss = F.cross_entropy(logits, data.y)

            total_loss += loss.item() * data.num_graphs
            pred = logits.argmax(dim=1)
            probs = F.softmax(logits, dim=1)[:, 1]  # Probability of malicious class

            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    try:
        roc_auc = roc_auc_score(all_labels, all_probs)
    except:
        roc_auc = 0.0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc
    }


def train_model(config=None):
    """Main training function"""
    if config is None:
        config = {
            'hidden_dim': 64,
            'num_layers': 3,
            'dropout': 0.3,
            'learning_rate': 0.001,
            'epochs': 100,
            'batch_size': 32,
            'patience': 15
        }

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(script_dir, '../../datasets/raw')
    print(f"Loading dataset from: {dataset_dir}")

    graphs = load_dataset(dataset_dir)

    if len(graphs) < 4:
        print(f"WARNING: Only {len(graphs)} samples found. Need at least 4 for train/val split.")
        print("Add more .sh files to datasets/raw/malicious/ and datasets/raw/benign/")
        return

    # Train/val split (80/20)
    split_idx = int(0.8 * len(graphs))
    train_data = graphs[:split_idx]
    val_data = graphs[split_idx:]

    print(f"Train: {len(train_data)} samples, Val: {len(val_data)} samples")

    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False)

    # Initialize model
    model = MalwareGNN(
        input_dim=12,
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    print(f"\nModel Info:")
    for key, value in get_model_info(model).items():
        print(f"  {key}: {value}")

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    early_stopping = EarlyStopping(patience=config['patience'])

    # Training loop
    print("\nStarting training...")
    best_val_f1 = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_metrics': []}

    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_metrics'].append(val_metrics)

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{config['epochs']}")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
            print(f"  Val F1: {val_metrics['f1']:.4f}, Val AUC: {val_metrics['roc_auc']:.4f}")

        # Save best model
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            save_model(model, config, val_metrics, history)

        # Early stopping
        if early_stopping(val_metrics['loss'], model):
            print(f"Early stopping at epoch {epoch}")
            break

    print("\nTraining completed!")
    print(f"Best Val F1: {best_val_f1:.4f}")

    return model, history


def save_model(model, config, metrics, history):
    """Save trained model and metadata"""
    model_dir = '../../models/gnn'
    os.makedirs(model_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model weights
    model_path = os.path.join(model_dir, 'gnn_malware_detector.pth')
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to: {model_path}")

    # Save config
    config_path = os.path.join(model_dir, 'model_config.json')
    with open(config_path, 'w') as f:
        json.dump({
            'model_architecture': get_model_info(model),
            'training_config': config,
            'timestamp': timestamp
        }, f, indent=2)

    # Save metrics
    metrics_path = os.path.join(model_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump({
            'best_metrics': metrics,
            'training_history': history,
            'timestamp': timestamp
        }, f, indent=2)

    print(f"Config and metrics saved")


if __name__ == '__main__':
    train_model()
