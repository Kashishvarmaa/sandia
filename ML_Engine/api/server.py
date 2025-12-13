"""
ML Engine API Server
Flask server for GNN inference
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.gnn.inference import get_predictor

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Load predictor on startup
try:
    predictor = get_predictor()
    MODEL_LOADED = True
except Exception as e:
    print(f"[ML API] Warning: Model loading failed: {e}")
    print(f"[ML API] Train model first: python src/gnn/train.py")
    MODEL_LOADED = False


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'ML Engine API',
        'model_loaded': MODEL_LOADED
    })


@app.route('/api/ml/gnn/analyze', methods=['POST'])
def analyze_gnn():
    """
    GNN analysis endpoint

    Request body:
    {
        "script_content": "#!/bin/bash\nwget ..."
    }

    Response:
    {
        "is_malicious": true,
        "risk_score": 87.5,
        "confidence": 0.92,
        "attack_pattern": "download-execute-cleanup",
        "graph_metadata": {...}
    }
    """
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model not loaded',
            'message': 'Train model first using: python src/gnn/train.py'
        }), 503

    try:
        data = request.get_json()

        if not data or 'script_content' not in data:
            return jsonify({'error': 'Missing script_content in request body'}), 400

        script_content = data['script_content']

        # Run GNN prediction
        result = predictor.predict(script_content)

        if 'error' in result:
            return jsonify(result), 500

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            'error': str(e),
            'message': 'Internal server error during GNN analysis'
        }), 500


@app.route('/api/ml/models/info', methods=['GET'])
def models_info():
    """Get information about loaded models"""
    info = {
        'gnn': {
            'status': 'loaded' if MODEL_LOADED else 'not_loaded',
            'model_path': '../../models/gnn/gnn_malware_detector.pth'
        },
        'bert': {
            'status': 'not_implemented',
            'message': 'BERT integration coming soon'
        },
        'bedrock': {
            'status': 'not_implemented',
            'message': 'Bedrock integration coming soon'
        }
    }

    return jsonify(info), 200


if __name__ == '__main__':
    port = int(os.environ.get('ML_API_PORT', 5001))
    print(f"[ML API] Starting server on port {port}")
    print(f"[ML API] Model loaded: {MODEL_LOADED}")
    print(f"[ML API] Endpoints:")
    print(f"  - POST http://localhost:{port}/api/ml/gnn/analyze")
    print(f"  - GET  http://localhost:{port}/api/ml/models/info")
    print(f"  - GET  http://localhost:{port}/health")

    app.run(host='0.0.0.0', port=port, debug=True)
