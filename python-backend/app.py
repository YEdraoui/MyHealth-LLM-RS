"""
Flask API Wrapper for Ophthalmology AI
Bridges Next.js frontend with existing Python inference pipeline
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import sys
from pathlib import Path
import json
from datetime import datetime

# Add your existing project to path
ORIGINAL_PROJECT_PATH = r'C:\MyCapstone\ophthalmology-ai'
sys.path.insert(0, ORIGINAL_PROJECT_PATH)

# Import your existing pipeline
try:
    from inference.pipeline import ClinicalInferencePipeline
    PIPELINE_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import pipeline: {e}")
    PIPELINE_AVAILABLE = False

app = Flask(__name__)

# Enable CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["http://localhost:3000", "http://127.0.0.1:3000"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize pipeline
pipeline = None
if PIPELINE_AVAILABLE:
    try:
        pipeline = ClinicalInferencePipeline()
        print("✅ Pipeline initialized successfully!")
    except Exception as e:
        print(f"❌ Pipeline initialization failed: {e}")

# Upload folder
UPLOAD_FOLDER = Path(__file__).parent / 'uploads'
UPLOAD_FOLDER.mkdir(exist_ok=True)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'pipeline_ready': pipeline is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/analyze', methods=['POST', 'OPTIONS'])
def analyze_image():
    """Analyze fundus image"""
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        if pipeline is None:
            return jsonify({'error': 'Pipeline not initialized'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'Empty filename'}), 400
        
        report_type = request.form.get('report_type', 'brief')
        
        # Save file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{file.filename}"
        filepath = UPLOAD_FOLDER / filename
        file.save(str(filepath))
        
        print(f"📁 Saved file: {filepath}")
        print(f"📋 Report type: {report_type}")
        print(f"🔄 Processing...")
        
        # Get raw result
        raw = pipeline.process_patient(str(filepath), report_type)
        
        print(f"✅ Analysis complete!")
        print(f"📊 Raw result keys: {list(raw.keys())}")
        
        # Map the response to expected format
        predictions = raw.get('predictions', {})
        
        # Extract severity from predictions
        severity_data = predictions.get('severity', {})
        severity_level = severity_data.get('class', 0)
        severity_confidence = severity_data.get('confidence', 0.0)
        
        severity_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        severity_name = severity_names[severity_level] if 0 <= severity_level < 5 else 'Unknown'
        
        # Extract conditions from predictions
        condition_names = [
            'Microaneurysms', 'Hemorrhages', 'Hard Exudates', 'Soft Exudates',
            'Neovascularization', 'Vitreous Hemorrhage', 'Preretinal Hemorrhage',
            'Macular Edema', 'CSME', 'Laser Scars', 'Retinal Detachment',
            'Tractional Detachment', 'Other Lesions'
        ]
        
        conditions_data = predictions.get('conditions', {})
        conditions = []
        
        for name in condition_names:
            cond_info = conditions_data.get(name, {})
            
            # Handle both dict and float formats
            if isinstance(cond_info, dict):
                prob = cond_info.get('probability', 0.0)
            else:
                prob = float(cond_info) if cond_info else 0.0
            
            conditions.append({
                'name': name,
                'probability': prob,
                'detected': prob >= 0.5
            })
        
        # Build response
        result = {
            'severity': {
                'level': severity_level,
                'confidence': severity_confidence,
                'name': severity_name,
                'distribution': severity_data.get('probabilities', [])
            },
            'conditions': conditions,
            'report': raw.get('report', 'No report generated'),
            'gradcam': raw.get('gradcam', None),
            'processing_time': {
                'total': raw.get('processing_time_seconds', 0),
                'vision': 0,
                'vlm': 0
            },
            'timestamp': raw.get('timestamp', timestamp),
            'metadata': {
                'filename': file.filename,
                'timestamp': timestamp,
                'report_type': report_type
            }
        }
        
        print(f"📤 Sending response:")
        print(f"   Severity: {result['severity']['name']} ({result['severity']['confidence']*100:.1f}%)")
        print(f"   Conditions: {sum(1 for c in conditions if c['detected'])} detected")
        
        return jsonify(result)
    
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Analysis failed', 'details': str(e)}), 500

@app.route('/api/batch', methods=['POST'])
def batch_analyze():
    return jsonify({'error': 'Not implemented yet'}), 501

if __name__ == '__main__':
    print("=" * 70)
    print("🏥 OPHTHALMOLOGY AI - PYTHON BACKEND")
    print("=" * 70)
    print(f"Original project: {ORIGINAL_PROJECT_PATH}")
    print(f"Pipeline status: {'✅ Ready' if pipeline else '❌ Not available'}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    print("=" * 70)
    print("Starting Flask server on http://localhost:8000")
    print("CORS enabled for: http://localhost:3000")
    print("=" * 70)
    
    app.run(host='0.0.0.0', port=8000, debug=True)
