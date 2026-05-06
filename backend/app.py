"""
Glassdoor Jobs — Salary Prediction Flask Backend
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, render_template
import joblib
import json
import os
import numpy as np
import pandas as pd

app = Flask(__name__)

# ── Hardcoded fallback features (exact order used during training) ──────────
FALLBACK_FEATURES = [
    'python_yn', 'R_yn', 'spark', 'aws', 'excel',
    'Rating', 'age', 'desc_len', 'num_comp',
    'same_state', 'hourly', 'employer_provided',
    'job_simp_enc', 'seniority_enc', 'sector_enc',
    'size_enc', 'state_enc'
]

# ── Load model & metadata ───────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model          = None
features       = []
dashboard_data = {}

def load_artifacts():
    global model, features, dashboard_data

    model_path = os.path.join(BASE_DIR, 'salary_model.pkl')
    feat_path  = os.path.join(BASE_DIR, 'model_features.json')
    dash_path  = os.path.join(BASE_DIR, 'dashboard_data.json')

    # ── Model ──
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("✅ Model loaded:", type(model).__name__)
    else:
        print("⚠️  salary_model.pkl not found.")

    # ── Features ──
    if os.path.exists(feat_path):
        with open(feat_path) as f:
            features = json.load(f)
        print(f"✅ Features loaded: {features}")
    else:
        print("⚠️  model_features.json not found — using hardcoded fallback.")

    # Always fall back to hardcoded if empty
    if not features:
        features = FALLBACK_FEATURES
        print(f"⚠️  Using fallback features: {features}")

    # ── Dashboard data ──
    if os.path.exists(dash_path):
        with open(dash_path) as f:
            dashboard_data = json.load(f)
        print("✅ Dashboard data loaded.")
    else:
        print("⚠️  dashboard_data.json not found — charts will be empty.")

load_artifacts()


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/dashboard', methods=['GET'])
def get_dashboard():
    return jsonify(dashboard_data)


@app.route('/api/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded. Place salary_model.pkl in the backend/ folder.'}), 503

    data = request.get_json(force=True)
    print("📥 Received payload:", data)
    print("📋 Using features:", features)

    try:
        # Build as DataFrame — preserves feature names for sklearn
        row = {f: float(data.get(f, 0)) for f in features}
        X   = pd.DataFrame([row], columns=features)
        print("📐 Shape:", X.shape)

        prediction = float(model.predict(X)[0])
        print(f"💰 Prediction: ${prediction:.1f}K")

        return jsonify({
            'predicted_salary_k':   round(prediction, 1),
            'predicted_salary_usd': round(prediction * 1000, 0)
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 400


@app.route('/api/debug', methods=['GET'])
def debug():
    """Health check — visit http://localhost:5000/api/debug"""
    return jsonify({
        'model_loaded':     model is not None,
        'model_type':       type(model).__name__ if model else None,
        'features':         features,
        'feature_count':    len(features),
        'dashboard_loaded': bool(dashboard_data),
    })


@app.route('/api/encodings', methods=['GET'])
def get_encodings():
    job_roles = {
        'analyst': 0, 'data engineer': 1, 'data scientist': 2,
        'director': 3, 'manager': 4, 'mle': 5, 'na': 6
    }
    seniority = {'jr': 0, 'na': 1, 'senior': 2}
    sectors = {
        'Accounting & Legal': 0, 'Aerospace & Defense': 1,
        'Agriculture & Forestry': 2, 'Arts, Entertainment & Recreation': 3,
        'Biotech & Pharmaceuticals': 4, 'Business Services': 5,
        'Construction, Repair & Maintenance': 6, 'Consumer Services': 7,
        'Education': 8, 'Finance': 9, 'Government': 10,
        'Health Care': 11, 'Information Technology': 12,
        'Insurance': 13, 'Manufacturing': 14, 'Media': 15,
        'Non-Profit': 16, 'Oil, Gas, Energy & Utilities': 17,
        'Real Estate': 18, 'Retail': 19, 'Telecommunications': 20,
        'Transportation & Logistics': 21, 'Travel & Tourism': 22
    }
    sizes = {
        '1 to 50 employees': 0, '51 to 200 employees': 1,
        '201 to 500 employees': 2, '501 to 1000 employees': 3,
        '1001 to 5000 employees': 4, '5001 to 10000 employees': 5,
        '10000+ employees': 6, 'Unknown': 7
    }
    states = {s: i for i, s in enumerate(sorted([
        'AL','AR','AZ','CA','CO','CT','DC','DE','FL','GA','HI','IA',
        'IL','IN','KS','KY','LA','MA','MD','ME','MI','MN','MO','MS',
        'NC','NE','NJ','NM','NV','NY','OH','OK','OR','PA','RI','SC',
        'TN','TX','UT','VA','WA','WI','WV','Unknown'
    ]))}
    return jsonify({
        'job_roles': job_roles,
        'seniority': seniority,
        'sectors':   sectors,
        'sizes':     sizes,
        'states':    states
    })


if __name__ == '__main__':
    print("\n🚀 Starting Glassdoor Salary Predictor...")
    print(f"   Features ({len(features)}): {features}")
    print("   Open:  http://localhost:5000")
    print("   Debug: http://localhost:5000/api/debug\n")
    app.run(debug=True, host='0.0.0.0', port=5000)
