from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import traceback

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
def load_model():
    """Load the trained model and scaler"""
    try:
        # In Render, files are in the same directory
        model_path = 'random_forest_model.pkl'
        scaler_path = 'scaler.pkl'

        print(f"🔍 Looking for model at: {model_path}")
        print(f"🔍 Looking for scaler at: {scaler_path}")
        
        # List all files for debugging
        print("📁 Files in directory:", os.listdir('.'))
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {model_path}")
            return None, None
            
        if not os.path.exists(scaler_path):
            print(f"❌ Scaler file not found at: {scaler_path}")
            return None, None

        print("✅ Model and scaler files found. Loading...")
        
        # Load model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"✅ Model loaded. Type: {type(model)}")
        
        # Load scaler
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        print(f"✅ Scaler loaded. Type: {type(scaler)}")
        
        return model, scaler
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return None, None

# Load model at startup
model, scaler = load_model()

# Feature names (must match training data)
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

def _safe_float(value, default=0.0):
    """Convert value to float safely"""
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == "":
            return default
        return float(value)
    except Exception:
        return default

@app.route('/')
def home():
    """Render the home page with input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if not model or not scaler:
            raise RuntimeError("Model or scaler not loaded on server.")

        data = request.form.to_dict()
        input_data = [_safe_float(data.get(feature, 0)) for feature in FEATURE_NAMES]

        # Create DataFrame and scale
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0]
            no_diabetes_prob = probability[0] * 100
            diabetes_prob = probability[1] * 100
            confidence = float(max(probability))
        else:
            if prediction == 1:
                no_diabetes_prob, diabetes_prob = 0.0, 100.0
                confidence = 1.0
            else:
                no_diabetes_prob, diabetes_prob = 100.0, 0.0
                confidence = 1.0

        # Determine result
        if prediction == 1:
            result = "DIABETES"
            color = "danger"
            recommendation = "Please consult with a healthcare professional for further evaluation and management."
        else:
            result = "NO DIABETES"
            color = "success"
            recommendation = "Maintain a healthy lifestyle with regular exercise and balanced diet."

        response_data = {
            'prediction': result,
            'color': color,
            'confidence': round(confidence * 100, 2),
            'no_diabetes_prob': round(no_diabetes_prob, 2),
            'diabetes_prob': round(diabetes_prob, 2),
            'recommendation': recommendation,
            'input_values': {FEATURE_NAMES[i]: input_data[i] for i in range(len(FEATURE_NAMES))}
        }

        return render_template('result.html', **response_data)

    except Exception as e:
        error_message = f"Error processing request: {str(e)}"
        return render_template('result.html',
                               prediction="ERROR",
                               color="warning",
                               error=error_message)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        if not model or not scaler:
            return jsonify({'error': 'Model or scaler not loaded on server.'}), 503

        data = request.get_json() or {}
        input_data = [_safe_float(data.get(feature, 0)) for feature in FEATURE_NAMES]
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0]
            no_p, yes_p = float(probability[0]), float(probability[1])
            confidence = float(max(probability))
        else:
            no_p, yes_p = (1.0, 0.0) if prediction == 0 else (0.0, 1.0)
            confidence = max(no_p, yes_p)

        return jsonify({
            'prediction': int(prediction),
            'probabilities': {
                'no_diabetes': no_p,
                'diabetes': yes_p
            },
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    if model and scaler:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

# Remove the debug server startup for production
