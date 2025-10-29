from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import traceback

# Add this after your imports in app.py
import os
print("🔍 Checking for model files...")
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))

model_path = 'random_forest_model.pkl'
scaler_path = 'scaler.pkl'

print(f"Model file exists: {os.path.exists(model_path)}")
print(f"Scaler file exists: {os.path.exists(scaler_path)}")

# Initialize Flask app
app = Flask(__name__)

# Load the trained model and scaler
def load_model():
    """Load the trained model and scaler from the script directory"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'random_forest_model.pkl')
        scaler_path = os.path.join(base_dir, 'scaler.pkl')

        print(f"🔍 Looking for model at: {model_path}")
        print(f"🔍 Looking for scaler at: {scaler_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {model_path}")
            # List files to help debug
            print("📁 Files in directory:", [f for f in os.listdir(base_dir) if f.endswith('.pkl')])
            return None, None
            
        if not os.path.exists(scaler_path):
            print(f"❌ Scaler file not found at: {scaler_path}")
            return None, None

        print("✅ Model and scaler files found. Loading...")
        
        # Try to load model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        print(f"✅ Model loaded. Type: {type(model)}")
        
        # Try to load scaler
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
        print(f"✅ Scaler loaded. Type: {type(scaler)}")
        
        # Test if model has predict method
        if hasattr(model, 'predict'):
            print("✅ Model has predict method")
        else:
            print("❌ Model doesn't have predict method!")
            
        return model, scaler
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("Full traceback:")
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
    """Convert value to float safely (handles None, empty strings, invalid input)."""
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
    """Handle prediction requests from the web form"""
    try:
        if not model or not scaler:
            raise RuntimeError("Model or scaler not loaded on server.")

        data = request.form.to_dict()

        # Convert to numerical values safely
        input_data = [_safe_float(data.get(feature, 0)) for feature in FEATURE_NAMES]

        # Create DataFrame and scale
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        input_scaled = scaler.transform(input_df)

        # Predict (handle models without predict_proba)
        prediction = model.predict(input_scaled)[0]
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0]
            no_diabetes_prob = probability[0] * 100
            diabetes_prob = probability[1] * 100
            confidence = float(max(probability))
        else:
            # fallback deterministic confidence
            if prediction == 1:
                no_diabetes_prob, diabetes_prob = 0.0, 100.0
                confidence = 1.0
            else:
                no_diabetes_prob, diabetes_prob = 100.0, 0.0
                confidence = 1.0

        # Determine result and color
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
    """API endpoint for predictions (for potential mobile app integration)"""
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

# Render.com specific configuration
if __name__ == '__main__':
    print("🚀 Starting Diabetes Prediction Web Application...")
    print("📊 Model Status:", "Loaded" if model else "Not Loaded")
    print("🌐 Web App will be available at: http://localhost:5000")
    
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    # Run without debug mode for production
    app.run(host='0.0.0.0', port=port, debug=False)
