from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
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
            # List files to help debug
            print("📁 Files in directory:", [f for f in os.listdir('.') if f.endswith('.pkl')])
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

# Feature names used by the model
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

def _safe_float(value, default=0.0):
    """Convert value to float safely"""
    try:
        if value is None:
            return default
        return float(value)
    except (ValueError, TypeError):
        return default

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if not model or not scaler:
            raise RuntimeError("Model or scaler not loaded on server.")

        data = request.form.to_dict()

        # Convert to numerical values safely
        input_data = [_safe_float(data.get(feature, 0)) for feature in FEATURE_NAMES]

        # Create DataFrame and scale
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        input_scaled = scaler.transform(input_df)

        # Predict
        prediction = model.predict(input_scaled)[0]
        
        # Get probabilities if available
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

        # Determine result
        if prediction == 1:
            result = "DIABETES"
            color = "danger"
            explanation = "The model predicts a high likelihood of diabetes. Please consult with a healthcare professional."
        else:
            result = "NO DIABETES"
            color = "success"
            explanation = "The model predicts a low likelihood of diabetes. Maintain a healthy lifestyle."

        return render_template('result.html',
                            result=result,
                            color=color,
                            explanation=explanation,
                            no_diabetes_prob=round(no_diabetes_prob, 2),
                            diabetes_prob=round(diabetes_prob, 2),
                            confidence=round(confidence * 100, 2))

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return render_template('error.html', error=str(e)), 500

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
        
        response = {
            'prediction': int(prediction),
            'result': 'diabetes' if prediction == 1 else 'no_diabetes'
        }

        # Add probabilities if available
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0]
            response['probabilities'] = {
                'no_diabetes': float(probability[0]),
                'diabetes': float(probability[1])
            }
            response['confidence'] = float(max(probability))

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    if model and scaler:
        return jsonify({'status': 'healthy', 'model_loaded': True}), 200
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

# Render.com will use the PORT environment variable
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("🚀 Starting Diabetes Prediction Web Application...")
    print("📊 Model Status:", "Loaded" if model else "Not Loaded")
    print(f"🌐 Web App will be available at: http://0.0.0.0:{port}")
    app.run(debug=False, host='0.0.0.0', port=port)
