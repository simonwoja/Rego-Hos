from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import traceback

app = Flask(__name__)

# Feature names (must match training data)
FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

def load_model():
    """Load the trained model and scaler with version compatibility"""
    try:
        # Try to load with current numpy version first
        with open('random_forest_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        print("✅ Model and scaler loaded successfully")
        return model, scaler
        
    except (ImportError, AttributeError) as e:
        print(f"⚠️ Version conflict detected: {e}")
        print("🔄 Attempting compatibility fix...")
        
        try:
            # Alternative loading method for version conflicts
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open('random_forest_model.pkl', 'rb') as f:
                    model = pickle.load(f, fix_imports=True)
                with open('scaler.pkl', 'rb') as f:
                    scaler = pickle.load(f, fix_imports=True)
            
            print("✅ Model and scaler loaded with compatibility mode")
            return model, scaler
            
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            return None, None
            
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        traceback.print_exc()
        return None, None

# Load model at startup
model, scaler = load_model()

def _safe_float(value, default=0.0):
    """Convert value to float safely"""
    try:
        if value is None or (isinstance(value, str) and value.strip() == ""):
            return default
        return float(value)
    except Exception:
        return default

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not model or not scaler:
            return render_template('result.html',
                                prediction="ERROR",
                                color="warning",
                                error="Model not loaded. Please contact administrator.")

        data = request.form.to_dict()
        input_data = [_safe_float(data.get(feature, 0)) for feature in FEATURE_NAMES]
        input_df = pd.DataFrame([input_data], columns=FEATURE_NAMES)
        input_scaled = scaler.transform(input_df)

        prediction = model.predict(input_scaled)[0]
        
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_scaled)[0]
            no_diabetes_prob = probability[0] * 100
            diabetes_prob = probability[1] * 100
            confidence = float(max(probability))
        else:
            no_diabetes_prob = 100.0 if prediction == 0 else 0.0
            diabetes_prob = 100.0 if prediction == 1 else 0.0
            confidence = 100.0

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
            'confidence': round(confidence, 2),
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

@app.route('/health')
def health_check():
    if model and scaler:
        return jsonify({'status': 'healthy', 'model_loaded': True})
    else:
        return jsonify({'status': 'unhealthy', 'model_loaded': False}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
