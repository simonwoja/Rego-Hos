from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import traceback
import warnings

print("🔍 Starting application...")
print("Python version info:")
try:
    import sys
    print(f"Python: {sys.version}")
    print(f"NumPy version: {np.__version__}")
    print(f"Pandas version: {pd.__version__}")
except:
    pass

print("🔍 Checking for model files...")
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))

# Initialize Flask app
app = Flask(__name__)

def load_model_with_compatibility():
    """Load model with multiple compatibility strategies"""
    try:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_dir, 'random_forest_model.pkl')
        scaler_path = os.path.join(base_dir, 'scaler.pkl')

        print(f"🔍 Looking for model at: {model_path}")
        print(f"🔍 Looking for scaler at: {scaler_path}")
        
        if not os.path.exists(model_path):
            print(f"❌ Model file not found at: {model_path}")
            return None, None
            
        if not os.path.exists(scaler_path):
            print(f"❌ Scaler file not found at: {scaler_path}")
            return None, None

        print("✅ Model and scaler files found. Loading with compatibility fixes...")
        
        # Strategy 1: Try normal loading first
        try:
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            print("✅ Model loaded with normal method")
        except Exception as e:
            print(f"❌ Normal loading failed: {e}")
            # Strategy 2: Try with warnings suppressed
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(model_path, 'rb') as file:
                        model = pickle.load(file)
                print("✅ Model loaded with warnings suppressed")
            except Exception as e2:
                print(f"❌ Warnings suppression failed: {e2}")
                # Strategy 3: Try with specific numpy compatibility
                try:
                    # This handles the numpy._core issue
                    import numpy as np
                    original_getattr = getattr
                    def compatible_getattr(obj, name):
                        if name == '_core' and hasattr(obj, 'core'):
                            return obj.core
                        return original_getattr(obj, name)
                    
                    # Temporarily replace getattr
                    import builtins
                    builtins.getattr = compatible_getattr
                    
                    with open(model_path, 'rb') as file:
                        model = pickle.load(file)
                    
                    # Restore original getattr
                    builtins.getattr = original_getattr
                    print("✅ Model loaded with numpy compatibility fix")
                except Exception as e3:
                    print(f"❌ All compatibility strategies failed: {e3}")
                    return None, None

        # Load scaler (usually has fewer compatibility issues)
        try:
            with open(scaler_path, 'rb') as file:
                scaler = pickle.load(file)
            print("✅ Scaler loaded successfully")
        except Exception as e:
            print(f"❌ Scaler loading failed: {e}")
            return None, None
        
        # Verify model functionality
        if hasattr(model, 'predict'):
            print("✅ Model has predict method - basic verification passed")
            
            # Test prediction with dummy data if possible
            try:
                # Create minimal test input
                test_input = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
                if hasattr(scaler, 'transform'):
                    test_input = scaler.transform(test_input)
                test_pred = model.predict(test_input)
                print(f"✅ Model test prediction successful: {test_pred}")
            except Exception as test_error:
                print(f"⚠️ Model test prediction failed (but model loaded): {test_error}")
        else:
            print("❌ Model doesn't have predict method!")
            return None, None
            
        return model, scaler
        
    except Exception as e:
        print(f"❌ Critical error in load_model_with_compatibility: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return None, None

# Load model at startup
print("🚀 Loading model with compatibility fixes...")
model, scaler = load_model_with_compatibility()

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
            error_msg = "Model or scaler not loaded on server. Please check the server logs."
            return render_template('result.html',
                                 prediction="ERROR",
                                 color="warning",
                                 error=error_msg)

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

@app.route('/debug')
def debug_info():
    """Debug information endpoint"""
    import sys
    info = {
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'pandas_version': pd.__version__,
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'current_directory': os.getcwd(),
        'files': os.listdir('.')
    }
    return jsonify(info)

# Render.com specific configuration
if __name__ == '__main__':
    print("🚀 Starting Diabetes Prediction Web Application...")
    print("📊 Model Status:", "Loaded" if model else "Not Loaded")
    
    # Get port from environment variable (Render sets this)
    port = int(os.environ.get('PORT', 5000))
    
    print(f"🌐 Web App will be available at: http://0.0.0.0:{port}")
    
    # Run without debug mode for production
    app.run(host='0.0.0.0', port=port, debug=False)
