# Force numpy compatibility at the VERY beginning
try:
    import force_numpy_compat
    print("✅ NumPy compatibility forced at startup")
except Exception as e:
    print(f"⚠️ Could not force numpy compatibility: {e}")

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import os
import traceback
import sys

print("🔍 Starting application...")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")

# Double-check numpy._core exists
if hasattr(np, '_core'):
    print("✅ numpy._core verification: EXISTS")
else:
    print("❌ numpy._core verification: MISSING")

print("🔍 Checking for model files...")
print("Current directory:", os.getcwd())
print("Files in directory:", os.listdir('.'))

app = Flask(__name__)

class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles numpy._core redirects"""
    def find_class(self, module, name):
        # Redirect any numpy._core references to numpy.core
        if module.startswith('numpy._core'):
            new_module = module.replace('numpy._core', 'numpy.core', 1)
            try:
                return super().find_class(new_module, name)
            except (AttributeError, ModuleNotFoundError):
                # If redirect fails, try original
                return super().find_class(module, name)
        else:
            return super().find_class(module, name)

def load_model():
    """Load model with multiple fallback strategies"""
    try:
        model_path = 'random_forest_model.pkl'
        scaler_path = 'scaler.pkl'
        
        print(f"📁 Model exists: {os.path.exists(model_path)}")
        print(f"📁 Scaler exists: {os.path.exists(scaler_path)}")
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print("❌ Model or scaler files missing")
            return None, None

        model = None
        scaler = None
        
        # Strategy 1: Try with custom unpickler
        print("🔄 Strategy 1: Custom unpickler...")
        try:
            with open(model_path, 'rb') as f:
                model = CustomUnpickler(f).load()
            print("✅ Model loaded with custom unpickler")
        except Exception as e1:
            print(f"❌ Custom unpickler failed: {e1}")
            
            # Strategy 2: Try direct load with forced compatibility
            print("🔄 Strategy 2: Direct load with compatibility...")
            try:
                # Ensure numpy._core exists
                if not hasattr(np, '_core'):
                    np._core = np.core
                
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                print("✅ Model loaded with direct method")
            except Exception as e2:
                print(f"❌ Direct load failed: {e2}")
                
                # Strategy 3: Nuclear option - monkey patch find_class
                print("🔄 Strategy 3: Monkey patch approach...")
                try:
                    original_find_class = pickle.Unpickler.find_class
                    
                    def patched_find_class(self, module, name):
                        if module.startswith('numpy._core'):
                            module = module.replace('numpy._core', 'numpy.core', 1)
                        return original_find_class(self, module, name)
                    
                    pickle.Unpickler.find_class = patched_find_class
                    
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    # Restore original method
                    pickle.Unpickler.find_class = original_find_class
                    print("✅ Model loaded with monkey patch")
                except Exception as e3:
                    print(f"❌ All strategies failed: {e3}")
                    return None, None

        # Load scaler (usually simpler)
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
            print("✅ Scaler loaded")
        except Exception as e:
            print(f"❌ Scaler load failed: {e}")
            return None, None

        # Verify model
        if model and hasattr(model, 'predict'):
            print("✅ Model verification passed")
            return model, scaler
        else:
            print("❌ Model verification failed")
            return None, None
            
    except Exception as e:
        print(f"❌ Critical error in load_model: {e}")
        traceback.print_exc()
        return None, None

# Load the model
print("🚀 Loading model...")
model, scaler = load_model()

if model and scaler:
    print("🎉 Model and scaler successfully loaded!")
else:
    print("💥 Failed to load model and scaler")

FEATURE_NAMES = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
]

def _safe_float(value, default=0.0):
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
                                 error="Model not loaded. Please check server logs.")

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
            no_diabetes_prob, diabetes_prob = (100.0, 0.0) if prediction == 0 else (0.0, 100.0)
            confidence = 1.0

        result = "DIABETES" if prediction == 1 else "NO DIABETES"
        color = "danger" if prediction == 1 else "success"
        recommendation = "Consult healthcare professional." if prediction == 1 else "Maintain healthy lifestyle."

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
        return render_template('result.html',
                             prediction="ERROR",
                             color="warning",
                             error=f"Prediction error: {str(e)}")

@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy' if model and scaler else 'unhealthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
