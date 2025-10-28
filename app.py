import os
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify
import traceback

app = Flask(__name__)

# Initialize model and scaler
model = None
scaler = None

def load_artifacts():
    global model, scaler
    try:
        print("🔍 Looking for model and scaler files...")
        
        # Try different possible file locations
        possible_model_paths = [
            'random_forest_model.pkl',
            'models/random_forest_model.pkl',
            './random_forest_model.pkl',
            'diabetes_model.pkl',
            'models/diabetes_model.pkl'
        ]
        
        possible_scaler_paths = [
            'scaler.pkl',
            'models/scaler.pkl',
            './scaler.pkl'
        ]
        
        model_path = None
        scaler_path = None
        
        for path in possible_model_paths:
            if os.path.exists(path):
                model_path = path
                print(f"✅ Found model at: {path}")
                break
                
        for path in possible_scaler_paths:
            if os.path.exists(path):
                scaler_path = path
                print(f"✅ Found scaler at: {path}")
                break
        
        if not model_path:
            print("❌ Model file not found in any location")
            print(f"📁 Files in current directory: {os.listdir('.')}")
            return False
            
        if not scaler_path:
            print("❌ Scaler file not found in any location")
            return False
        
        # Load model
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        # Load scaler
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
            
        print("✅ Model and scaler loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def render_error_template(error_message):
    """Safely render error template with fallback to JSON"""
    try:
        return render_template('error.html', error=error_message), 500
    except Exception as e:
        # Fallback if template doesn't exist
        print(f"❌ Could not render error template: {e}")
        return jsonify({'error': error_message}), 500

def render_result_template(prediction_text):
    """Safely render result template with fallback to JSON"""
    try:
        return render_template('result.html', prediction_text=prediction_text)
    except Exception as e:
        # Fallback if template doesn't exist
        print(f"❌ Could not render result template: {e}")
        return jsonify({'prediction': prediction_text})

# Load artifacts when app starts
print("🚀 Starting application...")
load_artifacts()

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"""
        <html>
            <body>
                <h1>Diabetes Prediction App</h1>
                <p>Home page is loading...</p>
                <p><a href="/health">Check health status</a></p>
            </body>
        </html>
        """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            # Try to reload
            if not load_artifacts():
                error_msg = "Service temporarily unavailable. Model not loaded. Please try again in a moment."
                return render_error_template(error_msg)
        
        # Get form data
        data = [float(x) for x in request.form.values()]
        final_features = [np.array(data)]
        
        # Scale features and predict
        scaled_features = scaler.transform(final_features)
        prediction = model.predict(scaled_features)
        
        output = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        return render_result_template(output)
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"❌ {error_msg}")
        return render_error_template(error_msg)

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'status': 'ready' if (model and scaler) else 'not ready',
        'templates_working': True
    }
    return jsonify(status)

@app.route('/debug')
def debug():
    """Debug endpoint to check files and status"""
    debug_info = {
        'current_directory': os.getcwd(),
        'files': os.listdir('.'),
        'templates_exists': os.path.exists('templates'),
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }
    if os.path.exists('templates'):
        debug_info['template_files'] = os.listdir('templates')
    return jsonify(debug_info)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
