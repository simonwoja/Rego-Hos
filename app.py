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
        
        # List all files for debugging
        print(f"📁 Files in current directory: {os.listdir('.')}")
        
        # Check if templates directory exists
        if os.path.exists('templates'):
            print(f"📁 Files in templates directory: {os.listdir('templates')}")
        else:
            print("❌ Templates directory not found!")
        
        # Try different possible file locations
        possible_model_paths = [
            'random_forest_model.pkl',
            'diabetes_model.pkl',
            'models/random_forest_model.pkl',
            'models/diabetes_model.pkl'
        ]
        
        possible_scaler_paths = [
            'scaler.pkl',
            'models/scaler.pkl'
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
            return False
            
        if not scaler_path:
            print("❌ Scaler file not found in any location")
            return False
        
        # Load model
        print("📦 Loading model...")
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        
        # Load scaler
        print("📦 Loading scaler...")
        with open(scaler_path, 'rb') as file:
            scaler = pickle.load(file)
            
        print("✅ Model and scaler loaded successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")
        print("Full traceback:")
        traceback.print_exc()
        return False

def create_basic_templates():
    """Create basic templates if they don't exist"""
    templates_dir = 'templates'
    if not os.path.exists(templates_dir):
        os.makedirs(templates_dir)
        print("✅ Created templates directory")
    
    # Create basic index.html if it doesn't exist
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Diabetes Prediction</title>
    <style>
        body { font-family: Arial, sans-serif; max-width: 600px; margin: 50px auto; padding: 20px; }
        .container { background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
        .form-group { margin: 15px 0; }
        label { display: block; margin-bottom: 5px; font-weight: bold; }
        input { width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }
        button { background: #2196F3; color: white; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; }
        button:hover { background: #1976D2; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Diabetes Prediction</h1>
        <form action="/predict" method="POST">
            <div class="form-group">
                <label>Pregnancies:</label>
                <input type="number" name="Pregnancies" required>
            </div>
            <div class="form-group">
                <label>Glucose:</label>
                <input type="number" name="Glucose" required>
            </div>
            <div class="form-group">
                <label>Blood Pressure:</label>
                <input type="number" name="BloodPressure" required>
            </div>
            <div class="form-group">
                <label>Skin Thickness:</label>
                <input type="number" name="SkinThickness" required>
            </div>
            <div class="form-group">
                <label>Insulin:</label>
                <input type="number" name="Insulin" required>
            </div>
            <div class="form-group">
                <label>BMI:</label>
                <input type="number" step="0.1" name="BMI" required>
            </div>
            <div class="form-group">
                <label>Diabetes Pedigree Function:</label>
                <input type="number" step="0.001" name="DiabetesPedigreeFunction" required>
            </div>
            <div class="form-group">
                <label>Age:</label>
                <input type="number" name="Age" required>
            </div>
            <button type="submit">Predict</button>
        </form>
        <p><a href="/health">Health Check</a> | <a href="/debug">Debug Info</a></p>
    </div>
</body>
</html>"""
    
    with open(os.path.join(templates_dir, 'index.html'), 'w') as f:
        f.write(index_html)
    print("✅ Created basic index.html")

# Initialize app
print("🚀 Starting Diabetes Prediction App...")
create_basic_templates()
load_artifacts()

@app.route('/')
def home():
    try:
        return render_template('index.html')
    except Exception as e:
        return f"""
        <html>
            <head><title>Diabetes Prediction</title></head>
            <body>
                <h1>Diabetes Prediction App</h1>
                <p>Application is running!</p>
                <p><a href="/health">Health Status</a> | <a href="/debug">Debug Info</a></p>
                <p>Note: The main interface is being loaded...</p>
            </body>
        </html>
        """

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded', 
                'status': 'unhealthy'
            }), 503
        
        # Get form data
        data = [float(x) for x in request.form.values()]
        final_features = [np.array(data)]
        
        # Scale features and predict
        scaled_features = scaler.transform(final_features)
        prediction = model.predict(scaled_features)
        
        output = "Diabetic" if prediction[0] == 1 else "Non-Diabetic"
        
        # Try to render result template, fallback to JSON
        try:
            return render_template('result.html', prediction_text=output)
        except:
            return jsonify({'prediction': output})
        
    except Exception as e:
        error_msg = f"Prediction error: {str(e)}"
        print(f"❌ {error_msg}")
        
        # Try to render error template, fallback to JSON
        try:
            return render_template('error.html', error=error_msg), 500
        except:
            return jsonify({'error': error_msg}), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    status = {
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'status': 'healthy' if (model is not None and scaler is not None) else 'unhealthy',
        'service': 'running'
    }
    return jsonify(status)

@app.route('/debug')
def debug():
    """Debug endpoint to check files and status"""
    current_dir = os.getcwd()
    files = os.listdir('.')
    
    debug_info = {
        'current_directory': current_dir,
        'files_in_root': files,
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None,
        'templates_directory_exists': os.path.exists('templates'),
        'model_files_exist': any(f.endswith('.pkl') for f in files)
    }
    
    if os.path.exists('templates'):
        debug_info['template_files'] = os.listdir('templates')
    
    # Check for specific model files
    debug_info['random_forest_model_exists'] = os.path.exists('random_forest_model.pkl')
    debug_info['diabetes_model_exists'] = os.path.exists('diabetes_model.pkl')
    debug_info['scaler_exists'] = os.path.exists('scaler.pkl')
    
    return jsonify(debug_info)

@app.route('/check-models')
def check_models():
    """Check specifically for model files"""
    model_files = {
        'random_forest_model.pkl': os.path.exists('random_forest_model.pkl'),
        'diabetes_model.pkl': os.path.exists('diabetes_model.pkl'),
        'scaler.pkl': os.path.exists('scaler.pkl'),
        'all_files': os.listdir('.')
    }
    return jsonify(model_files)

if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0', port=5000)
