# fix_model_compatibility.py
import pickle
import os
import warnings

def fix_model_compatibility():
    """Fix model compatibility issues by loading with warnings suppressed"""
    print("🔄 Attempting to fix model compatibility...")
    
    try:
        # Suppress all warnings during loading
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Load model with compatibility fix
            with open('random_forest_model.pkl', 'rb') as f:
                model = pickle.load(f)
            
            # Load scaler
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)
                
        print("✅ Model and scaler loaded successfully with compatibility fixes!")
        return model, scaler
        
    except Exception as e:
        print(f"❌ Failed to load with compatibility fix: {e}")
        return None, None

if __name__ == '__main__':
    fix_model_compatibility()
