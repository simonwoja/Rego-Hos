# force_numpy_compat.py
import sys
import types

def force_numpy_compatibility():
    """Force create numpy._core module structure"""
    print("🔧 Forcing numpy._core compatibility...")
    
    try:
        import numpy as np
        
        # Check if we need to create the compatibility layer
        if not hasattr(np, '_core'):
            print("🔄 Creating numpy._core structure...")
            
            # Create _core as a direct reference to core
            np._core = np.core
            
            # Add _core to sys.modules
            sys.modules['numpy._core'] = np.core
            
            # Handle common submodules that might be referenced
            if hasattr(np.core, 'multiarray'):
                np._core.multiarray = np.core.multiarray
                sys.modules['numpy._core.multiarray'] = np.core.multiarray
                
            if hasattr(np.core, '_multiarray_umath'):
                np._core._multiarray_umath = np.core._multiarray_umath
                sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
                
            if hasattr(np.core, 'numeric'):
                np._core.numeric = np.core.numeric
                sys.modules['numpy._core.numeric'] = np.core.numeric
            
            print("✅ numpy._core compatibility forced")
        else:
            print("✅ numpy._core already exists")
            
        return True
        
    except Exception as e:
        print(f"❌ Failed to force numpy compatibility: {e}")
        return False

# Run immediately when imported
force_numpy_compatibility()
