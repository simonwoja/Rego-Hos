# numpy_compat.py
# This file creates the missing numpy._core module for compatibility
import numpy as np
import sys

# Check if numpy._core exists, if not create it
if not hasattr(np, '_core'):
    print("🔄 Creating numpy._core compatibility layer...")
    np._core = np.core
    sys.modules['numpy._core'] = np.core
    sys.modules['numpy._core._multiarray_umath'] = np.core._multiarray_umath
    print("✅ numpy._core compatibility layer created")
else:
    print("✅ numpy._core already exists")
