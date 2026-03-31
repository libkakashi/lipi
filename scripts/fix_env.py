#!/usr/bin/env python3
"""Fix the Python environment on GPU machine. Run: python scripts/fix_env.py"""
import subprocess, sys

def run(cmd):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Install everything needed
run(f"{sys.executable} -m pip install Pillow freetype-py numpy scipy")

# Verify freetype
try:
    import freetype
    print(f"\nfreetype-py: OK (version {freetype.__freetype_version__})")
except Exception as e:
    print(f"\nfreetype-py: FAILED ({e})")
    print("Try: apt install libfreetype6-dev && pip install freetype-py")

# Verify PIL
try:
    from PIL import Image
    print(f"Pillow: OK")
except Exception as e:
    print(f"Pillow: FAILED ({e})")

# Verify torch CUDA
try:
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"PyTorch: {e}")

print("\nDone. Now run: python scripts/check_fonts.py")
