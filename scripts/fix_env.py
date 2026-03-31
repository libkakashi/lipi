#!/usr/bin/env python3
"""Fix the Python environment on GPU machine. Run: python scripts/fix_env.py"""
import subprocess, sys

def run(cmd):
    print(f"$ {cmd}")
    subprocess.run(cmd, shell=True, check=True)

# Fix Pillow (needs FreeType for font rendering)
run(f"{sys.executable} -m pip uninstall pillow pillow-simd -y")
run(f"{sys.executable} -m pip install Pillow")

# Verify
try:
    from PIL import ImageFont
    f = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
    print("\nPillow FreeType: OK")
except Exception:
    # Try with our downloaded font
    import importlib
    import PIL
    importlib.reload(PIL)
    from PIL import ImageFont
    from pathlib import Path
    fonts = list(Path("training_data/fonts").glob("NotoSans-Regular*"))
    if fonts:
        f = ImageFont.truetype(str(fonts[0]), 12)
        print("\nPillow FreeType: OK (using Noto)")
    else:
        print("\nWARNING: Pillow FreeType still broken")

# Verify torch CUDA
try:
    import torch
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
except Exception as e:
    print(f"PyTorch: {e}")

print("\nDone. Now run: python scripts/check_fonts.py")
