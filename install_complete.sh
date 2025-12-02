#!/bin/bash
# Complete Installation Script - Sovereign AI Collective Image Generator
# with ai-forever Real-ESRGAN (Python 3.13 compatible)

set -e  # Exit on error

echo "========================================================================"
echo "  Sovereign AI Collective - Complete Installation"
echo "  Python 3.13 + CUDA 12.8 + ai-forever Real-ESRGAN"
echo "========================================================================"
echo ""

# 1. Verify we're in the right directory
if [ ! -f "generate_gui.py" ]; then
    echo "❌ Error: generate_gui.py not found"
    echo "   Please run this script from your imagegen directory"
    exit 1
fi

# 2. Verify venv is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "❌ Error: Virtual environment not activated"
    echo "   Please run: source venv/bin/activate"
    exit 1
fi

echo "✓ Virtual environment active: $VIRTUAL_ENV"
echo ""

# 3. Install core dependencies (everything except Real-ESRGAN)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 1/3: Installing core dependencies"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

pip install torch torchvision diffusers transformers accelerate \
  compel Pillow opencv-python "numpy<2.0" gradio huggingface-hub \
  safetensors psutil xformers --break-system-packages

if [ $? -ne 0 ]; then
    echo "❌ Failed to install core dependencies"
    exit 1
fi

echo ""
echo "✓ Core dependencies installed"
echo ""

# 4. Install ai-forever Real-ESRGAN
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 2/3: Installing ai-forever Real-ESRGAN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

pip install git+https://github.com/ai-forever/Real-ESRGAN.git --break-system-packages

if [ $? -ne 0 ]; then
    echo "❌ Failed to install Real-ESRGAN"
    echo "   The GUI will still work but fall back to LANCZOS upscaling"
fi

echo ""
echo "✓ Real-ESRGAN installed (ai-forever implementation)"
echo ""

# 5. Download SDXL models
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Step 3/3: Downloading SDXL models"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ -d "./models/juggernaut-xl-v9" ] && [ -d "./models/realvisxl-v4" ]; then
    echo "✓ Models already downloaded, skipping"
else
    python download_all_assets.py
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to download models"
        echo "   You can run this manually later: python download_all_assets.py"
    fi
fi

echo ""
echo "========================================================================"
echo "  Installation Complete! 🔥"
echo "========================================================================"
echo ""
echo "Installed components:"
echo "  ✓ PyTorch $(python -c 'import torch; print(torch.__version__)')"
echo "  ✓ Diffusers (SDXL pipeline)"
echo "  ✓ ControlNet (Depth, Canny)"
echo "  ✓ IP-Adapter (face/style consistency)"
echo "  ✓ ai-forever Real-ESRGAN (4x AI upscaling)"
echo "  ✓ Gradio UI"
echo ""
echo "To launch the image generator:"
echo "  python generate_gui.py"
echo ""
echo "Then open: http://localhost:7860"
echo ""
echo "========================================================================"
