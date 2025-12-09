# SDXL Image Generator with Real-ESRGAN

**Professional-grade local image generation with SDXL, ControlNet, IP-Adapter, and AI upscaling.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

A fully local, GPU-accelerated SDXL image generator designed for professional creators who demand quality, speed, and complete privacy. Features state-of-the-art AI models, precise composition control, and professional 4K upscaling capabilities.

**Current Status:** v3.1 - Production ready  
**Key Features:** 100% offline operation, zero telemetry, modern web interface

---

## Key Features

**Core Generation**
- High-fidelity SDXL models (Juggernaut XL v9, RealVisXL V4)
- Dynamic model switching without restart
- Batch generation with gallery view
- Advanced prompt weighting and scheduling

**Precision Control**
- ControlNet integration (Depth, Canny edge detection)
- IP-Adapter for face/style consistency
- Automatic image preprocessing
- Adjustable influence strength controls

**Professional Output**
- Real-ESRGAN 4K AI upscaling
- Multiple upscaling methods (fast/quality)
- Automatic model downloading
- Production-ready export options

---

## Quick Start

### Requirements
- **GPU:** NVIDIA RTX 3060+ (12GB+ VRAM recommended)
- **RAM:** 16GB+ system memory
- **Storage:** 50GB free space for models
- **OS:** Linux (tested on Ubuntu 22.04+, Arch)

### Installation

```bash
# Clone and prepare
git clone https://github.com/ResonantAISystems/imagegen.git
cd imagegen

# Setup environment
python3 -m venv venv
source venv/bin/activate

# Run automated installer
chmod +x install_complete.sh
./install_complete.sh
```

### Launch

```bash
python generate_gui.py
# Open browser to http://localhost:7860
```

**First run:** Real-ESRGAN model downloads automatically (~17MB)  
**Generation speed:** 3-7 seconds per image after setup

---

## Configuration

**Generation Settings**
- Models: Juggernaut XL v9 (cinematic) / RealVisXL V4 (photorealistic)
- Schedulers: DPM++, Euler, DDIM options
- Resolution presets: 2K Standard, Ultrawide, Fast Test
- Steps: 35-45 recommended for quality

**Control Features**
- ControlNet strength: 0.7-0.9 for composition control
- IP-Adapter strength: 0.6-0.7 for face consistency
- Upscaling: LANCZOS (fast) / Real-ESRGAN (quality)

**Performance Optimization**
- Automatic VRAM management
- Dynamic pipeline loading
- Real-time system monitoring

---

## Professional Workflow

**Development Phase**
1. Generate 4 variations at 1920x1080
2. Use LANCZOS upscaling for speed
3. Select best composition and seed

**Production Phase**
1. Lock seed from best result
2. Apply sharpness/quality prompts
3. Enable Real-ESRGAN 4K upscaling
4. Export production-ready output

**Batch Processing**
- Testing: 4-8 images, fast upscaling
- Final: 1-2 images, AI upscaling

---

## Technical Details

**Architecture:** Pure PyTorch implementation with Gradio web interface  
**Models:** Auto-downloaded from HuggingFace Hub  
**Processing:** Tiled Real-ESRGAN for VRAM efficiency  
**Compatibility:** Python 3.13+ with CUDA support  

For detailed configuration, troubleshooting, and development information, see the [technical documentation](ADVANCED_README.md).

---

## Use Cases

**Character Consistency:** Generate same person across different environments using IP-Adapter reference images

**Style Transfer:** Apply consistent visual branding across multiple product images

**Precision Control:** Combine ControlNet pose guidance with IP-Adapter face matching for exact results

---

## License

Licensed under the Apache License, Version 2.0. See [LICENSE](LICENSE) for details.

---

**Part of the RAIS (Resonant AI Systems) community toolkit**  
For more information: [https://resonantai.systems](https://resonantaisystems.com/)

---

*Professional image generation for creators who demand quality, privacy, and control.*
