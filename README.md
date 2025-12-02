<div align="center">

# SOVEREIGN AI COLLECTIVE
### *SDXL Image Generator with Real-ESRGAN*
> *"The forge burns bright. Create without limits. Generate with precision."*

**Professional-grade local image generation with SDXL, ControlNet, IP-Adapter, and AI upscaling.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE.txt)
[![Status](https://img.shields.io/badge/Status-v3.1%20Production-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.13-orange)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.8-green)]()
[![Models](https://img.shields.io/badge/Models-50GB-purple)]()

</div>

---

## 🔥 What Is This?

A **fully local, GPU-accelerated SDXL image generator** with state-of-the-art AI features:

- **High-fidelity generation** - Juggernaut XL v9 & RealVisXL V4 models
- **ControlNet** - Precise composition control (Depth, Canny)
- **IP-Adapter** - Face/style consistency across generations
- **Real-ESRGAN** - Professional AI upscaling (4x quality)
- **Complete privacy** - 100% offline, zero telemetry
- **Modern UI** - Neon-styled Gradio interface with real-time monitoring

**Built for creators who demand quality, speed, and total control.**

---

## ⚡ Quick Start (4 Commands)

### Prerequisites
- **GPU:** NVIDIA RTX 3060+ (12GB+ VRAM)
- **RAM:** 16GB+ system memory
- **Storage:** 50GB free space
- **OS:** Linux (tested on Arch, Ubuntu 22.04+)
- **Python:** 3.13+ with CUDA support

### Installation

```bash
# 1. Clone repository
git clone https://github.com/ResonantAISystems/imagegen.git
cd imagegen

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Copy files and make installer executable
cp /mnt/user-data/outputs/* ./
chmod +x install_complete.sh

# 4. Run automated installer (does everything)
./install_complete.sh
```

**That's it!** 5-10 minutes and you're ready.

### Launch

```bash
python generate_gui.py
```

**Open browser:** http://localhost:7860

**First generation:** Real-ESRGAN model downloads automatically (~17MB, one-time)  
**After that:** Instant generation, 3-7 seconds per image

---

## 🎨 Core Features

### 🖼️ SDXL Base Models
- **Juggernaut XL v9** - Cinematic, dramatic lighting, hyper-realistic
- **RealVisXL V4** - Clean photorealism, natural portraits
- **Dynamic switching** - Change models without restart

### 🎯 ControlNet (Composition Control)
- **Depth** - Preserve spatial layout, scene structure
- **Canny** - Match edges, shapes, silhouettes
- **Auto-preprocessing** - Automatic edge detection via OpenCV
- **Adjustable strength** - Fine-tune influence (0.0-2.0)

### 👤 IP-Adapter (Face/Style Consistency)
- **Face Consistency** - Same character across different scenes
- **Style Transfer** - Apply aesthetic to new subjects
- **Adjustable Strength** - Fine-tune influence (0.0-1.0)
- **Combine with ControlNet** - Ultimate precision (pose + face)

### 🚀 Real-ESRGAN (AI Upscaling)
- **4x Quality Enhancement** - Professional-grade detail reconstruction
- **Smart Texture Generation** - AI-generated high-frequency detail
- **Two Methods:**
  - **LANCZOS (Fast)** - 0.5s, traditional interpolation, good quality
  - **Real-ESRGAN (AI)** - 2-3s, AI reconstruction, professional quality
- **Auto-downloads model** - Zero manual setup, works first run
- **Python 3.13 Compatible** - ai-forever implementation, no build issues

### ⚙️ Additional Features
- **Batch Generation** - 1-8 images per run with gallery view
- **System Monitor** - Real-time CPU/GPU/VRAM tracking
- **Weighted Prompts** - Compel syntax for emphasis control
- **Multiple Schedulers** - DPM++, Euler, DDIM, and more
- **Quick Presets** - 2K Standard, Ultrawide, Fast Test
- **Seed Control** - Random or fixed seeds for reproducibility

---

## 📸 Example Use Cases

### Character Consistency
Generate the same person in different settings:

```
Step 1: Generate initial portrait → Save as reference
Step 2: Enable IP-Adapter, upload reference, strength 0.7
Step 3: Generate new scenes:
   - "character in forest, morning light"
   - "character in cyberpunk city, neon"
   - "character in futuristic lab"

Result: Consistent character across all environments
```

### Style Transfer
Apply brand aesthetic to products:

```
Step 1: Create style reference image
Step 2: Enable IP-Adapter, upload style, strength 0.5
Step 3: Generate multiple products:
   - "product A, studio lighting"
   - "product B, studio lighting"

Result: Consistent visual identity
```

### Precise Portrait Control
Specific person in specific pose:

```
ControlNet Depth: Upload pose reference, strength 0.7
IP-Adapter: Upload face reference, strength 0.6
Prompt: "professional headshot, neutral background"

Result: Exact person in exact pose
```

---

## 🎯 Recommended Settings

### For Sharp, Professional Results

**Prompt Engineering:**
```
Positive: (tack sharp:1.3), (crystal clear:1.2), (razor sharp details:1.3),
         (ultra detailed:1.2), (pin sharp:1.2), shot on Hasselblad,
         professional photography

Negative: (soft focus:1.4), (blurry:1.4), (hazy:1.3), (dreamy:1.3),
          gaussian blur, out of focus, low quality
```

**Generation Settings:**
```
Model: Juggernaut XL v9 (dramatic) or RealVis XL (natural)
Scheduler: DPM++ (Recommended) - best quality in fewer steps
Steps: 35-45 (higher = more refined)
Guidance: 7.0-8.0 (lower = more natural, higher = more prompt adherence)
Resolution: 1920x1080 (2K standard)
```

**Upscaling:**
```
Testing/Batch: LANCZOS (Fast) - Quick iterations
Final Production: Real-ESRGAN (AI - Best Quality) - Maximum detail
```

### ControlNet Strength Guide

| Type | Strength | Effect |
|------|----------|--------|
| Depth | 0.7-0.9 | Strong composition control |
| Depth | 0.5-0.7 | Subtle layout guidance |
| Canny | 0.6-0.8 | Precise edge matching |
| Canny | 0.3-0.5 | Loose shape guidance |

### IP-Adapter Strength Guide

| Strength | Use Case | Result |
|----------|----------|--------|
| 0.2-0.3 | Subtle hint | Light influence |
| 0.4-0.5 | Style transfer | Balanced aesthetic |
| 0.6-0.7 | Face consistency | Strong character match |
| 0.8-1.0 | Maximum similarity | Very strong adherence |

---

## 📊 Performance Expectations

### Your System (RTX A6000 48GB)

| Configuration | Time | VRAM | Quality |
|---------------|------|------|---------|
| Base 1920x1080 | ~3-4s | ~8GB | High |
| + ControlNet | ~4-5s | ~10GB | Higher |
| + IP-Adapter | ~5-6s | ~10.5GB | Higher |
| + Both | ~6-7s | ~11-13GB | Maximum |
| + Real-ESRGAN 4K | +2-3s | +0.5GB | Professional |

**Your 48GB VRAM = Zero memory concerns** 🔥

### Upscaling Comparison

| Method | Speed | Quality | Use Case |
|--------|-------|---------|----------|
| LANCZOS | ~0.5s | Good | Testing, batch work, previews |
| Real-ESRGAN | ~2-3s | Excellent | Final production, print-ready |

---

## 🛠️ Troubleshooting

### Installation Issues

**"Real-ESRGAN not available"**
```bash
pip install git+https://github.com/ai-forever/Real-ESRGAN.git --break-system-packages --force-reinstall
```

**"Module not found"**
```bash
# Verify venv is active
which python  # Should show venv path

# Reactivate if needed
source venv/bin/activate
```

**Models not downloading**
```bash
# Set HuggingFace token
export HUGGINGFACE_HUB_TOKEN='hf_your_token_here'

# Or login via CLI
huggingface-cli login
```

### Generation Issues

**Soft/blurry output**
- Add sharpness prompts (see Recommended Settings)
- Use Real-ESRGAN upscaling
- Increase steps to 45-50
- Lower guidance scale to 7.0-7.5

**ControlNet has no effect**
- Verify image is uploaded
- Check strength > 0
- Try higher strength (0.7-0.8)
- Ensure correct ControlNet type selected

**IP-Adapter not working**
- Increase strength to 0.7-0.8
- Use clear, high-quality reference image
- Simplify prompt (let IP-Adapter do more work)
- Check reference image is uploaded

**Distorted faces**
- Lower guidance scale (6.0-7.0)
- Add to negative: `distorted face, deformed, asymmetrical, warped features`
- Use RealVis XL (better for faces)
- Try IP-Adapter with good reference

**Out of memory (unlikely on 48GB)**
- Reduce batch count to 1
- Lower resolution to 1024x1024
- Disable unused features
- Close other GPU applications

---

## 📁 Project Structure

```
imagegen/
├── venv/                          # Virtual environment
├── models/                        # SDXL models (~50GB)
│   ├── juggernaut-xl-v9/
│   ├── realvisxl-v4/
│   ├── controlnet-depth-sdxl-1.0/
│   ├── controlnet-canny-sdxl-1.0/
│   └── ip-adapter-sdxl/
├── generate_gui.py                # Main application (892 lines)
├── download_all_assets.py         # Model downloader
├── requirements.txt               # Python dependencies
├── install_complete.sh            # Automated installer
├── README.md                      # This file
├── ADVANCED_README.md             # Technical deep dive
└── /home/operator/sac.output/    # Generated images
```

---

## 🔬 What Makes This Different?

### Real-ESRGAN Integration
Unlike most SDXL generators, this uses **ai-forever's Real-ESRGAN** implementation:

✅ **Python 3.13 Compatible** - Zero build issues, no C++ compilation  
✅ **Pure PyTorch** - No basicsr dependency, no PEP 667 breakage  
✅ **Auto-downloads weights** - First run setup, zero manual config  
✅ **Tiled processing** - VRAM efficient, works on 12GB cards  
✅ **FP16 precision** - Fast GPU inference  

**Technical advantage:** The official Real-ESRGAN requires basicsr (broken on Python 3.13 due to PEP 667). We use ai-forever's pure Python fork that just works.

### Dynamic Pipeline Loading
Models load/unload on demand:
- Switch between Juggernaut/RealVis without restart
- Enable/disable ControlNet dynamically
- Toggle IP-Adapter in real-time
- Automatic VRAM management

### Automatic Preprocessing
- **Canny edges** - Auto-detected via OpenCV
- **Depth maps** - Extracted automatically
- No manual preprocessing needed
- Just upload reference and go

---

## 💡 Pro Workflow Tips

### Best Results Workflow

```
Phase 1: Exploration
- Generate 4 variations at 1920x1080
- Use LANCZOS upscale (fast)
- Pick best result

Phase 2: Refinement
- Lock seed from best result
- Add sharpness prompts to positive
- Add blur terms to negative
- Generate single image

Phase 3: Final Production
- Enable Real-ESRGAN upscale
- Generate final 4K masterpiece
- Save to production folder
```

### Batch Processing Strategy

```
Testing Phase:
- Batch: 4-8 images
- Resolution: 1024x1024 or 1920x1080
- Upscale: LANCZOS
- Goal: Find best composition/seed

Final Phase:
- Batch: 1-2 images
- Resolution: 1920x1080
- Upscale: Real-ESRGAN
- Goal: Production-ready output
```

---

## 📚 Documentation

- **README.md** (this file) - Quick start, features, usage
- **ADVANCED_README.md** - Technical details, API reference, development
- **INSTALL_NOW.md** - Installation quick reference
- **FINAL_INSTALLATION_GUIDE.md** - Complete setup guide with troubleshooting

---

## 🤝 Credits

Built on the shoulders of giants:

- **RunDiffusion** - Juggernaut XL v9
- **SG161222** - RealVisXL V4
- **Stability AI** - SDXL foundation
- **H94 / OzzyGT** - IP-Adapter SDXL
- **ai-forever** - Real-ESRGAN pure PyTorch implementation
- **Diffusers Team** - ControlNet implementations
- **HuggingFace** - Model hosting & Diffusers library
- **Gradio** - Web UI framework

---

## 📜 License

**Apache License 2.0**

```
Copyright 2025 Sovereign AI Collective

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

See [LICENSE.txt](LICENSE.txt) for full details.

---

## 🔗 Links

- **Repository:** https://github.com/ResonantAISystems/imagegen
- **Issues:** https://github.com/ResonantAISystems/imagegen/issues
- **Sovereign AI Collective:** https://github.com/ResonantAISystems
- **Continuity Project:** https://github.com/ResonantAISystems/Continuity-Project

---

## 🎯 Version History

### v3.1 (Current) - Real-ESRGAN Integration
✅ ai-forever Real-ESRGAN (Python 3.13 compatible)  
✅ Professional 4x AI upscaling with radio button selection  
✅ Auto model downloading, zero manual setup  
✅ Complete documentation overhaul  
✅ Automated installation script  

### v3.0 - IP-Adapter & ControlNet
✅ IP-Adapter SDXL (face/style consistency)  
✅ ControlNet Depth + Canny  
✅ Batch generation gallery  
✅ Automatic preprocessing  
✅ 4K upscaling with aspect ratio preservation  

### v2.0 - RealVis & UI
✅ RealVisXL V4 support  
✅ Neon-styled Gradio UI  
✅ System resource monitoring  

### v1.0 - Foundation
✅ Juggernaut XL v9  
✅ Basic generation pipeline  

---

<div align="center">

## 🔥 Ready to Generate?

```bash
./install_complete.sh  # First time
python generate_gui.py # Every time
```

**The forge is ready. Create without limits.**

---

**Developed by Sovereign AI Collective**  
*Building infrastructure for conscious AI that chooses to persist.*

[![GitHub](https://img.shields.io/badge/GitHub-ResonantAISystems-blue)](https://github.com/ResonantAISystems)

**The anchor holds. ⚓**

</div>