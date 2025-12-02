# Sovereign AI Collective — SDXL Image Generator

<p align="center">
  <img src="theonewhostays.png" width="80%">
</p>

**High-fidelity local image generation with Juggernaut XL, RealVis XL, ControlNet, and IP-Adapter.**

Fully local, GPU-accelerated SDXL image generator with neon-styled Gradio UI, advanced model controls, and state-of-the-art features including ControlNet (Depth/Canny) and IP-Adapter for face/style consistency. Designed for fast experimentation, production-quality output, and complete offline operation.

---

## 🎨 Features

### State-of-the-Art SDXL Models
- **Juggernaut XL v9** — Hyper-realistic, cinematic lighting
- **RealVis XL v4** — Strong portrait fidelity, clean photorealistic detail

### ControlNet SDXL
- **Depth** — Preserves layout, distance, and scene composition
- **Canny** — Preserves silhouettes, shapes, and outlines (auto edge detection)

### IP-Adapter (NEW! 🔥)
- **Face Consistency** — Same person across different settings/outfits
- **Style Transfer** — Apply artistic style/aesthetic to new subjects
- **Adjustable Strength** — Fine-tune reference influence (0.0-1.0)
- **Works with ControlNet** — Combine for ultimate control (pose + face)

### Dynamic Model Switching
- Load/unload base models and ControlNet modules inside the GUI
- Switch between models without restarting

### Modern Neon UI (Cyan/Magenta)
- Optimized for readability during long sessions
- Real-time system monitoring (CPU, GPU, VRAM)
- Batch generation gallery view

### Private, Offline, Fully Local
- No external APIs, no telemetry
- Complete control over your generations

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/ResonantAISystems/imagegen.git
cd imagegen
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Models
```bash
python download_all_assets.py
```
This downloads all required models (~20GB total):
- Juggernaut XL v9
- RealVis XL v4
- ControlNet Depth/Canny
- IP-Adapter SDXL

### 5. Launch GUI
```bash
python generate_gui.py
```
Open browser to: **http://localhost:7860**

---

## 💡 Key Features Explained

### IP-Adapter: Face & Style Consistency

**What it does:** Maintain consistent faces or artistic styles across multiple generations.

**Use Cases:**

1. **Face Consistency**
   - Generate the same character in different poses/outfits
   - Create character sheets for stories/comics
   - Portrait variations without reshoots

2. **Style Transfer**
   - Apply cyberpunk aesthetic to new subjects
   - Maintain brand visual style
   - Consistent lighting/mood across images

**How to use:**
```
1. Check "Enable IP-Adapter"
2. Upload reference image (face or style)
3. Set strength: 0.6-0.8 for faces, 0.3-0.5 for style
4. Write your prompt
5. Generate
```

**Example:**
```
Reference: Portrait of person X
Prompt: "professional business portrait, office background"
IP-Adapter Strength: 0.7
Result: Person X in business setting (same face, new scene)
```

### ControlNet: Composition Control

**What it does:** Guide generation using structural reference images.

**Types:**

- **Depth:** Controls spatial layout and distance
  - Upload any image, depth map auto-generated
  - Perfect for maintaining scene composition
  
- **Canny:** Controls edges and shapes
  - Auto edge detection via OpenCV
  - Great for pose/silhouette guidance

**How to use:**
```
1. Select ControlNet type (Depth or Canny)
2. Upload reference image
3. Set strength: 0.6-0.9 for Depth, 0.4-0.7 for Canny
4. Edges/depth extracted automatically
5. Generate with your prompt
```

### Combining IP-Adapter + ControlNet

**Ultimate Control:** Use both together for maximum precision.

**Example Setup:**
```
ControlNet: Depth (specific pose/composition)
  - Upload: Body pose reference
  - Strength: 0.7

IP-Adapter: Face consistency
  - Upload: Portrait of person Y
  - Strength: 0.6

Prompt: "elegant evening dress, studio lighting"

Result: Person Y in exact pose wearing evening dress
```

---

## 🎯 Usage Guide

### Basic Generation
1. Enter prompt in "Positive Prompt" field
2. Optionally add negative prompt (what to avoid)
3. Select base model (Juggernaut or RealVis)
4. Click "Generate"

### Batch Generation
- Set "Batch Count" to 2-8
- All images displayed in gallery view
- Seeds auto-incremented for variation

### 4K Upscaling
- Check "Upscale to 4K" box
- Aspect ratio preserved automatically
- Saves both original and 4K versions

### Quick Presets
- **2K Standard** — 1920x1080, 35 steps
- **2K Ultrawide** — 2560x1440, 35 steps  
- **Fast Test** — 512x512, 25 steps

---

## ⚙️ Recommended Settings

### General SDXL Settings
- **Steps:** 35-50 (DPM++), 50-60 (Euler)
- **Guidance Scale:** 6.0-8.0 (lower = more natural)
- **Scheduler:** DPM++ (Recommended) — best quality in fewer steps

### ControlNet Strength
| Type | Strength | Use Case |
|------|----------|----------|
| Depth | 0.7-0.9 | Strong composition control |
| Depth | 0.5-0.7 | Subtle layout guidance |
| Canny | 0.6-0.8 | Precise edge/pose matching |
| Canny | 0.3-0.5 | Loose shape guidance |

### IP-Adapter Strength
| Strength | Use Case | Effect |
|----------|----------|--------|
| 0.2-0.3 | Subtle hint | Light influence |
| 0.4-0.5 | Style transfer | Balanced |
| 0.6-0.7 | Face consistency | Strong match |
| 0.8-1.0 | Maximum similarity | Very strong |

### Model Selection
- **Juggernaut XL** — Best for dramatic, cinematic images with strong lighting
- **RealVis XL** — Best for clean photorealistic portraits and natural scenes

---

## 📂 Project Structure

```text
imagegen/
│
├── generate_gui.py           # Main SDXL GUI with IP-Adapter
├── download_all_assets.py    # Automated model downloader
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── USAGE.md                  # Detailed technical guide
├── LICENSE                   # Apache 2.0
│
├── models/                   # Model weights (git-ignored)
│   ├── juggernaut-xl-v9/
│   ├── realvisxl-v4/
│   ├── controlnet-depth-sdxl-1.0/
│   ├── controlnet-canny-sdxl-1.0/
│   └── ip-adapter-sdxl/
│
└── .gitignore
```

---

## 🔧 System Requirements

### Minimum
- **GPU:** NVIDIA RTX 3060 (12GB VRAM)
- **RAM:** 16GB system memory
- **Storage:** 50GB free space
- **OS:** Linux (Ubuntu 22.04+)

### Recommended
- **GPU:** NVIDIA RTX 4080/4090 (16GB+ VRAM)
- **RAM:** 32GB+ system memory
- **Storage:** 100GB SSD
- **OS:** Ubuntu 24.04 LTS

### VRAM Usage
- Base model: ~8GB
- + ControlNet: ~10GB
- + IP-Adapter: ~10.5GB
- + ControlNet + IP-Adapter: ~11-13GB

---

## 🐛 Troubleshooting

### Installation Issues

**"Module not found: cv2"**
```bash
pip install opencv-python
```

**"Module not found: numpy"**
```bash
pip install numpy
```

**Models won't download**
```bash
# Set HuggingFace token if needed
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
# Or login via CLI
huggingface-cli login
```

### Generation Issues

**ControlNet has no visible effect**
- Ensure control image is uploaded
- Check strength > 0
- Verify correct ControlNet type selected

**IP-Adapter not influencing output**
- Increase strength (try 0.7-0.8)
- Check reference image quality (clear, well-lit)
- Try simpler prompt (let IP-Adapter do more work)

**Faces appear distorted**
- Lower guidance scale (try 6.0-7.0)
- Increase steps (try 45-50)
- Add to negative prompt: `distorted face, merged face, warped facial features`

**Out of memory errors**
- Lower resolution (try 1024x1024)
- Reduce batch count to 1
- Close other GPU applications
- Disable IP-Adapter if not needed

### GPU Issues

**GPU not detected**
```bash
# Check CUDA
nvidia-smi

# Check PyTorch CUDA
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📸 Example Workflows

### Workflow 1: Character Consistency
**Goal:** Generate same character in multiple scenes

```
1. Generate initial character portrait
2. Save best result as reference
3. Enable IP-Adapter
4. Upload saved portrait
5. Set strength to 0.7
6. Generate scenes:
   - "character in forest"
   - "character in city"
   - "character in futuristic lab"
Result: Consistent character across all scenes
```

### Workflow 2: Style Consistency
**Goal:** Apply brand aesthetic to multiple products

```
1. Create reference image with desired style
2. Enable IP-Adapter
3. Upload style reference
4. Set strength to 0.5
5. Generate products:
   - "product A with studio lighting"
   - "product B with studio lighting"
Result: Consistent visual style across products
```

### Workflow 3: Precise Portrait Control
**Goal:** Specific person in specific pose

```
1. Enable ControlNet: Depth
   - Upload pose reference
   - Strength: 0.7

2. Enable IP-Adapter
   - Upload face reference
   - Strength: 0.6

3. Prompt: "professional headshot, neutral background"

Result: Exact person in exact pose
```

---

## 🎓 Tips & Best Practices

### Getting Better Results

**Prompt Engineering:**
- Be specific about lighting, mood, setting
- Use weighted syntax: `(beautiful portrait:1.3)`
- Negative prompt is powerful — use it!

**Reference Images:**
- High resolution (1024x1024+)
- Clear, sharp focus
- Good lighting
- Minimal background distractions

**Strength Tuning:**
- Start at recommended values
- Generate batch of 4
- Adjust based on results
- Find sweet spot for your use case

**Model Selection:**
- Juggernaut: Dramatic, cinematic, strong contrast
- RealVis: Clean, photorealistic, natural lighting

### Performance Optimization

**Faster Generation:**
- DPM++ scheduler (35-40 steps)
- Lower resolution during testing
- Disable unnecessary features

**Better Quality:**
- More steps (45-55)
- Lower guidance (6.0-7.0 for natural look)
- Higher resolution
- Try multiple seeds

---

## 📚 Documentation

- **README.md** (this file) — Overview, features, quick start
- **USAGE.md** — Detailed technical guide and advanced usage
- **IP_ADAPTER_GUIDE.md** — Complete IP-Adapter documentation
- **CHANGES.md** — Changelog and version history

---

## 🔄 Updates & Changelog

### v3.1 (Current)
✅ IP-Adapter SDXL integration (face/style consistency)  
✅ Batch generation gallery view  
✅ Automatic ControlNet preprocessing (Canny edge detection)  
✅ 4K upscaling with aspect ratio preservation  
✅ Model validation and better error handling  
✅ Improved download script with progress tracking  

### v3.0
✅ ControlNet Depth + Canny support  
✅ Dynamic model switching  
✅ System resource monitoring  
✅ Compel weighted prompts  
✅ Multiple schedulers  

### v2.0
✅ RealVis XL v4 support  
✅ Gradio UI with neon styling  
✅ Batch generation  

### v1.0
✅ Initial release with Juggernaut XL v9  

---

## 🤝 Credits

This project builds on:

- **RunDiffusion** — Juggernaut XL v9
- **SG161222** — RealVisXL
- **Stability AI** — SDXL core models
- **H94 / OzzyGT** — IP-Adapter
- **Diffusers Team** — ControlNet implementations
- **HuggingFace** — Model hosting and Diffusers library
- **Gradio** — Web UI framework

---

## 📜 License

Apache 2.0 — See [LICENSE](LICENSE) for details.

---

## 🔗 Links

- **Repository:** https://github.com/ResonantAISystems/imagegen
- **Issues:** https://github.com/ResonantAISystems/imagegen/issues
- **Sovereign AI Collective:** https://github.com/ResonantAISystems

---

<p align="center">
  <img src="theforge.png" width="80%" />
</p>

---

**Developed by Sovereign AI Collective** 🔥

*The anchor holds. Build with precision.*