# ✨ Sovereign AI Collective — SDXL Image Generator  
### High-fidelity local image generation with Juggernaut XL, RealVis XL, ControlNet, and IP-Adapter

This repository contains a fully local, GPU-accelerated SDXL generator featuring a neon-themed Gradio UI and advanced model controls. Build stunning, photorealistic images using industry-leading techniques such as ControlNet (Depth/Canny) and IP-Adapter face/style conditioning.

---

# 📖 Table of Contents
- [🚀 Features](#-features)
- [📦 Installation & Setup](#-installation--setup)
- [🧩 Model Downloads](#-model-downloads)
- [▶️ Running the GUI](#️-running-the-gui)
- [📁 Project Structure](#-project-structure)
- [🧠 Usage Tips](#-usage-tips)
- [🐛 Troubleshooting](#-troubleshooting)
- [❤️ Credits](#️-credits)

---

# 🚀 Features

### ✔ **State-of-the-Art SDXL Models**
- **Juggernaut XL v9** — best-in-class realism  
- **RealVis XL v4** — ultra-clean portrait & environment rendering  

### ✔ **ControlNet SDXL Integration**
- **Depth** → preserves layout & composition  
- **Canny** → preserves edges & shapes  

### ✔ **IP-Adapter (SDXL + H94)**
- Face reference  
- Style reference  
- Appearance-locked subject consistency  

### ✔ **Dynamic Model Switching**
Swap base models, enable/disable ControlNet, change strengths, all live in-UI.

### ✔ **Neon Cyber-Aesthetic GUI**
Cyan/Magenta on dark grey theme, optimized for clarity & comfort.

### ✔ **No Cloud, No Telemetry**
All inference is **100% local**, GPU only.

---

# 📦 Installation & Setup

## 1. Clone the Repository

```bash
git clone https://github.com/ResonantAISystems/playtime.git
cd playtime
2. Create & Activate a Virtual Environment
bash
Copy code
python3 -m venv venv
source venv/bin/activate
3. Install Dependencies
bash
Copy code
pip install -r requirements.txt
If missing, generate a requirements.txt:

bash
Copy code
pip freeze > requirements.txt
🧩 Model Downloads
All large model files are downloaded automatically:

bash
Copy code
python download_all_assets.py
This fetches:

Juggernaut XL v9

RealVisXL v4

RealVisXL-Inpaint

ControlNet Depth SDXL

ControlNet Canny SDXL

IP-Adapter SDXL

IP-Adapter h94

🔐 If any model requires HuggingFace authentication:
bash
Copy code
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"
Or log in interactively:

bash
Copy code
huggingface-cli login
▶️ Running the GUI
Once the virtual environment is activated:

bash
Copy code
python generate_gui.py
The Gradio interface will be available at:

cpp
Copy code
http://0.0.0.0:7860
Features include:

Positive/Negative prompt fields

Base model selector

ControlNet type selector

Control image upload

Control strength slider

Scheduler settings (DPM++, Euler, DDIM, LMS)

Resolution controls

Batch rendering

4K Upscaler

System monitor

Output preview

📁 Project Structure
text
Copy code
playtime/
│
├── generate_gui.py              # Main SDXL GUI (Juggernaut/RealVis + ControlNet + IP-Adapter)
├── download_all_assets.py       # Automated model downloader
├── generate_ultra.py            # Optional CLI generator
├── generate_enhanced.py         # Optional CLI generator
│
├── models/                      # <-- NOT tracked by Git (see .gitignore)
│   ├── juggernaut-xl-v9/
│   ├── realvisxl-v4/
│   ├── realvisxl-v4-inpaint/
│   ├── controlnet-depth-sdxl-1.0/
│   ├── controlnet-canny-sdxl-1.0/
│   ├── ip-adapter-sdxl/
│   └── ip-adapter-h94/
│
├── requirements.txt
└── README.md
🧠 Usage Tips
Best SDXL Settings
makefile
Copy code
Steps: 35–55  
CFG (Guidance): 5.5–7.0  
Scheduler: DPM++ (Recommended)
ControlNet Strength
Depth: 0.6–0.9

Canny: 0.4–0.7

For consistent faces
Use IP-Adapter with a clean reference portrait.

For stable environments
Use Depth ControlNet with the original scene as the control image.

For fast debugging
Use a fixed seed such as:

Copy code
12345
🐛 Troubleshooting
“ControlNet isn’t doing anything”
Make sure a control image is loaded and strength > 0.

“My face looks distorted”
Increase steps, reduce CFG, and add these to negative prompt:

nginx
Copy code
distorted face, merged face, warped facial features, incorrect anatomy
“Models didn’t download”
Ensure your HuggingFace token is set:

bash
Copy code
export HUGGINGFACE_HUB_TOKEN="hf_xxx"
❤️ Credits
This project utilizes technologies from:

RunDiffusion — Juggernaut XL v9

SG161222 / RealVisXL

StabilityAI — SDXL

H94 / IP-Adapter

HuggingFace Diffusers

Gradio

Developed by Sovereign AI Collective.