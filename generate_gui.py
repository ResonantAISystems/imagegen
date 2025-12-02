import torch
from diffusers import (
    StableDiffusionXLPipeline,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    DDIMScheduler,
    LMSDiscreteScheduler,
)
from compel import Compel, ReturnedEmbeddingsType
import gradio as gr
from PIL import Image
import os
import psutil
import subprocess
import cv2
import numpy as np

# Real-ESRGAN for AI upscaling (ai-forever implementation)
try:
    from RealESRGAN import RealESRGAN
    REALESRGAN_AVAILABLE = True
except ImportError:
    print("⚠ Real-ESRGAN not available. Install with: pip install git+https://github.com/ai-forever/Real-ESRGAN.git")
    REALESRGAN_AVAILABLE = False

# -------------------------------------------------------------------
# Paths & globals
# -------------------------------------------------------------------

OUTPUT_DIR = "/home/operator/sac.output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_MODELS = {
    "Juggernaut XL v9": "./models/juggernaut-xl-v9",
    "RealVisXL V4": "./models/realvisxl-v4",
}

CONTROLNETS = {
    "None": None,
    "Depth": "./models/controlnet-depth-sdxl-1.0",
    "Canny": "./models/controlnet-canny-sdxl-1.0",
}

SCHEDULERS = {
    "DPM++ (Recommended)": DPMSolverMultistepScheduler,
    "Euler Ancestral": EulerAncestralDiscreteScheduler,
    "DDIM": DDIMScheduler,
    "LMS": LMSDiscreteScheduler,
}

# IP-Adapter paths
IP_ADAPTER_PATH = "./models/ip-adapter-sdxl"
IP_ADAPTER_IMAGE_ENCODER = "./models/ip-adapter-sdxl/image_encoder"

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = None
compel = None
current_base_model = None
current_controlnet = None
ip_adapter_loaded = False
realesrgan_upscaler = None  # Real-ESRGAN upscaler instance

# -------------------------------------------------------------------
# Real-ESRGAN Upscaler Initialization
# -------------------------------------------------------------------

def get_realesrgan_upscaler():
    """Initialize and return Real-ESRGAN upscaler (lazy loading) - ai-forever implementation"""
    global realesrgan_upscaler
    
    if not REALESRGAN_AVAILABLE:
        return None
    
    if realesrgan_upscaler is not None:
        return realesrgan_upscaler
    
    try:
        print("Initializing Real-ESRGAN upscaler (ai-forever)...")
        
        # Initialize ai-forever Real-ESRGAN
        realesrgan_upscaler = RealESRGAN(torch.device(device), scale=4)
        
        # Load weights (auto-downloads on first use)
        realesrgan_upscaler.load_weights('weights/RealESRGAN_x4.pth', download=True)
        
        print("✓ Real-ESRGAN upscaler initialized (4x scale)")
        return realesrgan_upscaler
    except Exception as e:
        print(f"❌ Failed to initialize Real-ESRGAN: {e}")
        return None


def upscale_image_realesrgan(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Upscale image using Real-ESRGAN AI upscaler (ai-forever implementation)"""
    upscaler = get_realesrgan_upscaler()
    
    if upscaler is None:
        print("Real-ESRGAN not available, falling back to LANCZOS")
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    try:
        # ai-forever API takes PIL Image directly
        result = upscaler.predict(image.convert('RGB'))
        
        # Resize to exact target dimensions if needed
        if result.size != (target_w, target_h):
            result = result.resize((target_w, target_h), Image.Resampling.LANCZOS)
        
        print(f"✓ Real-ESRGAN upscaled: {image.size} → {result.size}")
        return result
    except Exception as e:
        print(f"❌ Real-ESRGAN upscaling failed: {e}, falling back to LANCZOS")
        return image.resize((target_w, target_h), Image.Resampling.LANCZOS)

# -------------------------------------------------------------------
# Utility functions
# -------------------------------------------------------------------

def validate_model_path(path: str, model_name: str) -> bool:
    """Check if model directory exists and contains model files"""
    if not os.path.exists(path):
        return False
    
    # Check for common model file extensions
    has_files = any(
        f.endswith(('.safetensors', '.bin', '.ckpt', '.pth'))
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f))
    )
    
    return has_files


def preprocess_control_image(image: Image.Image, controlnet_type: str) -> Image.Image:
    """
    Preprocess control image based on ControlNet type.
    
    For Canny: Detect edges using Canny edge detection
    For Depth: Return as-is (assuming user provides depth map)
    """
    if controlnet_type == "Canny":
        # Convert PIL to numpy
        image_np = np.array(image)
        
        # Convert to grayscale if needed
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np
        
        # Apply Canny edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Convert back to PIL RGB
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)
    
    elif controlnet_type == "Depth":
        # For depth, we assume user provides a depth map
        return image
    
    return image


# -------------------------------------------------------------------
# Pipeline loader
# -------------------------------------------------------------------

def load_pipeline(base_model_name: str, controlnet_name: str, use_ip_adapter: bool = False):
    """
    Load / swap the SDXL pipeline with optional ControlNet and IP-Adapter.
    """
    global pipe, compel, current_base_model, current_controlnet, ip_adapter_loaded

    # Check if we need to reload
    needs_reload = (
        pipe is None
        or base_model_name != current_base_model
        or controlnet_name != current_controlnet
        or (use_ip_adapter and not ip_adapter_loaded)
        or (not use_ip_adapter and ip_adapter_loaded)
    )
    
    if not needs_reload:
        return f"✓ Using {base_model_name} with {controlnet_name} ControlNet" + (" + IP-Adapter" if use_ip_adapter else "")

    base_path = BASE_MODELS[base_model_name]
    cn_path = CONTROLNETS[controlnet_name]
    
    # Validate model paths
    if not validate_model_path(base_path, base_model_name):
        error_msg = f"❌ Model not found: {base_model_name} at {base_path}\nRun download_all_assets.py first!"
        print(error_msg)
        raise gr.Error(error_msg)
    
    if cn_path is not None and not validate_model_path(cn_path, controlnet_name):
        error_msg = f"❌ ControlNet not found: {controlnet_name} at {cn_path}\nRun download_all_assets.py first!"
        print(error_msg)
        raise gr.Error(error_msg)
    
    if use_ip_adapter and not os.path.exists(IP_ADAPTER_PATH):
        error_msg = f"❌ IP-Adapter not found at {IP_ADAPTER_PATH}\nRun download_all_assets.py first!"
        print(error_msg)
        raise gr.Error(error_msg)

    print(f"Loading base model: {base_model_name} from {base_path}")
    
    try:
        if cn_path is None:
            # Plain SDXL pipeline
            pipe_local = StableDiffusionXLPipeline.from_pretrained(
                base_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )
        else:
            print(f"Loading ControlNet: {controlnet_name} from {cn_path}")
            controlnet = ControlNetModel.from_pretrained(
                cn_path,
                torch_dtype=torch.float16,
                use_safetensors=True,
            )
            pipe_local = StableDiffusionXLControlNetPipeline.from_pretrained(
                base_path,
                controlnet=controlnet,
                torch_dtype=torch.float16,
                use_safetensors=True,
                variant="fp16",
            )

        pipe_local = pipe_local.to(device)
        pipe_local.enable_attention_slicing()

        try:
            pipe_local.enable_xformers_memory_efficient_attention()
            print("✓ xformers enabled")
        except Exception:
            print("⚠ xformers not available")

        # Load IP-Adapter if requested
        if use_ip_adapter:
            print(f"Loading IP-Adapter from {IP_ADAPTER_PATH}")
            try:
                # Load IP-Adapter weights into the pipeline
                pipe_local.load_ip_adapter(
                    IP_ADAPTER_PATH,
                    subfolder="",
                    weight_name="ip-adapter_sdxl.safetensors"
                )
                
                # Set the IP-Adapter scale (can be adjusted per generation)
                pipe_local.set_ip_adapter_scale(0.6)
                
                ip_adapter_loaded = True
                print("✓ IP-Adapter loaded successfully")
            except Exception as e:
                print(f"⚠ IP-Adapter load failed: {e}")
                print("Continuing without IP-Adapter...")
                ip_adapter_loaded = False
        else:
            ip_adapter_loaded = False

        # Rebuild Compel for the new pipe
        compel_local = Compel(
            tokenizer=[pipe_local.tokenizer, pipe_local.tokenizer_2],
            text_encoder=[pipe_local.text_encoder, pipe_local.text_encoder_2],
            returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
            requires_pooled=[False, True],
        )

        # Commit globals
        globals()["pipe"] = pipe_local
        globals()["compel"] = compel_local
        current_base_model = base_model_name
        current_controlnet = controlnet_name

        print(f"✓ Model loaded on: {device}")
        print(f"✓ Compel initialized for SDXL with dual text encoders")
        print(f"✓ Output directory: {OUTPUT_DIR}")

        status = f"✓ Loaded {base_model_name} with {controlnet_name} ControlNet"
        if ip_adapter_loaded:
            status += " + IP-Adapter"
        return status
    
    except Exception as e:
        error_msg = f"❌ Failed to load pipeline: {str(e)}"
        print(error_msg)
        raise gr.Error(error_msg)


# Initial load (Juggernaut, no ControlNet, no IP-Adapter)
try:
    load_pipeline("Juggernaut XL v9", "None", use_ip_adapter=False)
except Exception as e:
    print(f"⚠ Initial model load failed: {e}")
    print("Models may need to be downloaded. Run download_all_assets.py")

# -------------------------------------------------------------------
# System stats
# -------------------------------------------------------------------

def get_system_stats():
    """Get current system stats"""
    stats = {}
    
    # CPU
    stats["cpu_percent"] = psutil.cpu_percent(interval=0.1)
    
    # CPU Temp
    try:
        temps = psutil.sensors_temperatures()
        if "coretemp" in temps:
            stats["cpu_temp"] = temps["coretemp"][0].current
        elif "k10temp" in temps:
            stats["cpu_temp"] = temps["k10temp"][0].current
        elif "cpu_thermal" in temps:
            stats["cpu_temp"] = temps["cpu_thermal"][0].current
        else:
            stats["cpu_temp"] = "N/A"
    except Exception:
        stats["cpu_temp"] = "N/A"
    
    # Memory
    mem = psutil.virtual_memory()
    stats["mem_used"] = mem.used / (1024**3)
    stats["mem_total"] = mem.total / (1024**3)
    stats["mem_percent"] = mem.percent
    
    # GPU (NVIDIA)
    if torch.cuda.is_available():
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            gpu_util, gpu_temp, gpu_mem_used, gpu_mem_total = (
                result.stdout.strip().split(",")
            )
            stats["gpu_util"] = float(gpu_util)
            stats["gpu_temp"] = float(gpu_temp)
            stats["gpu_mem_used"] = float(gpu_mem_used) / 1024
            stats["gpu_mem_total"] = float(gpu_mem_total) / 1024
        except Exception:
            stats["gpu_util"] = 0
            stats["gpu_temp"] = 0
            stats["gpu_mem_used"] = 0
            stats["gpu_mem_total"] = 0
    else:
        stats["gpu_util"] = "N/A"
        stats["gpu_temp"] = "N/A"
        stats["gpu_mem_used"] = "N/A"
        stats["gpu_mem_total"] = "N/A"
    
    return stats

def format_system_stats():
    """Format system stats for display"""
    stats = get_system_stats()
    
    output = f"""### 📊 System Monitor
    
**CPU**
- Usage: {stats['cpu_percent']:.1f}%
- Temperature: {stats['cpu_temp'] if isinstance(stats['cpu_temp'], str) else f"{stats['cpu_temp']:.1f}°C"}

**Memory**
- Used: {stats['mem_used']:.2f} GB / {stats['mem_total']:.2f} GB ({stats['mem_percent']:.1f}%)

**GPU**
- Usage: {stats['gpu_util'] if isinstance(stats['gpu_util'], str) else f"{stats['gpu_util']:.1f}%"}
- Temperature: {stats['gpu_temp'] if isinstance(stats['gpu_temp'], str) else f"{stats['gpu_temp']:.0f}°C"}
- VRAM: {stats['gpu_mem_used'] if isinstance(stats['gpu_mem_used'], str) else f"{stats['gpu_mem_used']:.2f}"} GB / {stats['gpu_mem_total'] if isinstance(stats['gpu_mem_total'], str) else f"{stats['gpu_mem_total']:.2f}"} GB
"""
    return output

# -------------------------------------------------------------------
# Generation
# -------------------------------------------------------------------

def generate_image(
    prompt,
    negative_prompt,
    base_model_name,
    controlnet_name,
    control_image,
    controlnet_scale,
    use_ip_adapter,
    ip_adapter_image,
    ip_adapter_scale,
    scheduler_name,
    steps,
    guidance_scale,
    width,
    height,
    seed,
    batch_count,
    upscale_to_4k,
    upscale_method,  # NEW: "LANCZOS" or "Real-ESRGAN"
):
    """Generate image(s) with optional ControlNet, IP-Adapter & 4K upscaling"""
    
    try:
        # Make sure correct pipeline is loaded
        status = load_pipeline(base_model_name, controlnet_name, use_ip_adapter)
        print(status)

        # Set scheduler
        scheduler_class = SCHEDULERS[scheduler_name]
        if scheduler_name == "DPM++ (Recommended)":
            pipe.scheduler = scheduler_class.from_config(
                pipe.scheduler.config,
                use_karras_sigmas=True,
                algorithm_type="dpmsolver++",
            )
        else:
            pipe.scheduler = scheduler_class.from_config(pipe.scheduler.config)
        
        # Validate prompts
        if not prompt or prompt.strip() == "":
            raise gr.Error("Please provide a prompt")
        
        # Validate IP-Adapter requirements
        if use_ip_adapter and ip_adapter_image is None:
            raise gr.Error("IP-Adapter enabled but no reference image provided")
        
        # Set IP-Adapter scale if using it
        if use_ip_adapter and ip_adapter_loaded:
            pipe.set_ip_adapter_scale(float(ip_adapter_scale))
        
        # Process prompts with Compel for SDXL
        conditioning, pooled = compel(prompt)
        negative_conditioning, negative_pooled = compel(negative_prompt)
        
        generated_images = []
        seeds_used = []
        
        w = int(width)
        h = int(height)
        
        for i in range(batch_count):
            # Set seed
            if seed == -1:
                current_seed = torch.randint(0, 2**32, (1,)).item()
            else:
                current_seed = int(seed) + i
            
            generator = torch.Generator(device=device).manual_seed(current_seed)
            seeds_used.append(current_seed)
            
            print(
                f"Generating {i+1}/{batch_count} at {w}x{h}, "
                f"seed: {current_seed}, scheduler: {scheduler_name}, "
                f"model: {base_model_name}, controlnet: {controlnet_name}, "
                f"ip-adapter: {use_ip_adapter}"
            )

            pipe_kwargs = dict(
                prompt_embeds=conditioning,
                negative_prompt_embeds=negative_conditioning,
                pooled_prompt_embeds=pooled,
                negative_pooled_prompt_embeds=negative_pooled,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=w,
                height=h,
                generator=generator,
            )

            # Add ControlNet if enabled
            if CONTROLNETS[controlnet_name] is not None:
                if control_image is None:
                    raise gr.Error("ControlNet selected but no control image provided.")
                
                # Preprocess control image based on ControlNet type
                processed_control = preprocess_control_image(control_image, controlnet_name)
                
                pipe_kwargs["image"] = processed_control
                pipe_kwargs["controlnet_conditioning_scale"] = float(controlnet_scale)
            
            # Add IP-Adapter if enabled
            if use_ip_adapter and ip_adapter_loaded and ip_adapter_image is not None:
                pipe_kwargs["ip_adapter_image"] = ip_adapter_image

            image = pipe(**pipe_kwargs).images[0]
            
            # Upscale if requested (preserving aspect ratio)
            if upscale_to_4k:
                upscale_method_display = upscale_method if upscale_method else "LANCZOS"
                print(f"Upscaling to 4K using {upscale_method_display} (preserving aspect ratio)...")
                
                # Calculate 4K dimensions while preserving aspect ratio
                aspect_ratio = w / h
                if aspect_ratio >= 1.77:  # Wide (16:9 or wider)
                    target_w = 3840
                    target_h = int(3840 / aspect_ratio)
                else:  # Tall or square
                    target_h = 2160
                    target_w = int(2160 * aspect_ratio)
                
                # Choose upscaling method
                if upscale_method == "Real-ESRGAN (AI - Best Quality)":
                    image_4k = upscale_image_realesrgan(image, target_w, target_h)
                else:  # LANCZOS (Fast) or default
                    image_4k = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
                
                # Save both versions
                filename_original = os.path.join(
                    OUTPUT_DIR, f"sac_{w}x{h}_seed{current_seed}.png"
                )
                filename_4k = os.path.join(
                    OUTPUT_DIR, f"sac_4k_{target_w}x{target_h}_seed{current_seed}.png"
                )
                
                image.save(filename_original, quality=100)
                image_4k.save(filename_4k, quality=100)
                
                generated_images.append(image_4k)
                print(f"✓ Saved: {filename_original} and {filename_4k}")
            else:
                filename = os.path.join(
                    OUTPUT_DIR, f"sac_{w}x{h}_seed{current_seed}.png"
                )
                image.save(filename, quality=100)
                generated_images.append(image)
                print(f"✓ Saved: {filename}")
        
        # Generate info text
        features = []
        features.append(f"Model: {base_model_name}")
        if controlnet_name != "None":
            features.append(f"ControlNet: {controlnet_name}")
        if use_ip_adapter:
            features.append(f"IP-Adapter: Scale {ip_adapter_scale}")
        
        upscale_info = f"{upscale_to_4k}"
        if upscale_to_4k:
            upscale_info += f" ({upscale_method})"
        
        info = f"""✓ Generated {batch_count} image(s) at {w}x{h}
{', '.join(features)}
Scheduler: {scheduler_name}
Steps: {steps}, Guidance: {guidance_scale}
Seeds: {', '.join(map(str, seeds_used))}
Upscaled to 4K: {upscale_info}
Saved to: {OUTPUT_DIR}"""
        
        # Return all images in gallery format
        return generated_images, info
    
    except Exception as e:
        error_msg = f"❌ Generation failed: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        raise gr.Error(error_msg)

# -------------------------------------------------------------------
# Gradio UI
# -------------------------------------------------------------------

with gr.Blocks(title="Sovereign AI Collective Image Generator") as demo:
    # Simple cyan/magenta on dark grey styling
    gr.HTML(
        """
<style>
body, gradio-app, .gradio-container {
    background: #1a1a1d !important;
    color: #e0e0e0 !important;
}
#sac-header {
    text-align: center;
    font-size: 2.4rem;
    font-weight: 800;
    margin-bottom: 20px;
    background: linear-gradient(90deg, #00eaff, #ff00e6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 0 18px #00f0ff55, 0 0 22px #ff00ff44;
}
.gr-textbox textarea, .gr-dropdown, .gr-number, .gr-slider {
    background-color: #2a2a2d !important;
    color: #f2f2f2 !important;
    border: 1px solid #444444 !important;
}
.gr-textbox textarea:focus {
    border: 1px solid #00eaff !important;
    box-shadow: 0 0 12px #00eaff99;
}
button, .gr-button {
    background: linear-gradient(90deg, #00cfff, #ff00d4) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    border-radius: 6px !important;
    box-shadow: 0 0 10px #00eaff77;
}
button:hover, .gr-button:hover {
    box-shadow: 0 0 15px #ff00ffcc, 0 0 25px #00eaffcc;
    transform: translateY(-2px);
    transition: 0.15s ease-in-out;
}
.gr-panel, .gr-accordion {
    background-color: #242427 !important;
    border: 1px solid #333333 !important;
}
canvas, img {
    border-radius: 8px !important;
    box-shadow: 0 0 20px #00eaff33;
}
</style>
"""
    )

    gr.Markdown('<div id="sac-header">Sovereign AI Collective Image Generator</div>')
    gr.Markdown(
        "**Optimized with SDXL (Juggernaut / RealVis) + Compel + DPM++ + IP-Adapter for photorealistic generation.**"
    )
    
    with gr.Row():
        # Left column - Controls
        with gr.Column(scale=2):
            # Prompts
            prompt_input = gr.Textbox(
                label="Positive Prompt (supports weighted syntax: (feature:1.3))",
                value="",
                lines=8,
                placeholder="Describe what you want to see...",
            )
            
            negative_input = gr.Textbox(
                label="Negative Prompt",
                value="",
                lines=5,
                placeholder="Describe what to avoid...",
            )

            with gr.Row():
                base_model_dropdown = gr.Dropdown(
                    choices=list(BASE_MODELS.keys()),
                    value="Juggernaut XL v9",
                    label="Base Model",
                )
                controlnet_dropdown = gr.Dropdown(
                    choices=list(CONTROLNETS.keys()),
                    value="None",
                    label="ControlNet",
                )

            # ControlNet section
            gr.Markdown("### ControlNet (Optional)")
            control_image = gr.Image(
                label="Control Image (for depth/canny edge guidance)",
                type="pil",
                visible=True,
            )
            controlnet_scale = gr.Slider(
                minimum=0.0,
                maximum=2.0,
                value=0.8,
                step=0.05,
                label="ControlNet Strength",
                info="0 = ignore, ~0.5–1.0 = normal influence",
            )
            
            # IP-Adapter section
            gr.Markdown("### IP-Adapter (Face/Style Reference)")
            with gr.Row():
                use_ip_adapter_check = gr.Checkbox(
                    value=False,
                    label="Enable IP-Adapter",
                    info="Use reference image for face/style consistency",
                )
            
            ip_adapter_image = gr.Image(
                label="Reference Image (face or style to match)",
                type="pil",
                visible=True,
            )
            
            ip_adapter_scale = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.6,
                step=0.05,
                label="IP-Adapter Strength",
                info="0 = ignore, 0.6 = balanced, 1.0 = maximum influence",
            )

            pipeline_status = gr.Markdown(
                "✓ Loaded Juggernaut XL v9 with None ControlNet"
            )
            
            # Scheduler selection
            scheduler_dropdown = gr.Dropdown(
                choices=list(SCHEDULERS.keys()),
                value="DPM++ (Recommended)",
                label="Scheduler",
                info="DPM++ gives best quality in fewer steps",
            )
            
            # Generation settings
            with gr.Row():
                steps_slider = gr.Slider(
                    minimum=20,
                    maximum=100,
                    value=35,
                    step=5,
                    label="Inference Steps",
                    info="DPM++: 30-40 optimal, Euler: 50-60",
                )
                
                guidance_slider = gr.Slider(
                    minimum=1.0,
                    maximum=15.0,
                    value=8.0,
                    step=0.5,
                    label="Guidance Scale",
                    info="Lower = more natural (7-9 recommended)",
                )
            
            # Resolution settings
            with gr.Row():
                width_slider = gr.Slider(
                    minimum=512,
                    maximum=2560,
                    value=1920,
                    step=128,
                    label="Width",
                    info="2K = 1920",
                )
                
                height_slider = gr.Slider(
                    minimum=512,
                    maximum=1440,
                    value=1080,
                    step=128,
                    label="Height",
                    info="2K = 1080",
                )
            
            # Additional options
            with gr.Row():
                seed_input = gr.Number(
                    value=-1,
                    label="Seed",
                    info="-1 for random",
                )
                
                batch_slider = gr.Slider(
                    minimum=1,
                    maximum=8,
                    value=1,
                    step=1,
                    label="Batch Count",
                    info="Generate multiple variations",
                )
                
                upscale_check = gr.Checkbox(
                    value=False,
                    label="Upscale to 4K",
                    info="Preserves aspect ratio when upscaling",
                )
            
            # Upscale method selection
            upscale_method_radio = gr.Radio(
                choices=["LANCZOS (Fast)", "Real-ESRGAN (AI - Best Quality)"],
                value="LANCZOS (Fast)",
                label="Upscale Method",
                info="LANCZOS is fast, Real-ESRGAN uses AI for superior quality"
            )
            
            # Preset buttons
            gr.Markdown("### Quick Presets")
            with gr.Row():
                btn_2k = gr.Button("2K Standard")
                btn_2k_wide = gr.Button("2K Ultrawide")
                btn_fast = gr.Button("Fast Test (512x512, 25 steps)")
            
            # Generate button
            generate_btn = gr.Button("🎨 Generate", variant="primary")
        
        # Right column - Output and System Monitor
        with gr.Column(scale=1):
            system_monitor = gr.Markdown(
                value=format_system_stats(),
                label="System Monitor",
            )
            
            refresh_btn = gr.Button("🔄 Refresh Stats")
            
            # Gallery for batch images
            output_gallery = gr.Gallery(
                label="Generated Images", 
                show_label=True,
                columns=2,
                rows=2,
                object_fit="contain",
                height="auto"
            )
            output_info = gr.Textbox(label="Generation Info", lines=6)
    
    # Preset click handlers
    def set_2k():
        return 1920, 1080, 35
    
    def set_2k_wide():
        return 2560, 1440, 35
    
    def set_fast():
        return 512, 512, 25
    
    btn_2k.click(set_2k, outputs=[width_slider, height_slider, steps_slider])
    btn_2k_wide.click(set_2k_wide, outputs=[width_slider, height_slider, steps_slider])
    btn_fast.click(set_fast, outputs=[width_slider, height_slider, steps_slider])
    
    # Refresh system stats
    refresh_btn.click(fn=format_system_stats, outputs=system_monitor)

    # Update pipeline when base model, ControlNet, or IP-Adapter changes
    def on_pipe_change(base_model_name, controlnet_name, use_ip_adapter):
        return load_pipeline(base_model_name, controlnet_name, use_ip_adapter)

    base_model_dropdown.change(
        on_pipe_change,
        inputs=[base_model_dropdown, controlnet_dropdown, use_ip_adapter_check],
        outputs=pipeline_status,
    )
    controlnet_dropdown.change(
        on_pipe_change,
        inputs=[base_model_dropdown, controlnet_dropdown, use_ip_adapter_check],
        outputs=pipeline_status,
    )
    use_ip_adapter_check.change(
        on_pipe_change,
        inputs=[base_model_dropdown, controlnet_dropdown, use_ip_adapter_check],
        outputs=pipeline_status,
    )
    
    # Generate button handler
    generate_btn.click(
        fn=generate_image,
        inputs=[
            prompt_input,
            negative_input,
            base_model_dropdown,
            controlnet_dropdown,
            control_image,
            controlnet_scale,
            use_ip_adapter_check,
            ip_adapter_image,
            ip_adapter_scale,
            scheduler_dropdown,
            steps_slider,
            guidance_slider,
            width_slider,
            height_slider,
            seed_input,
            batch_slider,
            upscale_check,
            upscale_method_radio,  # NEW: Upscale method selection
        ],
        outputs=[output_gallery, output_info],
    )

# Launch the interface
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )