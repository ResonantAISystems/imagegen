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

device = "cuda" if torch.cuda.is_available() else "cpu"

pipe = None
compel = None
current_base_model = None
current_controlnet = None

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
    For Depth: Return as-is (assuming user provides depth map, or we'd need a depth model)
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
        # In production, you'd use a depth estimation model here
        return image
    
    return image


# -------------------------------------------------------------------
# Pipeline loader
# -------------------------------------------------------------------

def load_pipeline(base_model_name: str, controlnet_name: str):
    """
    Load / swap the SDXL pipeline with optional ControlNet.
    """
    global pipe, compel, current_base_model, current_controlnet

    if (
        pipe is not None
        and base_model_name == current_base_model
        and controlnet_name == current_controlnet
    ):
        # No change
        return f"✓ Using {base_model_name} with {controlnet_name} ControlNet"

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

        return f"✓ Loaded {base_model_name} with {controlnet_name} ControlNet"
    
    except Exception as e:
        error_msg = f"❌ Failed to load pipeline: {str(e)}"
        print(error_msg)
        raise gr.Error(error_msg)


# Initial load (Juggernaut, no ControlNet)
try:
    load_pipeline("Juggernaut XL v9", "None")
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
    scheduler_name,
    steps,
    guidance_scale,
    width,
    height,
    seed,
    batch_count,
    upscale_to_4k,
):
    """Generate image(s) with optional ControlNet & 4K upscaling"""
    
    try:
        # Make sure correct pipeline is loaded
        status = load_pipeline(base_model_name, controlnet_name)
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
        
        # Process prompts with Compel for SDXL (returns embeddings + pooled embeddings)
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
                f"model: {base_model_name}, controlnet: {controlnet_name}"
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

            if CONTROLNETS[controlnet_name] is not None:
                if control_image is None:
                    raise gr.Error("ControlNet selected but no control image provided.")
                
                # Preprocess control image based on ControlNet type
                processed_control = preprocess_control_image(control_image, controlnet_name)
                
                pipe_kwargs["image"] = processed_control
                pipe_kwargs["controlnet_conditioning_scale"] = float(controlnet_scale)

            image = pipe(**pipe_kwargs).images[0]
            
            # Upscale if requested (preserving aspect ratio)
            if upscale_to_4k:
                print("Upscaling to 4K (preserving aspect ratio)...")
                
                # Calculate 4K dimensions while preserving aspect ratio
                aspect_ratio = w / h
                if aspect_ratio >= 1.77:  # Wide (16:9 or wider)
                    target_w = 3840
                    target_h = int(3840 / aspect_ratio)
                else:  # Tall or square
                    target_h = 2160
                    target_w = int(2160 * aspect_ratio)
                
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
        info = f"""✓ Generated {batch_count} image(s) at {w}x{h}
Base model: {base_model_name}
ControlNet: {controlnet_name}
Scheduler: {scheduler_name}
Steps: {steps}, Guidance: {guidance_scale}
Seeds: {', '.join(map(str, seeds_used))}
Upscaled to 4K: {upscale_to_4k}
Saved to: {OUTPUT_DIR}"""
        
        # Return all images in gallery format
        return generated_images, info
    
    except Exception as e:
        error_msg = f"❌ Generation failed: {str(e)}"
        print(error_msg)
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
        "**Optimized with SDXL (Juggernaut / RealVis) + Compel + DPM++ for photorealistic generation.**"
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

            control_image = gr.Image(
                label="Control Image (for ControlNet – depth/canny)",
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
            
            # Changed to Gallery to show all batch images
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

    # Update pipeline when base model or ControlNet changes
    def on_pipe_change(base_model_name, controlnet_name):
        return load_pipeline(base_model_name, controlnet_name)

    base_model_dropdown.change(
        on_pipe_change,
        inputs=[base_model_dropdown, controlnet_dropdown],
        outputs=pipeline_status,
    )
    controlnet_dropdown.change(
        on_pipe_change,
        inputs=[base_model_dropdown, controlnet_dropdown],
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
            scheduler_dropdown,
            steps_slider,
            guidance_slider,
            width_slider,
            height_slider,
            seed_input,
            batch_slider,
            upscale_check,
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