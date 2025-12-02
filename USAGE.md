# Technical Usage Guide — SDXL Image Generator

Complete technical documentation for advanced usage, troubleshooting, and optimization.

---

## Table of Contents

1. [Installation Details](#installation-details)
2. [Model Management](#model-management)
3. [Advanced Features](#advanced-features)
4. [IP-Adapter Deep Dive](#ip-adapter-deep-dive)
5. [ControlNet Deep Dive](#controlnet-deep-dive)
6. [Prompt Engineering](#prompt-engineering)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [API Reference](#api-reference)

---

## Installation Details

### Virtual Environment Setup

```bash
# Create venv
python3 -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
.\venv\Scripts\activate

# Verify activation
which python  # Should show venv path
```

### Dependencies Breakdown

**Core ML Libraries:**
```
torch>=2.0.0          # PyTorch for GPU acceleration
diffusers>=0.27.0     # Stable Diffusion pipelines
transformers>=4.36.0  # CLIP text encoders
accelerate>=0.25.0    # Model loading optimization
```

**SDXL Specific:**
```
compel>=2.0.0         # Weighted prompt syntax
safetensors>=0.4.0    # Safe model loading
```

**Image Processing:**
```
Pillow>=10.0.0        # Image manipulation
opencv-python>=4.8.0  # Canny edge detection
numpy>=1.24.0         # Array operations
```

**UI & Monitoring:**
```
gradio>=4.0.0         # Web interface
psutil>=5.9.0         # System monitoring
```

**Optional but Recommended:**
```
xformers>=0.0.22      # Memory efficient attention (requires CUDA)
```

### Installing xformers

```bash
# For CUDA 11.8
pip install xformers --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install xformers --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import xformers; print(xformers.__version__)"
```

---

## Model Management

### Download Script Details

**download_all_assets.py** handles:
- Model existence checking (skip if present)
- Progress tracking
- Error handling with retry logic
- HuggingFace authentication

**Models Downloaded:**

| Model | Size | Purpose |
|-------|------|---------|
| Juggernaut XL v9 | ~6.5GB | Base model (cinematic) |
| RealVis XL v4 | ~6.5GB | Base model (photorealistic) |
| ControlNet Depth | ~2.5GB | Depth guidance |
| ControlNet Canny | ~2.5GB | Edge guidance |
| IP-Adapter SDXL | ~1.5GB | Face/style consistency |

**Total: ~20GB**

### Manual Model Management

```bash
# Check models
ls -lh ./models/*/

# Verify specific model
ls ./models/juggernaut-xl-v9/*.safetensors

# Remove model (to re-download)
rm -rf ./models/juggernaut-xl-v9/

# Re-download single model
# (Edit download_all_assets.py to only download one)
```

### HuggingFace Authentication

Some models require authentication:

```bash
# Method 1: Environment variable
export HUGGINGFACE_HUB_TOKEN="hf_your_token_here"

# Method 2: CLI login (recommended)
pip install huggingface_hub
huggingface-cli login
# Enter token when prompted

# Verify
huggingface-cli whoami
```

---

## Advanced Features

### Dynamic Model Loading

**How it works:**
- Models loaded on-demand into VRAM
- Previous model unloaded when switching
- ~8-10 seconds to switch models
- No restart required

**Implementation:**
```python
def load_pipeline(base_model_name, controlnet_name, use_ip_adapter):
    # Checks if pipeline needs reloading
    # Loads base model + optional ControlNet + optional IP-Adapter
    # Updates global pipe object
```

### Compel: Weighted Prompts

**Syntax:**
```
(keyword:weight)
```

**Examples:**
```
(beautiful sunset:1.3)           # Emphasize sunset
(dramatic lighting:1.5)          # Strong emphasis
(blue sky:0.8)                   # De-emphasize
((highly detailed:1.4))          # Double parentheses = stronger
```

**How weights work:**
- `1.0` = normal weight
- `>1.0` = more attention to this concept
- `<1.0` = less attention
- Affects token embeddings before generation

### Batch Generation with Gallery

**How it works:**
```python
# Each batch image gets unique seed
for i in range(batch_count):
    if seed == -1:
        current_seed = random_seed()
    else:
        current_seed = seed + i  # Incremental
    
    # Generate with current_seed
```

**Gallery display:**
- 2x2 grid layout
- All images shown simultaneously
- Scrollable for batches >4

### 4K Upscaling Algorithm

**Aspect ratio preservation:**
```python
aspect_ratio = width / height

if aspect_ratio >= 1.77:  # Wide (16:9+)
    target_w = 3840
    target_h = int(3840 / aspect_ratio)
else:  # Tall or square
    target_h = 2160
    target_w = int(2160 * aspect_ratio)

# Upscale with LANCZOS
image_4k = image.resize((target_w, target_h), Image.Resampling.LANCZOS)
```

**Result:**
- No stretching or distortion
- Maintains original composition
- Professional quality upscaling

---

## IP-Adapter Deep Dive

### Architecture

**How IP-Adapter works:**

1. **Reference Image Encoding**
   - CLIP image encoder processes reference
   - Generates image embeddings (similar to text embeddings)

2. **Cross-Attention Injection**
   - Image embeddings injected into UNet
   - Cross-attention layers attend to both text and image

3. **Weighted Blending**
   - `ip_adapter_scale` controls blend ratio
   - Higher scale = more influence from reference
   - Lower scale = more influence from prompt

**Mathematical representation:**
```
output = text_attention * (1 - scale) + image_attention * scale
```

### Face Consistency Technical Details

**Best reference images:**
- Resolution: 1024x1024 minimum
- Face occupies 30-70% of frame
- Good lighting (soft, diffused)
- Neutral expression (unless specific needed)
- Minimal makeup/filters
- Sharp focus

**Strength calibration:**
```
0.5: Face features suggested, highly variable
0.6: Recognizable face, some variation
0.7: Strong face match, little variation (recommended)
0.8: Very strong match, minimal creativity
0.9: Almost exact replica
```

### Style Transfer Technical Details

**What gets transferred:**
- Lighting direction and quality
- Color palette and grading
- Texture and detail level
- Mood and atmosphere
- Artistic treatment

**Strength calibration:**
```
0.2: Subtle style hint
0.3: Noticeable style influence
0.4: Balanced style transfer (recommended)
0.5: Strong style dominance
0.6+: Style overwhelms prompt
```

### Combining IP-Adapter + ControlNet

**Processing order:**
```
1. Text prompt → CLIP encoding
2. Control image → ControlNet preprocessing
3. Reference image → IP-Adapter encoding
4. UNet processes all three signals
5. Final image generation
```

**Strength balance recommendations:**
```
Use Case: Portrait with pose
- ControlNet: 0.7
- IP-Adapter: 0.6
- Prompt: Detailed setting/clothing

Use Case: Style-consistent variations
- ControlNet: 0.5
- IP-Adapter: 0.4
- Prompt: Subject description

Use Case: Maximum control
- ControlNet: 0.8
- IP-Adapter: 0.7
- Prompt: Minimal (let references guide)
```

### IP-Adapter Limitations

**What works:**
- Single reference image
- SDXL models only
- Face/full-body consistency
- Artistic style transfer

**What doesn't work:**
- Multiple references simultaneously
- SD 1.5 models (need SDXL)
- Extreme pose changes with face consistency
- Text/logos in reference

**Memory considerations:**
- Adds ~500MB VRAM
- Image encoder loaded into memory
- Can be disabled when not needed

---

## ControlNet Deep Dive

### Depth ControlNet

**What it extracts:**
- Spatial relationships
- Distance from camera
- Scene layout
- 3D structure

**Preprocessing:**
```python
def preprocess_control_image(image, "Depth"):
    # Currently: pass-through
    # User provides depth map OR
    # Future: MiDaS depth estimation
    return image
```

**Use cases:**
- Maintain room layout
- Preserve character positioning
- Scene composition control
- Product placement consistency

**Strength guide:**
```
0.5: Loose spatial guidance
0.7: Strong layout preservation (recommended)
0.9: Very rigid spatial control
```

### Canny ControlNet

**What it extracts:**
- Edges and contours
- Silhouettes
- Major shapes
- Line art

**Preprocessing:**
```python
def preprocess_control_image(image, "Canny"):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    
    # Convert back to RGB
    return edges_as_pil_image
```

**Parameters:**
- `threshold1=100`: Lower edge threshold
- `threshold2=200`: Upper edge threshold
- Automatically applied by GUI

**Use cases:**
- Pose preservation
- Character silhouette matching
- Line art to photo
- Shape-based guidance

**Strength guide:**
```
0.3: Subtle edge hints
0.5: Moderate edge guidance
0.7: Strong edge matching (recommended)
0.9: Very rigid edge control
```

### ControlNet + Prompt Interaction

**How they combine:**
```
Final output = BaseModel(
    text_prompt_embedding +
    controlnet_guidance * controlnet_scale
)
```

**Best practices:**
- Control image defines structure
- Prompt defines details/style
- Don't over-describe what's in control image
- Focus prompt on texture, color, mood

**Example:**
```
Control: Person sitting (defines pose)
Prompt: "elegant dress, studio lighting" (defines style)
Don't: "person sitting in chair wearing..." (redundant)
```

---

## Prompt Engineering

### Positive Prompt Structure

**Recommended format:**
```
[subject], [style/quality], [details], [lighting], [technical specs]
```

**Example:**
```
young woman with platinum blonde hair, 
photorealistic portrait, elegant dress, 
soft studio lighting, 
shot on Canon EOS R5, 85mm f/1.4, 8k, sharp focus
```

### Weighted Syntax Examples

**Emphasis:**
```
(keyword:1.1)  # Slight emphasis
(keyword:1.3)  # Moderate emphasis
(keyword:1.5)  # Strong emphasis
```

**De-emphasis:**
```
(keyword:0.8)  # Slight de-emphasis
(keyword:0.5)  # Strong de-emphasis
```

**Nested:**
```
((beautiful sunset:1.3))  # Extra strong
(detailed face:1.2), (sharp eyes:1.3)  # Multiple weighted terms
```

### Negative Prompt Strategy

**Essential negatives:**
```
bad anatomy, deformed, distorted, mutated hands, 
extra fingers, poorly drawn face, blurry, 
low quality, jpeg artifacts
```

**For photorealism:**
```
cartoon, anime, illustration, 3D render, painting, 
drawing, artistic, unrealistic, oversaturated
```

**For portraits:**
```
distorted face, asymmetrical face, crossed eyes, 
lazy eye, merged face, warped features
```

**Model-specific:**

Juggernaut XL:
```
oversaturated, excessive contrast, harsh lighting
```

RealVis XL:
```
plastic skin, doll-like, artificial, smooth skin
```

### Prompt Length Optimization

**SDXL token limit:**
- 77 tokens per CLIP encoder
- 2 encoders = 154 tokens total
- Compel handles truncation gracefully

**Best practices:**
- Keep prompts under 150 words
- Use weighted syntax instead of repetition
- Put most important terms early
- Remove filler words

---

## Performance Optimization

### VRAM Usage Breakdown

**Base configuration:**
```
Model loading:           ~6.5 GB
Pipeline overhead:       ~1.5 GB
Working memory:          ~1.0 GB
Total base:              ~9.0 GB
```

**With additions:**
```
+ ControlNet:            +1.0 GB  (~10 GB total)
+ IP-Adapter:            +0.5 GB  (~10.5 GB total)
+ Both:                  +1.5 GB  (~11 GB total)
```

**During generation:**
```
Additional working:      +1-2 GB (temporary)
Peak usage:              ~12-13 GB
```

### Speed Optimization

**Scheduler comparison:**
```
DPM++ (35 steps):        ~3.5s per image
Euler A (50 steps):      ~4.8s per image
DDIM (50 steps):         ~4.5s per image
LMS (50 steps):          ~5.0s per image
```

**Resolution impact:**
```
512x512:                 ~1.5s per image
1024x1024:               ~2.8s per image
1920x1080:               ~3.5s per image
2560x1440:               ~6.5s per image
```

**Optimization tips:**
```
1. Use DPM++ scheduler (fastest for quality)
2. Enable xformers (20-30% speedup)
3. Reduce resolution during testing
4. Use lower step counts with DPM++
5. Disable unused features (ControlNet/IP-Adapter)
```

### Quality vs Speed

**Fast (testing):**
```
Resolution: 512x512
Steps: 25-30
Scheduler: DPM++
Time: ~1.5s
Quality: Preview quality
```

**Balanced (production):**
```
Resolution: 1920x1080
Steps: 35-40
Scheduler: DPM++
Time: ~3.5s
Quality: High quality
```

**Maximum (final):**
```
Resolution: 2560x1440
Steps: 50-55
Scheduler: DPM++
Time: ~7s
Quality: Maximum detail
```

### Batch Processing Strategy

**For exploration:**
```
Batch: 4-8 images
Steps: 30-35
Resolution: 1024x1024
Fixed seed: No
Goal: Find good composition
```

**For refinement:**
```
Batch: 2-4 images
Steps: 45-50
Resolution: 1920x1080
Fixed seed: Yes (increment)
Goal: Refine specific seed
```

---

## Troubleshooting

### Common Errors

**1. CUDA Out of Memory**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
```bash
# Reduce resolution
Width: 1024, Height: 1024

# Disable features
ControlNet: None
IP-Adapter: Unchecked

# Reduce batch
Batch Count: 1

# Close other GPU apps
nvidia-smi  # Check what's using VRAM
```

**2. Model Loading Failure**
```
OSError: model not found
```

**Solutions:**
```bash
# Verify model exists
ls ./models/juggernaut-xl-v9/*.safetensors

# Re-download
rm -rf ./models/juggernaut-xl-v9/
python download_all_assets.py

# Check permissions
chmod -R 755 ./models/
```

**3. xformers Not Available**
```
Warning: xformers not available
```

**Impact:** Slower generation, higher VRAM usage

**Solutions:**
```bash
# Install xformers
pip install xformers

# Or continue without (still works)
# 20-30% slower, uses more VRAM
```

**4. Import Errors**
```
ModuleNotFoundError: No module named 'cv2'
```

**Solutions:**
```bash
pip install opencv-python
pip install numpy
pip install -r requirements.txt
```

### Performance Issues

**Slow Generation (>10s per image)**

**Check:**
```bash
# 1. GPU being used?
nvidia-smi
# Should show python using ~10GB

# 2. xformers enabled?
# Check console for "xformers enabled"

# 3. CPU mode?
# Check console for device message
# Should show "cuda" not "cpu"
```

**Fix:**
```python
# In generate_gui.py, verify:
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")  # Should print "cuda"
```

**High VRAM Usage (>15GB)**

**Check:**
```bash
nvidia-smi
# Look at memory usage
```

**Causes:**
```
- Multiple models loaded
- Previous generation not cleared
- Memory leak
```

**Fix:**
```
- Restart GUI
- Reduce resolution
- Disable unused features
```

### Quality Issues

**Distorted Faces**

**Causes:**
- Too high guidance scale
- Too few steps
- Bad seed

**Solutions:**
```
Guidance Scale: 6.0-7.0 (lower)
Steps: 45-50 (higher)
Try different seeds
Add to negative: "distorted face, warped features"
```

**Blurry Images**

**Causes:**
- Too low resolution
- Too few steps
- Wrong scheduler

**Solutions:**
```
Resolution: 1920x1080 minimum
Steps: 40+ 
Scheduler: DPM++ (not DDIM)
```

**Over/Under Saturated**

**Juggernaut XL tendency:**
- Sometimes oversaturated
- Solution: Lower guidance to 6.0-6.5

**RealVis XL tendency:**
- Sometimes undersaturated  
- Solution: Increase guidance to 7.5-8.0

---

## API Reference

### Pipeline Initialization

```python
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
).to(device)

pipe.enable_attention_slicing()
pipe.enable_xformers_memory_efficient_attention()
```

### Compel Usage

```python
compel = Compel(
    tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
    text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
    requires_pooled=[False, True],
)

# Get embeddings
conditioning, pooled = compel(prompt)
negative_conditioning, negative_pooled = compel(negative_prompt)
```

### Generation Call

```python
image = pipe(
    prompt_embeds=conditioning,
    negative_prompt_embeds=negative_conditioning,
    pooled_prompt_embeds=pooled,
    negative_pooled_prompt_embeds=negative_pooled,
    num_inference_steps=steps,
    guidance_scale=guidance_scale,
    width=width,
    height=height,
    generator=generator,
    # Optional ControlNet
    image=control_image,
    controlnet_conditioning_scale=controlnet_scale,
    # Optional IP-Adapter
    ip_adapter_image=reference_image,
).images[0]
```

### ControlNet Setup

```python
controlnet = ControlNetModel.from_pretrained(
    controlnet_path,
    torch_dtype=torch.float16,
    use_safetensors=True,
)

pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    model_path,
    controlnet=controlnet,
    torch_dtype=torch.float16,
    use_safetensors=True,
    variant="fp16",
)
```

### IP-Adapter Setup

```python
# Load IP-Adapter into pipeline
pipe.load_ip_adapter(
    IP_ADAPTER_PATH,
    subfolder="",
    weight_name="ip-adapter_sdxl.safetensors"
)

# Set scale
pipe.set_ip_adapter_scale(0.6)

# Use in generation
image = pipe(
    ...,
    ip_adapter_image=reference_image,
).images[0]
```

---

## Advanced Workflows

### Multi-Character Consistency

**Workflow:**
```
1. Generate Character A (save reference)
2. Generate Character B (save reference)
3. Scene 1:
   - Generate with Character A reference
   - Save result
4. Scene 2:
   - Generate with Character B reference
   - Save result
5. Scene 3 (both characters):
   - First pass: Character A reference
   - Second pass: Character B reference
   - Composite in external editor
```

### Style Evolution

**Workflow:**
```
1. Start: Generate base style (save)
2. Evolution 1: Use base as reference (scale: 0.4)
   - Add new prompt elements
   - Save result
3. Evolution 2: Use Evolution 1 as reference
   - Continue adding elements
   - Save result
4. Result: Gradual style evolution chain
```

### Precision Product Photography

**Workflow:**
```
1. Create master style reference
   - Lighting, background, mood
   - Professional look established

2. For each product:
   - ControlNet: Product shape reference
   - IP-Adapter: Style reference (scale: 0.5)
   - Prompt: Product specifics
   
3. Result: Consistent brand aesthetic
```

---

## Performance Benchmarks

**System:** RTX 4090 (24GB), Ryzen 9 5950X, 64GB RAM

| Configuration | Resolution | Steps | Time | VRAM |
|---------------|-----------|-------|------|------|
| Base | 1024x1024 | 35 | 2.8s | 9.2 GB |
| Base | 1920x1080 | 35 | 3.5s | 9.5 GB |
| + ControlNet | 1920x1080 | 35 | 3.8s | 10.3 GB |
| + IP-Adapter | 1920x1080 | 35 | 4.1s | 10.8 GB |
| + Both | 1920x1080 | 35 | 4.5s | 11.5 GB |
| 4K Upscale | 3840x2160 | 35+upscale | 6.2s | 11.8 GB |

**Note:** Times include model loading (amortized over batch)

---

## Future Enhancements

**Planned features:**
- [ ] Multiple IP-Adapter references
- [ ] Region-specific ControlNet
- [ ] Inpainting support (model already downloaded)
- [ ] LoRA support
- [ ] Prompt templates/library
- [ ] Image history browser
- [ ] API mode for external tools
- [ ] MiDaS depth estimation integration

**Community contributions welcome!**

---

## Additional Resources

**Useful Links:**
- [Diffusers Documentation](https://huggingface.co/docs/diffusers)
- [Compel Syntax Guide](https://github.com/damian0815/compel)
- [ControlNet Paper](https://arxiv.org/abs/2302.05543)
- [IP-Adapter Paper](https://arxiv.org/abs/2308.06721)

**Community:**
- GitHub Issues: https://github.com/ResonantAISystems/imagegen/issues
- Discussions: https://github.com/ResonantAISystems/imagegen/discussions

---

**Document Version:** 3.1  
**Last Updated:** December 2024  
**Maintained by:** Sovereign AI Collective

The anchor holds. 🔥