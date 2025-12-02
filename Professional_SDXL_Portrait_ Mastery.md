# Professional SDXL Portrait Mastery: Techniques for Print-Ready Quality

**The secret to jaw-dropping SDXL portraits lies in natural language prompting, conservative settings, and strategic model selection—not aggressive parameters.** Professional SDXL artists achieve razor-sharp results by thinking like photographers, keeping CFG between **4-7**, and using specific lens/lighting terminology that triggers SDXL's training on high-quality photographic data. For your before/after showcase, this guide provides the complete technical stack: optimal prompts for maximum sharpness, color vibrancy techniques for striking cyan eyes and magenta lips, head-to-head model recommendations, and Real-ESRGAN settings that preserve facial detail during upscaling.

## Critical resolution warning before you begin

**SDXL is not trained for 512x512**—this resolution will produce degraded, artifact-prone results. SDXL's native training resolution is **1024×1024** (approximately 1 megapixel total). For portrait work, generate at **832×1248** (2:3 portrait aspect) or **896×1152** (9:7), then downscale to 512×512 if that's your final target. Attempting native 512×512 generation bypasses SDXL's strengths and will significantly compromise your "after" showcase quality.

If you absolutely must generate at 512×512 directly, ByteDance's **ResAdapter** enables SDXL-Lightning to output this resolution, though quality remains compromised versus the generate-high-then-downscale approach.

## Razor-sharp detail starts with photography language

SDXL responds dramatically better to natural language descriptions than tag-style prompts. Write prompts as a photographer would describe a shot, not as keyword lists.

**Camera and lens terminology triggers quality recognition most effectively.** Professional artists consistently reference specific equipment: "Shot on Canon EOS 5D Mark IV with 85mm lens at f/1.8" or "Hasselblad X2D 100C, XCD 90mm f/2.5." The 85mm focal length is particularly effective—it's the classic portrait lens that produces flattering compression and bokeh. Include aperture values (f/1.4, f/1.8) alongside "shallow depth of field" to reinforce the cinematic look.

**Skin texture terminology prevents the plastic look that ruins photorealism.** Add "(realistic skin texture:1.2), visible pores, fine microtexture, natural imperfections" to your positive prompt. For striking but natural results, include "subtle freckles, faint smile lines" rather than leaving skin unnaturally smooth. Weight these terms modestly—SDXL is more sensitive to weights than SD 1.5, so keep all values under **1.4** to avoid artifacts.

**Professional lighting terminology transforms flat renders into dimensional portraits.** Specify lighting patterns by name: "three-point lighting setup," "Rembrandt lighting," "butterfly lighting with softbox," or "rim lighting for edge separation." Include "catchlights in eyes" explicitly—this single term dramatically increases eye detail and perceived sharpness.

### The professional positive prompt structure

Organize prompts in this sequence: subject → features → skin details → lighting → camera specs → focus → quality boosters:

```
Professional studio portrait of a woman with wavy auburn hair, 
(realistic skin texture:1.2), visible pores and natural imperfections, 
detailed iris with subtle catchlights, three-point lighting with softbox, 
shot on Canon 5D Mark IV with 85mm lens at f/1.8, shallow depth of field, 
subject in tack sharp focus, 8K UHD, RAW photo, hyperrealism
```

### Negative prompts require restraint with SDXL

Unlike SD 1.5, SDXL doesn't need extensive negative prompt lists. Professionals keep these minimal:

```
blurry, out of focus, soft focus, plastic skin, waxy, airbrushed, 
CGI, 3D render, cartoon, deformed, bad anatomy, extra limbs, watermark
```

Adding "smooth skin" to negatives prevents over-smoothing, while "oversaturated, extreme contrast" prevents color issues. Resist the urge to paste massive SD 1.5-era negative prompt blocks—they provide diminishing returns on SDXL and can sometimes hurt results.

## Achieving vivid color pops while maintaining balance

Creating striking cyan eyes and vibrant magenta lips requires careful prompt weighting combined with strategic negative prompts to prevent color bleeding.

**For bright cyan/turquoise eyes**, use terms like "(bright turquoise eyes:1.2), detailed iris, crystal clear eyes, catch light in eyes" in combination with "85mm lens, shallow depth of field focusing on eyes." The Perfect Eyes XL LoRA (trigger: `perfecteyes`) dramatically improves eye definition when applied at 0.6-0.8 strength. For post-generation enhancement, the **ADetailer extension** with the `mediapipe_face_mesh_eyes_only` model can selectively refine eye areas without affecting the rest of the image.

**For vivid magenta/pink lipstick**, specify "(vibrant magenta lipstick:1.2), glossy lips, defined lip edges, rich pigmented lips" alongside texture terms like "natural lip creases, subtle lip shine." The critical technique for preventing color bleeding is adding the unwanted color combinations to your negative prompt with high weight: `(magenta skin:1.4), (pink clothes:1.4), (pink background:1.3)`. This isolation technique keeps the magenta concentrated on lips only.

**The VectorscopeCC extension** provides granular color control during generation. Enable it with Alt mode active, then set saturation to **+4 to +6** for vivid colors while keeping brightness at **-2 to +2**. Combined with the SDXL Offset LoRA at 0.5-1.0 weight, this produces significantly more dynamic color range than default settings.

## Model selection: RealVisXL versus Juggernaut XL

Both models excel at photorealistic portraits, but they have distinct characteristics that make each better suited to different aesthetic goals.

**RealVisXL V4/V5 produces brighter, more vibrant outputs** with exceptional lifelike human rendering. It excels at natural skin tones, expressive eyes, and dynamic scenes. For your "after" showcase requiring vivid color pops, RealVisXL is likely the stronger choice—its inherent color richness complements the cyan eyes and magenta lips requirement without requiring aggressive saturation adjustments.

**Juggernaut XL v9 delivers superior skin details and texture control.** It renders freckles, moles, and fine skin imperfections more accurately than most alternatives. Version 8 and later specifically improved hands and feet—historically problematic areas. However, Juggernaut defaults to **darker, more dramatic tones** that may require brightness compensation for your vibrant showcase aesthetic.

**For maximum color vibrancy with photorealism, RealVisXL V4 is the recommended choice.** If you need maximum skin texture detail and don't mind compensating for darker defaults, Juggernaut XL v9 edges ahead. The emerging alternative **epiCRealism XL** is increasingly cited as the overall best SDXL photorealism model, worth testing against both.

## DPM++ scheduler and optimal generation settings

**DPM++ SDE Karras produces the best portrait quality** according to model creators and community testing. DPM++ 2M Karras offers faster generation with nearly equivalent results and serves as an excellent fallback when generation time matters.

The optimal parameter stack for photorealistic portraits:

| Parameter | Recommended Value | Reasoning |
|-----------|------------------|-----------|
| Resolution | 832×1248 or 1024×1024 | Native SDXL resolutions |
| Sampler | DPM++ SDE Karras | Best detail rendering |
| Steps | **25-30** | Sweet spot; beyond 35 shows diminishing returns |
| CFG Scale | **5-6** | Critical for photorealism; higher causes oversaturation |
| Refiner | 0.75 switch point | Essential for fine detail enhancement |

**CFG scale is the most commonly misconfigured parameter.** Values above 7 introduce hypersaturation and unnatural contrast that destroys photorealism. For some fine-tuned models like Cinematix, CFG as low as **2** produces optimal results. Start at **5** and adjust based on output.

## IP-Adapter and ControlNet depth configuration

**IP-Adapter face strength should be surprisingly conservative.** For the ip-adapter-plus-face_sdxl model, use **0.3** strength—higher values override prompt details excessively. The ip-adapter-faceid-plusv2_sdxl model tolerates higher values (**0.7-1.0**) because it combines face ID embedding with controllable CLIP embedding for better balance.

Set the **Starting Control Step to 0.5** when using IP-Adapter. This allows the base image to form before face influence takes effect, producing more natural results than full-duration control. Common IP-Adapter issues like face distortion typically indicate strength set too high—reduce incrementally until natural appearance returns.

**ControlNet Depth strength of 0.5 provides the optimal balance** between structural control and creative flexibility. For stricter adherence to reference depth maps, increase to 0.7-0.8. For portraits, the **Zoe Depth** estimation method outperforms Midas for close-up facial work. The xinsir/controlnet-depth-sdxl-1.0 model supports both detection methods for flexibility.

## Real-ESRGAN upscaling for print-ready output

**The RealESRGAN_x4plus model with the `--face_enhance` flag enabled is the optimal configuration for portrait upscaling.** This flag integrates GFPGAN (Generative Facial Prior GAN) specifically for facial detail restoration, adding fine details like eyelashes and realistic skin texture that standard upscaling loses.

**Preprocessing determines upscaling ceiling.** Before running Real-ESRGAN, denoise the source image to prevent amplifying noise, remove any JPEG compression artifacts, and fix white balance issues. Starting from the highest available source resolution is critical—AI upscalers hallucinate detail when pushed too hard on degraded inputs.

**For portrait-specific settings:**

```bash
python inference_realesrgan.py -n RealESRGAN_x4plus -i input.jpg -o output.jpg --face_enhance -dn 0.4
```

The denoise strength (`-dn`) parameter defaults to 0.5; for portraits, **0.4** preserves more natural skin texture while still reducing noise. Values above 0.6 risk producing waxy, plastic-looking skin.

**Consider 2x upscaling over 4x for web showcase.** While 4x sounds more impressive, faces can look artificial when over-processed. For print output, 4x is appropriate, but for web demonstration where images will be compressed anyway, **2x produces more reliably natural results**. If 4x introduces plasticity, try two sequential 2x passes instead—this sometimes produces superior results.

For post-upscaling, apply **light sharpening with 0.5-1.0 pixel radius**, but mask skin and sky areas from aggressive sharpening to avoid artifacts. Always review output at 100% zoom before finalizing.

## Mistakes that undermine professional results

**Using tag-style prompts instead of natural language** is the most common SDXL prompting error. "beautiful woman, portrait, high quality" produces mediocre results; "Cinematic portrait of a woman with soft freckles, studio three-point lighting, 85mm lens, shallow DOF, editorial magazine style" dramatically improves output.

**CFG scale above 7** causes hypersaturation and cartoon-like artifacts that destroy photorealism. This single setting mistake accounts for a significant portion of failed SDXL portrait attempts.

**Skipping the Refiner** leaves substantial quality on the table. For portraits, the SDXL Refiner (base-to-refiner workflow with 0.75 switch point) significantly enhances eye detail, skin texture, shadow gradation, and edge definition. Always enable it for showcase-quality work.

**Using SD 1.5 LoRAs with SDXL models** produces errors or degraded results—these are not compatible. Similarly, stacking more than **3 LoRAs simultaneously** destabilizes generation. For portraits, use a character LoRA at 0.6-0.9, a style LoRA at 0.4-0.7, and a detail LoRA at 0.3-0.6 as maximum.

**Ignoring aspect ratio effects** can cause the infamous "two heads" artifact in extreme portrait orientations. If this occurs, generate at 1:1 square and crop to portrait in post-processing.

## Complete workflow for your showcase image

For the jaw-dropping "after" image in your before/after demonstration, execute this sequence:

1. **Generate at 832×1248** using RealVisXL V4 with DPM++ SDE Karras, 28 steps, CFG 5
2. **Apply the SDXL Refiner** at 0.75 switch point for detail enhancement
3. **Enable ADetailer** with face_yolov8n.pt for automatic face optimization, plus mediapipe_face_mesh_eyes_only as second model for eye enhancement
4. **Use HiRes Fix** with 4x-UltraSharp upscaler, denoising 0.35-0.45, scale 1.5x
5. **Final upscale with Real-ESRGAN** x4plus with --face_enhance flag if additional resolution needed
6. **Post-process** with light sharpening (0.5-1.0px), masking skin from aggressive sharpening

This workflow, combined with the photography-style prompts containing specific lens references, lighting terminology, and conservative color weighting for the cyan eyes and magenta lips, will produce print-ready portraits that demonstrate the full capability of your SDXL stack.