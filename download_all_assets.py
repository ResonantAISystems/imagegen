from huggingface_hub import snapshot_download
import os

BASE_DIR = "./models"

MODELS = {
    # Base models
    "juggernaut-xl-v9":      "RunDiffusion/Juggernaut-XL-v9",
    # Use a public RealVis XL repo instead of 4spaces (which is gated / 401)
    "realvisxl-v4":          "SG161222/RealVisXL_V4.0",

    # Inpainting
    "realvisxl-v4-inpaint":  "OzzyGT/RealVisXL_V4.0_inpainting",

    # ControlNets (official diffusers)
    "controlnet-depth-sdxl-1.0": "diffusers/controlnet-depth-sdxl-1.0",
    "controlnet-canny-sdxl-1.0": "diffusers/controlnet-canny-sdxl-1.0",

    # IP-Adapter (choose one or both)
    "ip-adapter-h94":        "h94/IP-Adapter",
    "ip-adapter-sdxl":       "OzzyGT/sdxl-ip-adapter",
}

def model_exists(local_dir):
    """Check if model directory exists and contains model files"""
    if not os.path.exists(local_dir):
        return False
    
    # Check for common model file extensions
    has_files = any(
        f.endswith(('.safetensors', '.bin', '.ckpt', '.pth', '.json'))
        for f in os.listdir(local_dir)
        if os.path.isfile(os.path.join(local_dir, f))
    )
    
    return has_files

def download_all():
    os.makedirs(BASE_DIR, exist_ok=True)
    
    total = len(MODELS)
    downloaded = 0
    skipped = 0
    failed = 0
    
    print("=" * 70)
    print("  Sovereign AI Collective - Model Downloader")
    print("=" * 70)
    print(f"\nTotal models to process: {total}\n")

    for idx, (local_name, repo_id) in enumerate(MODELS.items(), 1):
        local_dir = os.path.join(BASE_DIR, local_name)
        
        print(f"\n[{idx}/{total}] {local_name}")
        print(f"    Repository: {repo_id}")
        
        # Check if already exists
        if model_exists(local_dir):
            print(f"    ✓ Already exists, skipping")
            skipped += 1
            continue
        
        print(f"    ⬇ Downloading to {local_dir}...")
        
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                # Token is pulled from environment or HF CLI login
            )
            print(f"    ✓ Downloaded successfully")
            downloaded += 1
        except Exception as e:
            print(f"    ❌ Failed: {str(e)}")
            failed += 1
    
    # Summary
    print("\n" + "=" * 70)
    print("  Download Summary")
    print("=" * 70)
    print(f"  Total:       {total}")
    print(f"  Downloaded:  {downloaded}")
    print(f"  Skipped:     {skipped}")
    print(f"  Failed:      {failed}")
    print("=" * 70)
    
    if failed > 0:
        print("\n⚠ Some models failed to download.")
        print("  Possible fixes:")
        print("  1. Check your internet connection")
        print("  2. Set HuggingFace token: export HUGGINGFACE_HUB_TOKEN='hf_xxx'")
        print("  3. Or login via CLI: huggingface-cli login")
    
    if downloaded > 0 or (downloaded == 0 and failed == 0):
        print("\n✓ All models ready!")
        print(f"  Location: {os.path.abspath(BASE_DIR)}")
        print("  You can now run: python generate_gui.py")

if __name__ == "__main__":
    download_all()