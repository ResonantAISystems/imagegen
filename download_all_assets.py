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

def download_all():
    os.makedirs(BASE_DIR, exist_ok=True)

    for local_name, repo_id in MODELS.items():
        local_dir = os.path.join(BASE_DIR, local_name)
        print(f"\n=== Downloading {repo_id} -> {local_dir}")
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                # `resume_download` & `local_dir_use_symlinks` are no longer needed
            )
            print(f"Done: {os.path.abspath(local_dir)}")
        except Exception as e:
            print(f"!!! Failed to download {repo_id}: {e}")

if __name__ == "__main__":
    download_all()
