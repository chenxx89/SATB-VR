import os
import shutil

import requests

REPO_IDS = {
    "CogVideoX1.5-5B": "zai-org/CogVideoX1.5-5B",
    "cogvlm2-llama3-caption": "zai-org/cogvlm2-llama3-caption",
    "SATB-VR": "chenxx89/SATB-VR",
}


def _check_hf_accessible(timeout=5):
    try:
        response = requests.get("https://huggingface.co", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


def download_ckpts(ckpt_path, enable_text_encoder=False):
    if not _check_hf_accessible():
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    from huggingface_hub import snapshot_download

    ckpt_names = ["CogVideoX1.5-5B", "SATB-VR"]
    if enable_text_encoder:
        ckpt_names.append("cogvlm2-llama3-caption")

    for ckpt_name in ckpt_names:
        local_dir = os.path.join(ckpt_path, ckpt_name)
        if not os.path.exists(local_dir):
            print(f"Start downloading {ckpt_name} to {local_dir}")
            snapshot_download(
                repo_id=REPO_IDS[ckpt_name],
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=4,
            )
            if ckpt_name == "cogvlm2-llama3-caption":
                ## remove the dependency on pytorchvideo
                shutil.copy2("SATB_VR/modeling_cogvlm.py", local_dir)


if __name__ == "__main__":
    download_ckpts("./ckpts")
