import os
from huggingface_hub import hf_hub_download

MODEL_DIR = "model_zoo"
os.makedirs(MODEL_DIR, exist_ok=True)

# 1) 下载 NaVid finetuned 权重（从 Hugging Face）
navid_filename = "navid-7b-full-224-video-fps-1-grid-2-r2r-rxr-training-split"
navid_path = hf_hub_download(
    repo_id="Jzzhang/NaVid",
    filename=navid_filename,
    local_dir=MODEL_DIR,
    local_dir_use_symlinks=False,
)
print(f"NaVid 模型已下载到: {navid_path}")

# 2) 下载 EVA-ViT-G 视觉编码器权重（从公开 URL）
# 官方地址来自 README 链接
eva_dst = os.path.join(MODEL_DIR, "eva_vit_g.pth")
if not os.path.exists(eva_dst):
    import requests

    url = "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth"
    print("Downloading EVA-ViT-G weights... This may take a while.")
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(eva_dst, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    print(f"EVA-ViT-G 权重已下载到: {eva_dst}")
else:
    print(f"EVA-ViT-G 权重已存在: {eva_dst}")

print("全部下载完成。")
