"""
Download Qwen2.5-1.5B-Instruct from ModelScope (Chinese mirror, much faster)
Fallback to HuggingFace if ModelScope fails
"""
import os
import sys

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "Qwen2.5-1.5B-Instruct")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f"Target: {MODEL_NAME}")
print(f"Save to: {SAVE_DIR}")

# === Method 1: Try ModelScope (Chinese mirror) ===
try:
    print("\n[Method 1] Trying ModelScope (modelscope.cn)...")
    from modelscope import snapshot_download

    snapshot_download(
        "qwen/Qwen2.5-1.5B-Instruct",
        cache_dir=SAVE_DIR,
        revision="master",
    )
    print("ModelScope download successful!")
    sys.exit(0)
except ImportError:
    print("  modelscope not installed, installing...")
    os.system(f"{sys.executable} -m pip install modelscope -i https://pypi.tuna.tsinghua.edu.cn/simple -q")
    try:
        from modelscope import snapshot_download
        snapshot_download(
            "qwen/Qwen2.5-1.5B-Instruct",
            cache_dir=SAVE_DIR,
            revision="master",
        )
        print("ModelScope download successful!")
        sys.exit(0)
    except Exception as e:
        print(f"  ModelScope also failed: {e}")

# === Method 2: HuggingFace with mirror ===
print("\n[Method 2] Trying HuggingFace with hf-mirror.com...")
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

try:
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=MODEL_NAME,
        local_dir=SAVE_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4,
    )
    print("HF mirror download successful!")
    sys.exit(0)
except Exception as e:
    print(f"  HF mirror failed: {e}")

# === Method 3: Use transformers directly (auto-cache) ===
print("\n[Method 3] Using transformers AutoModel (direct download)...")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.save_pretrained(SAVE_DIR)
print("Tokenizer saved.")

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="cpu",  # Use CPU for download/save
)
print("Saving model...")
model.save_pretrained(SAVE_DIR, safe_serialization=True)
print(f"Model saved to {SAVE_DIR}")

# Verify
files = os.listdir(SAVE_DIR)
print(f"\nFiles in {SAVE_DIR}:")
for f in sorted(files):
    path = os.path.join(SAVE_DIR, f)
    if os.path.isfile(path):
        mb = os.path.getsize(path) / 1024 / 1024
        print(f"  {f}: {mb:.1f} MB")
    else:
        print(f"  {f}/ (directory)")

print("\nDownload complete!")
