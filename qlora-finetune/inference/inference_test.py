"""
Inference comparison: Base Qwen2.5 vs Fine-tuned model
"""
import os, json, torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-1.5B-Instruct")
LORA_PATH = os.path.join(BASE_DIR, "output", "qwen2.5-lora", "final_lora")
OUT_DIR = os.path.join(BASE_DIR, "output")

print("=" * 60)
print("Inference Comparison: Base vs Fine-Tuned")
print("=" * 60)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

print("\nLoading tokenizer...")
tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

print("Loading base model (4-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.float16,
    max_memory={0: "6GB", "cpu": "16GB"},
)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

# Test prompts
test_prompts = [
    ("代码生成", "用Python写一个冒泡排序算法"),
    ("知识问答", "什么是深度学习中的梯度下降算法？"),
    ("生活建议", "推荐几种健康的早餐搭配"),
    ("技术解释", "请解释大语言模型微调中QLoRA方法的原理"),
    ("逻辑推理", "所有鸟类都有羽毛，企鹅是鸟类，企鹅有羽毛吗？请解释推理过程。"),
]

def generate(prompt, use_lora=True, max_tokens=200):
    if use_lora:
        model.enable_adapter_layers()
    else:
        model.disable_adapter_layers()

    text = f"<|im_start|>system\n你是一个有用的AI助手。<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tok(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=max_tokens, temperature=0.7,
            top_p=0.9, do_sample=True, pad_token_id=tok.pad_token_id,
        )

    response = tok.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

results = []
print("\nRunning comparisons...")
for category, prompt in test_prompts:
    print(f"\n{'='*60}")
    print(f"[{category}] {prompt[:60]}...")
    print("-" * 60)

    # Base model (disable LoRA)
    print("\n[Base Model (No Fine-Tuning)]")
    base_resp = generate(prompt, use_lora=False)
    print(base_resp[:200])

    # Fine-tuned (enable LoRA)
    print("\n[Fine-Tuned Model (QLoRA)]")
    ft_resp = generate(prompt, use_lora=True)
    print(ft_resp[:200])

    results.append({
        "category": category,
        "prompt": prompt,
        "base_response": base_resp,
        "finetuned_response": ft_resp,
    })

# Save results
with open(os.path.join(OUT_DIR, "inference_results.json"), "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved.")

# === Generate Comparison Visualization ===
print("\nGenerating comparison visualization...")
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis("off")

data = []
for r in results:
    base_len = len(r["base_response"])
    ft_len = len(r["finetuned_response"])
    data.append([
        r["category"],
        r["prompt"][:35] + "...",
        f"{base_len} chars",
        f"{ft_len} chars",
        "More detailed" if ft_len > base_len * 1.3 else "More concise" if ft_len < base_len * 0.7 else "Similar length",
    ])

table = ax.table(
    cellText=data,
    colLabels=["Category", "Prompt", "Base Model", "Fine-Tuned", "Change"],
    cellLoc="left",
    loc="center",
    colWidths=[0.10, 0.25, 0.15, 0.15, 0.15],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.8)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#40466e")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f0f0f0")

ax.set_title("Qwen2.5-1.5B Fine-Tuning: Before vs After Inference Comparison", fontsize=14, fontweight="bold", pad=20)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "inference_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: inference_comparison.png")

# === Response Length Bar Chart ===
fig, ax = plt.subplots(figsize=(12, 6))
categories = [r["category"] for r in results]
base_lens = [len(r["base_response"]) for r in results]
ft_lens = [len(r["finetuned_response"]) for r in results]

x = np.arange(len(categories))
width = 0.35
ax.bar(x - width/2, base_lens, width, label="Base Model", color="#5B9BD5")
ax.bar(x + width/2, ft_lens, width, label="Fine-Tuned Model", color="#ED7D31")

for i, (b, f) in enumerate(zip(base_lens, ft_lens)):
    ax.text(i - width/2, b + 5, str(b), ha="center", fontsize=9, fontweight="bold")
    ax.text(i + width/2, f + 5, str(f), ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylabel("Response Length (characters)", fontsize=12)
ax.set_title("Response Length Comparison: Base vs Fine-Tuned Model", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "response_length_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: response_length_comparison.png")

print("\nDone! All inference comparisons complete.")
