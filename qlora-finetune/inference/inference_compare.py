"""
Inference Comparison: Before vs After Fine-Tuning
==================================================
对比 Qwen2.5-1.5B-Instruct 微调前后的推理效果
"""
import os
import json
import torch
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
from datetime import datetime

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-1.5B-Instruct")
LORA_PATH = os.path.join(BASE_DIR, "output", "qwen2.5-lora", "final_lora")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test prompts covering different types of tasks
TEST_PROMPTS = [
    {
        "id": "coding",
        "instruction": "用Python写一个快速排序算法",
        "input": "",
        "category": "代码生成",
    },
    {
        "id": "knowledge",
        "instruction": "什么是深度学习中的梯度下降算法？",
        "input": "",
        "category": "知识问答",
    },
    {
        "id": "life",
        "instruction": "推荐几种健康的早餐搭配",
        "input": "",
        "category": "生活建议",
    },
    {
        "id": "reasoning",
        "instruction": "如果所有的猫都是动物，所有的动物都需要水，那么所有的猫都需要水吗？请解释你的推理过程。",
        "input": "",
        "category": "逻辑推理",
    },
    {
        "id": "explain",
        "instruction": "请解释大语言模型中LoRA微调的原理",
        "input": "",
        "category": "技术解释",
    },
]

print("=" * 60)
print("Qwen2.5-1.5B-Instruct: Before vs After Fine-Tuning")
print("=" * 60)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# === Load Base Model (4-bit) ===
print("\nLoading base model (4-bit)...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

# === Load Fine-Tuned Model ===
print("Loading fine-tuned model (base + LoRA)...")
finetuned_model = PeftModel.from_pretrained(base_model, LORA_PATH)
finetuned_model.eval()

print("\nBoth models loaded. Starting inference comparison...\n")

def generate_response(model, tokenizer, instruction, input_text="", max_length=256):
    """Generate response from model"""
    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    messages = [
        {"role": "system", "content": "你是一个有用的AI助手，请准确、简洁地回答用户的问题。"},
        {"role": "user", "content": user_content},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_length,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()


# === Run Comparison ===
results = []

for prompt in TEST_PROMPTS:
    print(f"\n{'='*60}")
    print(f"Test [{prompt['id']}] ({prompt['category']})")
    print(f"Instruction: {prompt['instruction'][:80]}...")
    print("-" * 60)

    # Use base model ONLY (disabling LoRA)
    base_model.disable_adapter_layers()
    print("\n[Before Fine-Tuning - Base Model]")
    base_response = generate_response(base_model, tokenizer, prompt["instruction"], prompt["input"])
    print(f"Response: {base_response[:300]}")

    # Use fine-tuned model (with LoRA)
    base_model.enable_adapter_layers()
    print("\n[After Fine-Tuning - Fine-Tuned Model]")
    ft_response = generate_response(base_model, tokenizer, prompt["instruction"], prompt["input"])
    print(f"Response: {ft_response[:300]}")

    results.append({
        "id": prompt["id"],
        "category": prompt["category"],
        "instruction": prompt["instruction"],
        "input": prompt["input"],
        "base_response": base_response,
        "finetuned_response": ft_response,
    })

# === Save Results ===
results_path = os.path.join(OUTPUT_DIR, "inference_comparison.json")
with open(results_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nResults saved to: {results_path}")

# === Generate Comparison Visualization ===
print("\nGenerating comparison visualization...")

# Create a summary table figure
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis("off")

table_data = []
headers = ["类别", "测试问题（摘要）", "基础模型回答长度", "微调模型回答长度", "主要变化"]

for r in results:
    table_data.append([
        r["category"],
        r["instruction"][:40] + "...",
        f"{len(r['base_response'])} chars",
        f"{len(r['finetuned_response'])} chars",
        "更详细" if len(r['finetuned_response']) > len(r['base_response']) * 1.2 else
        "更简洁" if len(r['finetuned_response']) < len(r['base_response']) * 0.8 else
        "长度相近",
    ])

table = ax.table(
    cellText=table_data,
    colLabels=headers,
    cellLoc="left",
    loc="center",
    colWidths=[0.12, 0.30, 0.18, 0.18, 0.12],
)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.2, 1.5)

# Style the table
for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor("#40466e")
        cell.set_text_props(color="white", fontweight="bold")
    elif row % 2 == 0:
        cell.set_facecolor("#f0f0f0")

ax.set_title(
    "Qwen2.5-1.5B-Instruct Fine-Tuning Inference Comparison\n"
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
    fontsize=14,
    fontweight="bold",
    pad=20,
)

table_path = os.path.join(OUTPUT_DIR, "inference_comparison_table.png")
plt.savefig(table_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Comparison table saved to: {table_path}")

# === Detailed Side-by-Side Comparison ===
fig, axes = plt.subplots(len(results), 1, figsize=(14, 3.5 * len(results)))
if len(results) == 1:
    axes = [axes]

for i, (ax, r) in enumerate(zip(axes, results)):
    ax.axis("off")

    base_short = r['base_response'][:200].replace('\n', ' ')
    ft_short = r['finetuned_response'][:200].replace('\n', ' ')

    comparison_text = (
        f"[{r['category']}] {r['instruction'][:60]}...\n\n"
        f"基础模型: {base_short}...\n\n"
        f"微调模型: {ft_short}..."
    )

    ax.text(
        0.5, 0.5, comparison_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="center",
        horizontalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8),
        wrap=True,
    )
    ax.set_title(f"Comparison {i+1}: {r['category']}", fontsize=12, fontweight="bold")

plt.tight_layout()
detail_path = os.path.join(OUTPUT_DIR, "inference_detail_comparison.png")
plt.savefig(detail_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Detail comparison saved to: {detail_path}")

# === Generate Perplexity Comparison ===
print("\nComputing perplexity comparison...")

test_texts = [
    "人工智能是计算机科学的一个重要分支，它研究如何让计算机模拟人类智能行为。",
    "Python是一种广泛使用的高级编程语言，以其简洁易读的语法而闻名。",
    "深度学习模型通常需要大量的训练数据和计算资源才能达到良好的性能。",
    "微调（Fine-tuning）是迁移学习中的一种技术，通过在预训练模型上进行额外训练来适应特定任务。",
]

def compute_perplexity(model, tokenizer, texts):
    """Compute perplexity for a list of texts"""
    perplexities = []
    for text in texts:
        encodings = tokenizer(text, return_tensors="pt")
        input_ids = encodings.input_ids.to(model.device)

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss
            if loss is not None:
                ppl = torch.exp(loss).item()
                perplexities.append(ppl)
    return perplexities

# Base model perplexity
base_model.disable_adapter_layers()
base_ppl = compute_perplexity(base_model, tokenizer, test_texts)
print(f"  Base model perplexity: {[f'{p:.2f}' for p in base_ppl]}")

# Fine-tuned model perplexity
base_model.enable_adapter_layers()
ft_ppl = compute_perplexity(base_model, tokenizer, test_texts)
print(f"  Fine-tuned model perplexity: {[f'{p:.2f}' for p in ft_ppl]}")

# Plot perplexity comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
ax1 = axes[0]
x = range(len(test_texts))
width = 0.35
labels = [f"Text {i+1}" for i in range(len(test_texts))]

bars1 = ax1.bar([i - width/2 for i in x], base_ppl, width, label="Base Model", color="#5B9BD5")
bars2 = ax1.bar([i + width/2 for i in x], ft_ppl, width, label="Fine-Tuned Model", color="#ED7D31")

for bar, val in zip(bars1, base_ppl):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}",
             ha="center", fontsize=9, fontweight="bold")
for bar, val in zip(bars2, ft_ppl):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, f"{val:.1f}",
             ha="center", fontsize=9, fontweight="bold")

ax1.set_xticks(x)
ax1.set_xticklabels(labels)
ax1.set_ylabel("Perplexity (lower is better)", fontsize=12)
ax1.set_title("Perplexity Comparison: Base vs Fine-Tuned", fontsize=14, fontweight="bold")
ax1.legend(fontsize=10)
ax1.grid(axis="y", alpha=0.3)

# Average comparison
ax2 = axes[1]
avg_base = sum(base_ppl) / len(base_ppl)
avg_ft = sum(ft_ppl) / len(ft_ppl)

bars = ax2.bar(["Base Model", "Fine-Tuned Model"], [avg_base, avg_ft],
               color=["#5B9BD5", "#ED7D31"], width=0.4)

for bar, val in zip(bars, [avg_base, avg_ft]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f"{val:.2f}",
             ha="center", fontsize=14, fontweight="bold")

improvement = (avg_base - avg_ft) / avg_base * 100
ax2.set_ylabel("Average Perplexity", fontsize=12)
ax2.set_title(f"Average Perplexity (Improvement: {improvement:.1f}%)", fontsize=14, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

plt.tight_layout()
ppl_path = os.path.join(OUTPUT_DIR, "perplexity_comparison.png")
plt.savefig(ppl_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Perplexity comparison saved to: {ppl_path}")

print("\n" + "=" * 60)
print("Inference comparison complete!")
print(f"All outputs saved to: {OUTPUT_DIR}")
print("=" * 60)
