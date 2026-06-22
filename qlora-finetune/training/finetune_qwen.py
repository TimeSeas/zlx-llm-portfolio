"""
Qwen2.5-1.5B-Instruct QLoRA Fine-Tuning Script
===============================================
微调方法: QLoRA (4-bit量化 + LoRA低秩适配)
训练框架: TRL SFTTrainer
"""
import os
import sys
import json
import torch
import numpy as np
from datetime import datetime
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt

# === Configuration ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "qwen2.5-lora")
LOG_DIR = os.path.join(BASE_DIR, "output", "logs")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Training hyperparameters
CONFIG = {
    "model_name": "Qwen2.5-1.5B-Instruct",
    "method": "QLoRA (4-bit NF4 quantization + LoRA)",
    "lora_r": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.05,
    "learning_rate": 2e-4,
    "num_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.03,
    "logging_steps": 10,
    "save_steps": 200,
    "eval_steps": 200,
    "max_seq_length": 512,
    "bf16": False,  # RTX 4060 may not support bf16 well
    "fp16": True,
}

print("=" * 60)
print("Qwen2.5-1.5B-Instruct QLoRA Fine-Tuning")
print("=" * 60)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Config: {json.dumps(CONFIG, indent=2, ensure_ascii=False)}")

# === Step 1: Load Model with 4-bit Quantization ===
print("\n[Step 1] Loading model with 4-bit quantization...")

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForSeq2Seq,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from datasets import Dataset, load_dataset
from trl import SFTTrainer

# 4-bit quantization config (NF4)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    padding_side="right",
)
# Qwen2.5 tokenizer may not have pad_token by default
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading base model (4-bit)...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    max_memory={0: "6GB", "cpu": "16GB"},
)

# Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)
model.config.use_cache = False  # Required for gradient checkpointing

# === Step 2: Configure LoRA ===
print("\n[Step 2] Configuring LoRA...")

lora_config = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=CONFIG["lora_dropout"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Step 3: Prepare Dataset ===
print("\n[Step 3] Preparing training dataset...")

def format_instruction(example):
    """Format instruction into Qwen2.5 chat template"""
    instruction = example.get("instruction", "")
    input_text = example.get("input", "")
    output = example.get("output", "")

    if input_text:
        user_content = f"{instruction}\n\n{input_text}"
    else:
        user_content = instruction

    # Qwen2.5 chat format
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手，请准确、简洁地回答用户的问题。"},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": output},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


def load_training_data():
    """Load Chinese instruction dataset"""
    # Try multiple dataset sources
    dataset_sources = [
        # Alpaca-zh from HuggingFace
        ("llm-wizard/alpaca-gpt4-data-zh", None),
        # Try Belle dataset
        ("BelleGroup/train_0.5M_CN", 10000),  # Only first 10k
        # Fallback: use a tiny custom dataset
    ]

    for source, max_samples in dataset_sources:
        try:
            print(f"  Trying to load: {source}...")
            dataset = load_dataset(source, split="train")
            if max_samples:
                dataset = dataset.select(range(min(max_samples, len(dataset))))
            # Take 1000 samples for training
            dataset = dataset.select(range(min(1000, len(dataset))))
            print(f"  Loaded {len(dataset)} samples from {source}")
            return dataset
        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # Ultimate fallback: create a small custom Chinese dataset
    print("  Using built-in Chinese instruction dataset...")
    return create_fallback_dataset()


def create_fallback_dataset():
    """Create a fallback Chinese instruction dataset for demonstration"""
    data = []

    samples = [
        # 编程类
        {"instruction": "用Python写一个计算斐波那契数列的函数", "input": "", "output": "def fibonacci(n):\n    if n <= 0:\n        return []\n    elif n == 1:\n        return [0]\n    elif n == 2:\n        return [0, 1]\n    \n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[-1] + fib[-2])\n    return fib\n\n# 使用示例\nprint(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"},
        {"instruction": "解释什么是递归函数", "input": "", "output": "递归函数是指在函数内部调用自身的函数。它通常包含两个关键部分：\n1. 基准条件（Base Case）：终止递归的条件，防止无限递归\n2. 递归条件（Recursive Case）：函数调用自身的部分\n\n例如，计算阶乘的递归函数：\ndef factorial(n):\n    if n <= 1:  # 基准条件\n        return 1\n    return n * factorial(n - 1)  # 递归条件"},
        {"instruction": "Python中列表和元组的区别是什么？", "input": "", "output": "列表（list）和元组（tuple）的主要区别：\n\n1. 可变性：列表是可变的（可以增删改元素），元组是不可变的（创建后不能修改）\n2. 语法：列表用方括号[]，元组用圆括号()\n3. 性能：元组比列表更轻量，访问速度更快\n4. 使用场景：元组适合存储不应修改的数据（如坐标、配置），列表适合需要动态修改的数据集合\n5. 哈希性：元组可作为字典的键（当元素都不可变时），列表不能"},

        # 生活类
        {"instruction": "推荐几道简单易做的家常菜", "input": "", "output": "以下是几道简单易做的家常菜：\n\n1. 番茄炒蛋：将鸡蛋打散炒熟盛出，番茄切块下锅炒至出汁，加入炒好的鸡蛋，调味即可\n2. 青椒肉丝：猪肉切丝用淀粉腌制，青椒切丝，先炒肉丝变色后加青椒翻炒调味\n3. 麻婆豆腐：豆腐切块焯水，炒香肉末和豆瓣酱，加入豆腐和少量水炖煮，勾芡出锅\n4. 蒜蓉西兰花：西兰花焯水，蒜末爆香后加入西兰花翻炒调味"},
        {"instruction": "如何保持健康的生活习惯？", "input": "", "output": "保持健康生活习惯的建议：\n\n1. 均衡饮食：多吃蔬菜水果，适量摄入蛋白质，减少高糖高脂食物\n2. 规律运动：每周至少150分钟中等强度运动，如快走、慢跑、游泳\n3. 充足睡眠：成年人每天7-8小时，保持规律的作息时间\n4. 心理健康：学会压力管理，保持积极心态，必要时寻求帮助\n5. 定期体检：每年进行一次健康体检，及早发现潜在问题\n6. 戒烟限酒：避免吸烟，限制酒精摄入"},

        # 知识类
        {"instruction": "什么是人工智能？", "input": "", "output": "人工智能（Artificial Intelligence，简称AI）是计算机科学的一个分支，旨在创建能够模拟人类智能的系统。主要包括：\n\n1. 机器学习（Machine Learning）：让计算机从数据中学习模式\n2. 深度学习（Deep Learning）：使用多层神经网络处理复杂问题\n3. 自然语言处理（NLP）：让计算机理解和生成人类语言\n4. 计算机视觉（Computer Vision）：让计算机理解和分析图像视频\n5. 大语言模型（LLM）：如GPT、Qwen等，能够理解和生成自然语言文本\n\nAI已广泛应用于医疗诊断、自动驾驶、智能助手、推荐系统等领域。"},
        {"instruction": "解释一下深度学习中的过拟合现象", "input": "", "output": "过拟合（Overfitting）是深度学习中的常见问题，指模型在训练数据上表现很好，但在新数据上表现差的现象。\n\n产生原因：\n1. 模型过于复杂，参数量过多\n2. 训练数据太少或不够多样化\n3. 训练时间过长\n\n解决方法：\n1. 正则化（Regularization）：L1/L2正则化限制权重大小\n2. Dropout：训练时随机丢弃部分神经元\n3. 早停（Early Stopping）：在验证集性能不再提升时停止训练\n4. 数据增强（Data Augmentation）：扩充训练数据\n5. 交叉验证（Cross Validation）：使用多折验证评估模型"},
        {"instruction": "什么是迁移学习？", "input": "", "output": "迁移学习（Transfer Learning）是一种机器学习方法，将在一个任务（源任务）上学到的知识应用到另一个相关任务（目标任务）上。\n\n核心思想：\n1. 预训练模型在大型数据集上学习通用特征\n2. 将这些学到的知识迁移到特定的下游任务\n3. 通过微调（Fine-tuning）适应新任务\n\n常见方法：\n1. 特征提取：冻结预训练模型的大部分层，只训练最后的分类层\n2. 全模型微调：在新数据上微调整个模型\n3. LoRA/QLoRA：使用低秩矩阵高效微调大模型\n\n本实验就是迁移学习的典型应用——在Qwen2.5预训练模型基础上进行指令微调。"},
        {"instruction": "介绍几种常见的排序算法", "input": "", "output": "常见排序算法及其特点：\n\n1. 冒泡排序（Bubble Sort）\n   时间：O(n²) | 稳定 | 简单但效率低\n\n2. 快速排序（Quick Sort）\n   时间：O(n log n)平均 | 不稳定 | 实际应用中最快之一\n\n3. 归并排序（Merge Sort）\n   时间：O(n log n) | 稳定 | 需要额外O(n)空间\n\n4. 插入排序（Insertion Sort）\n   时间：O(n²) | 稳定 | 小规模数据或近乎有序数据效率高\n\n5. 堆排序（Heap Sort）\n   时间：O(n log n) | 不稳定 | 原地排序，空间O(1)\n\n选择排序算法时需考虑数据规模、稳定性要求和内存限制。"},
        {"instruction": "用Python实现二分查找", "input": "", "output": "二分查找（Binary Search）针对有序数组的高效查找算法：\n\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    \n    while left <= right:\n        mid = left + (right - left) // 2\n        \n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    \n    return -1  # 未找到\n\n# 使用示例\narr = [1, 3, 5, 7, 9, 11, 13]\nprint(binary_search(arr, 7))   # 输出: 3\nprint(binary_search(arr, 8))   # 输出: -1\n\n时间复杂度：O(log n)，空间复杂度：O(1)"},
        {"instruction": "什么是大语言模型的微调？", "input": "", "output": "大语言模型微调（Fine-tuning）是指在预训练好的大语言模型基础上，使用特定领域或任务的数据进行进一步训练，使模型更好地适应特定需求。\n\n微调的主要方法：\n\n1. 全参数微调（Full Fine-tuning）\n   - 更新模型所有参数\n   - 效果最好但资源消耗大\n\n2. 参数高效微调（PEFT）\n   - LoRA（Low-Rank Adaptation）：在原始权重旁添加低秩矩阵\n   - QLoRA：结合4-bit量化的LoRA，大幅降低显存需求\n   - Adapter：在Transformer层间插入小型网络\n\n3. 指令微调（Instruction Tuning）\n   - 使用指令-回复对训练\n   - 提升模型遵循指令的能力\n\nQLoRA是目前最实用的微调方法之一，使消费者级GPU也能微调大模型。本实验正是采用此方法。"},
    ]

    # Expand the dataset with variations
    for s in samples:
        data.append(s)
        # Add a slightly modified version
        if s["instruction"].startswith("用Python"):
            data.append({
                "instruction": s["instruction"].replace("用Python", "请用Python代码"),
                "input": s["input"],
                "output": s["output"],
            })
        elif s["instruction"].startswith("介绍"):
            data.append({
                "instruction": s["instruction"].replace("介绍", "请介绍"),
                "input": s["input"],
                "output": s["output"],
            })

    return Dataset.from_list(data)


# Load data
raw_dataset = load_training_data()
print(f"  Total samples: {len(raw_dataset)}")

# Format to Qwen2.5 chat template
formatted_dataset = raw_dataset.map(format_instruction, remove_columns=raw_dataset.column_names)
print(f"  Formatted samples: {len(formatted_dataset)}")

# Split into train/eval
split_dataset = formatted_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
print(f"  Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

# === Step 4: Train with SFTTrainer ===
print("\n[Step 4] Starting QLoRA fine-tuning...")

# Track training loss for visualization
train_losses = []
eval_losses = []
step_times = []

class LossCallback(TrainerCallback):
    """Record training losses for visualization"""
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs:
                train_losses.append({
                    "step": state.global_step,
                    "loss": logs["loss"],
                })
            if "eval_loss" in logs:
                eval_losses.append({
                    "step": state.global_step,
                    "eval_loss": logs["eval_loss"],
                })

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=CONFIG["num_epochs"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_ratio=CONFIG["warmup_ratio"],
    logging_steps=CONFIG["logging_steps"],
    logging_dir=LOG_DIR,
    save_steps=CONFIG["save_steps"],
    eval_steps=CONFIG["eval_steps"],
    eval_strategy="steps",
    save_total_limit=2,
    fp16=CONFIG["fp16"],
    bf16=CONFIG["bf16"],
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="tensorboard",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    group_by_length=True,
    dataloader_num_workers=0,
    remove_unused_columns=False,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    max_seq_length=CONFIG["max_seq_length"],
    dataset_text_field="text",
    packing=False,
    callbacks=[LossCallback()],
)

# Start training
print(f"\n  Training config: epochs={CONFIG['num_epochs']}, "
      f"batch_size={CONFIG['per_device_train_batch_size']}×{CONFIG['gradient_accumulation_steps']}, "
      f"lr={CONFIG['learning_rate']}")
print(f"  Effective batch size: {CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']}")
print("-" * 60)

start_time = datetime.now()
trainer.train()
end_time = datetime.now()

training_duration = (end_time - start_time).total_seconds()
print(f"\n  Training completed in {training_duration:.1f}s ({training_duration/60:.1f} min)")

# === Step 5: Save Model & Metrics ===
print("\n[Step 5] Saving model and metrics...")

# Save LoRA adapter
lora_save_path = os.path.join(OUTPUT_DIR, "final_lora")
model.save_pretrained(lora_save_path)
tokenizer.save_pretrained(lora_save_path)
print(f"  LoRA adapter saved to: {lora_save_path}")

# Save merged model (optional, for easier inference)
print("  Merging LoRA weights with base model...")
try:
    merged_model = model.merge_and_unload()
    merged_save_path = os.path.join(OUTPUT_DIR, "merged_model")
    merged_model.save_pretrained(merged_save_path, safe_serialization=True)
    tokenizer.save_pretrained(merged_save_path)
    print(f"  Merged model saved to: {merged_save_path}")
except Exception as e:
    print(f"  Merge skipped (may be too large for RAM): {e}")

# Save training history
history_data = {
    "config": CONFIG,
    "train_losses": train_losses,
    "eval_losses": eval_losses,
    "training_duration_seconds": training_duration,
    "num_train_samples": len(train_dataset),
    "num_eval_samples": len(eval_dataset),
}
history_path = os.path.join(OUTPUT_DIR, "training_history.json")
with open(history_path, "w", encoding="utf-8") as f:
    json.dump(history_data, f, ensure_ascii=False, indent=2)
print(f"  Training history saved to: {history_path}")

# === Step 6: Plot Training Curves ===
print("\n[Step 6] Generating training curves...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax1 = axes[0]
if train_losses:
    steps = [x["step"] for x in train_losses]
    losses = [x["loss"] for x in train_losses]
    ax1.plot(steps, losses, "b-", linewidth=1.5, label="Training Loss")

    if eval_losses:
        eval_steps = [x["step"] for x in eval_losses]
        eval_ls = [x["eval_loss"] for x in eval_losses]
        ax1.plot(eval_steps, eval_ls, "r-o", markersize=4, linewidth=1.5, label="Evaluation Loss")

    ax1.set_xlabel("Training Steps", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("QLoRA Fine-Tuning Loss Curve", fontsize=14, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

# Smoothed loss curve
ax2 = axes[1]
if len(train_losses) > 5:
    steps_arr = np.array(steps)
    losses_arr = np.array(losses)

    # Moving average smoothing
    window = max(len(losses_arr) // 10, 3)
    smoothed = np.convolve(losses_arr, np.ones(window)/window, mode="valid")
    smooth_steps = steps_arr[window-1:]

    ax2.plot(smooth_steps, smoothed, "g-", linewidth=2, label=f"Smoothed Loss (window={window})")
    ax2.set_xlabel("Training Steps", fontsize=12)
    ax2.set_ylabel("Smoothed Loss", fontsize=12)
    ax2.set_title("Smoothed Training Loss", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Add final loss annotation
    final_loss = smoothed[-1]
    ax2.annotate(
        f"Final: {final_loss:.4f}",
        xy=(smooth_steps[-1], final_loss),
        xytext=(smooth_steps[-1] * 0.7, final_loss + 0.1),
        arrowprops=dict(arrowstyle="->", color="darkgreen"),
        fontsize=11,
        color="darkgreen",
        fontweight="bold",
    )

plt.tight_layout()
plot_path = os.path.join(OUTPUT_DIR, "training_curves.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Training curves saved to: {plot_path}")

print("\n" + "=" * 60)
print("Fine-tuning complete!")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 60)
