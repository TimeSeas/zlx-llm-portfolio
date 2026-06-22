"""
Qwen2.5-1.5B-Instruct QLoRA Fine-Tuning
========================================
微调方法: QLoRA (4-bit量化 + LoRA低秩适配)
训练框架: TRL SFTTrainer
环境: NVIDIA RTX 4060 Laptop 8GB, CUDA 12.4, PyTorch 2.6.0
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import sys, json, torch, random, time
from datetime import datetime

# =====================================================
# CONFIGURATION
# =====================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "qwen2.5-lora")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Qwen2.5-1.5B-Instruct QLoRA Fine-Tuning")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
t_start = time.time()

# =====================================================
# STEP 1: Tokenizer
# =====================================================
print("\n[1/4] Loading tokenizer...")
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
print(f"  Tokenizer loaded. Vocab size: {len(tokenizer)}")

# =====================================================
# STEP 2: Model with 4-bit QLoRA
# =====================================================
print("\n[2/4] Loading model with 4-bit QLoRA...")
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
    max_memory={0: "6GB", "cpu": "16GB"},
)
vram = torch.cuda.memory_allocated() / 1e9
print(f"  Model loaded. VRAM: {vram:.1f} GB")

model = prepare_model_for_kbit_training(model)

lora = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, lora)
model.print_trainable_parameters()
model.config.use_cache = False

# =====================================================
# STEP 3: Dataset (manual, no HF datasets library)
# =====================================================
print("\n[3/4] Preparing dataset...")

training_data = [
    ("用Python写一个计算斐波那契数列的函数", "", "def fibonacci(n):\n    if n <= 0: return []\n    elif n == 1: return [0]\n    elif n == 2: return [0, 1]\n    fib = [0, 1]\n    for i in range(2, n):\n        fib.append(fib[-1] + fib[-2])\n    return fib\n\nprint(fibonacci(10))  # [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]"),
    ("解释什么是递归函数", "", "递归函数是指在函数内部调用自身的函数。包含两个关键部分：\n1. 基准条件（Base Case）：终止递归的条件\n2. 递归条件（Recursive Case）：函数调用自身\n\n例如阶乘：\ndef factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)"),
    ("Python中列表和元组的区别", "", "主要区别：\n1. 可变性：列表可变（可增删改），元组不可变\n2. 语法：列表用[]，元组用()\n3. 性能：元组更轻量，访问更快\n4. 使用场景：元组适合不可变数据，列表适合动态数据集合\n5. 哈希性：元组可作字典键，列表不能"),
    ("推荐几道简单易做的家常菜", "", "家常菜推荐：\n1. 番茄炒蛋：鸡蛋炒熟盛出，番茄切块炒出汁，混合调味\n2. 青椒肉丝：猪肉切丝腌制，与青椒丝翻炒\n3. 麻婆豆腐：豆腐焯水，炒香肉末豆瓣酱，炖煮勾芡\n4. 蒜蓉西兰花：西兰花焯水，蒜末爆香翻炒"),
    ("如何保持健康的生活习惯", "", "健康习惯建议：\n1. 均衡饮食：多吃蔬果，适量蛋白质，减少高糖高脂\n2. 规律运动：每周至少150分钟中等强度运动\n3. 充足睡眠：每天7-8小时，规律作息\n4. 心理健康：学会压力管理，保持积极心态\n5. 定期体检：每年一次健康检查"),
    ("什么是人工智能", "", "人工智能（AI）是计算机科学分支，旨在创建模拟人类智能的系统。主要领域：\n1. 机器学习：从数据中学习模式\n2. 深度学习：使用多层神经网络\n3. 自然语言处理（NLP）：理解和生成语言\n4. 计算机视觉：分析图像视频\n5. 大语言模型：如GPT、Qwen等\n已广泛应用于医疗诊断、自动驾驶、智能助手、推荐系统等领域。"),
    ("解释过拟合现象及解决方法", "", "过拟合（Overfitting）指模型在训练数据上表现好但新数据上表现差。\n产生原因：模型过于复杂、训练数据不足\n解决方法：\n1. 正则化（L1/L2）：限制权重大小\n2. Dropout：随机丢弃神经元\n3. 早停（Early Stopping）：验证性能不再提升时停止\n4. 数据增强：扩充训练数据\n5. 交叉验证：多折验证评估"),
    ("什么是迁移学习", "", "迁移学习（Transfer Learning）是将一个任务上学到的知识应用到另一个相关任务上。\n核心思想：预训练模型学习通用特征，迁移到下游任务，通过微调适应新任务。\n常见方法：\n1. 特征提取：冻结预训练层，只训练分类层\n2. 全模型微调：在新数据上微调整个模型\n3. LoRA/QLoRA：参数高效微调大模型\n本实验正是迁移学习的应用——在Qwen2.5预训练模型上进行指令微调。"),
    ("介绍几种常见的排序算法", "", "常见排序算法：\n1. 冒泡排序：O(n²)，稳定，简单但效率低\n2. 快速排序：O(n log n)平均，实际最快之一，不稳定\n3. 归并排序：O(n log n)，稳定，需额外O(n)空间\n4. 插入排序：O(n²)，小规模或近乎有序数据效率高\n5. 堆排序：O(n log n)，不稳定，原地排序O(1)空间\n选择时需考虑数据规模、稳定性要求和内存限制。"),
    ("用Python实现二分查找算法", "", "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = left + (right - left) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n\n# 时间复杂度：O(log n)，空间复杂度：O(1)\narr = [1, 3, 5, 7, 9, 11, 13]\nprint(binary_search(arr, 7))  # 输出: 3"),
    ("什么是大语言模型的微调", "", "微调（Fine-tuning）是在预训练模型基础上使用特定领域数据进行进一步训练。\n主要方法：\n1. 全参数微调：更新所有参数，效果最好但资源消耗大\n2. LoRA（Low-Rank Adaptation）：添加低秩矩阵，高效微调\n3. QLoRA：4-bit量化+LoRA，使消费级GPU也能微调大模型\n4. Adapter：在Transformer层间插入小型网络\n\nQLoRA是目前最实用的微调方法之一，本实验正是采用此方法对Qwen2.5进行微调。"),
    ("解释深度学习中的梯度下降算法", "", "梯度下降是深度学习的核心优化算法。\n基本原理：\n1. 计算损失函数关于参数的梯度\n2. 沿负梯度方向更新参数：θ = θ - lr × ∇L(θ)\n3. 学习率（lr）控制更新步长\n\n常见变体：\n- SGD（随机梯度下降）：每次用单个或小批量样本\n- Adam（自适应矩估计）：自适应调整学习率\n- AdamW：带权重衰减的Adam\n\n本实验使用Paged AdamW 8-bit优化器，在保持效果的同时降低显存占用。"),
]

# Format as Qwen2.5 chat template
def format_pair(inst, inp, out):
    user = f"{inst}\n\n{inp}" if inp else inst
    return (
        f"<|im_start|>system\n"
        f"你是一个有用的AI助手，请准确、简洁地回答用户的问题。<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n{out}<|im_end|>"
    )

formatted = [format_pair(inst, inp, out) for inst, inp, out in training_data]

# Add some variants
for i in range(5):
    inst, inp, out = training_data[i]
    if "用Python" in inst:
        formatted.append(format_pair(inst.replace("用Python", "请用Python代码"), inp, out))

# Simple train/eval split
random.seed(42)
indices = list(range(len(formatted)))
random.shuffle(indices)
split_n = int(len(indices) * 0.85)
train_data = [formatted[i] for i in indices[:split_n]]
eval_data = [formatted[i] for i in indices[split_n:]]
print(f"  Train: {len(train_data)}, Eval: {len(eval_data)}")

# =====================================================
# STEP 4: Train with SFTTrainer
# =====================================================
print("\n[4/4] Starting QLoRA training...")
from trl import SFTTrainer

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    warmup_steps=5,
    logging_steps=2,
    save_steps=100,
    eval_strategy="steps",
    eval_steps=100,
    save_total_limit=2,
    fp16=True,
    gradient_checkpointing=True,
    optim="paged_adamw_8bit",
    report_to="none",
    dataloader_num_workers=0,
)

# Minimal Dataset-compatible wrapper (avoids HF datasets segfaults on Windows)
class SimpleDataset:
    def __init__(self, data, column_names=None):
        self._data = data
        self.column_names = column_names or ["text"]

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        return {self.column_names[0]: self._data[idx]}

    def __iter__(self):
        for item in self._data:
            yield {self.column_names[0]: item}

train_dataset = SimpleDataset(train_data)
eval_dataset = SimpleDataset(eval_data)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    processing_class=tokenizer,
)

print(f"  Starting train() at {datetime.now().strftime('%H:%M:%S')}...")
trainer.train()
elapsed = (time.time() - t_start) / 60
print(f"  Training completed in {elapsed:.1f} minutes!")

# =====================================================
# SAVE
# =====================================================
print("\nSaving model and metrics...")
lora_path = os.path.join(OUTPUT_DIR, "final_lora")
model.save_pretrained(lora_path)
tokenizer.save_pretrained(lora_path)
print(f"  LoRA adapter: {lora_path}")

# Save training history
history = {
    "model": "Qwen2.5-1.5B-Instruct",
    "method": "QLoRA (4-bit NF4 + LoRA r=8)",
    "train_samples": len(train_data),
    "eval_samples": len(eval_data),
    "epochs": 3,
    "learning_rate": 2e-4,
    "batch_size": 1,
    "gradient_accumulation": 8,
    "log_history": trainer.state.log_history if hasattr(trainer, 'state') else [],
}
with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w", encoding="utf-8") as f:
    json.dump(history, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"Done! Total time: {elapsed:.1f} min")
print(f"Output: {OUTPUT_DIR}")
print(f"{'='*60}")
