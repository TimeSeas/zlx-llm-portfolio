"""
Qwen2.5-1.5B-Instruct QLoRA Fine-Tuning — Custom Training Loop
===============================================================
Avoids: datasets library, TRL SFTTrainer, Trainer, DataCollator — all known to segfault on Windows.
微调方法: QLoRA (4-bit量化 + LoRA低秩适配)
使用纯 PyTorch 训练循环
"""
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

import sys, json, torch, random, time, math
from datetime import datetime
from torch.utils.data import Dataset, DataLoader

print("=" * 60)
print("Qwen2.5-1.5B-Instruct QLoRA Fine-Tuning")
print(f"Start: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 60)
t0 = time.time()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "Qwen2.5-1.5B-Instruct")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "qwen2.5-lora")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "checkpoints"), exist_ok=True)

# =====================================================
# 1. TOKENIZER + MODEL (4-bit QLoRA)
# =====================================================
print("\n[1/3] Loading tokenizer + model...")
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
print(f"  Tokenizer: {len(tok)} tokens")

bnb = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, quantization_config=bnb, device_map="auto",
    trust_remote_code=True, torch_dtype=torch.float16,
    max_memory={0: "6GB", "cpu": "16GB"},
)
print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.1f}GB")
model = prepare_model_for_kbit_training(model)

peft_cfg = LoraConfig(
    r=8, lora_alpha=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_dropout=0.05, bias="none", task_type=TaskType.CAUSAL_LM,
)
model = get_peft_model(model, peft_cfg)
model.print_trainable_parameters()
model.config.use_cache = False
model.train()

# =====================================================
# 2. DATASET
# =====================================================
print("\n[2/3] Preparing dataset...")
raw_data = [
    ("用Python写斐波那契数列","","def fibonacci(n):\n    if n<=0:return[]\n    elif n==1:return[0]\n    elif n==2:return[0,1]\n    fib=[0,1]\n    for i in range(2,n):\n        fib.append(fib[-1]+fib[-2])\n    return fib"),
    ("解释递归函数","","递归函数在函数内部调用自身。包含基准条件(终止递归)和递归条件(调用自身)。\n例如阶乘:def factorial(n):return 1 if n<=1 else n*factorial(n-1)"),
    ("Python列表和元组的区别","","主要区别:\n1.可变性:列表可变可增删改,元组不可变\n2.语法:列表用[],元组用()\n3.性能:元组更轻量,访问更快\n4.使用场景:元组适合不可变数据,列表适合动态数据\n5.哈希性:元组可作字典键,列表不能"),
    ("推荐家常菜","","简单家常菜:\n1.番茄炒蛋:鸡蛋炒熟,番茄炒出汁混合\n2.青椒肉丝:肉丝腌制与青椒翻炒\n3.麻婆豆腐:豆腐焯水,炒肉末豆瓣酱炖煮勾芡\n4.蒜蓉西兰花:焯水后蒜末爆香翻炒"),
    ("保持健康习惯","","健康建议:\n1.均衡饮食:多蔬果适量蛋白,减少高糖高脂\n2.规律运动:每周150分钟中等强度\n3.充足睡眠:每天7-8小时,规律作息\n4.心理健康:压力管理,保持积极\n5.定期体检:每年一次"),
    ("什么是人工智能","","人工智能(AI)是计算机科学分支,模拟人类智能。包括:\n1.机器学习:从数据学习\n2.深度学习:多层神经网络\n3.自然语言处理:理解生成语言\n4.计算机视觉:分析图像视频\n5.大语言模型:如GPT、Qwen等\n应用于医疗、自动驾驶、智能助手等领域。"),
    ("解释过拟合","","过拟合:模型在训练数据表现好但新数据差。\n原因:模型复杂、数据不足\n解决:\n1.正则化L1/L2限制权重\n2.Dropout随机丢弃神经元\n3.早停验证不提升时停止\n4.数据增强扩充数据\n5.交叉验证多折评估"),
    ("什么是迁移学习","","迁移学习将一任务知识应用到另一任务。\n方法:\n1.特征提取:冻结预训练层,只训练分类层\n2.全模型微调:在新数据上微调整个模型\n3.LoRA/QLoRA:参数高效微调大模型\n本实验正是迁移学习应用——在Qwen2.5预训练模型上指令微调。"),
    ("常见排序算法","","1.冒泡O(n²)稳定简单\n2.快速O(n log n)平均最快,不稳定\n3.归并O(n log n)稳定,需O(n)空间\n4.插入O(n²)小规模数据好\n5.堆O(n log n)不稳定,原地O(1)\n选择考虑数据规模、稳定性、内存。"),
    ("Python二分查找","","def binary_search(arr,target):\n    l,r=0,len(arr)-1\n    while l<=r:\n        m=l+(r-l)//2\n        if arr[m]==target:return m\n        elif arr[m]<target:l=m+1\n        else:r=m-1\n    return -1\n# O(log n)时间 O(1)空间"),
    ("大语言模型微调方法","","微调在预训练模型上用特定数据训练。\n方法:\n1.全参数微调:效果最好但资源大\n2.LoRA:添加低秩矩阵,高效微调\n3.QLoRA:4bit量化+LoRA,消费级GPU可用\n4.Adapter:层间插入小型网络\nQLoRA最实用,本实验用此方法微调Qwen2.5。"),
    ("梯度下降算法原理","","梯度下降是核心优化算法。\n原理:计算损失梯度,沿负梯度更新θ=θ-lr×∇L(θ)\n变体:\n1.SGD随机梯度下降\n2.Adam自适应学习率\n3.AdamW权重衰减\n本实验用Paged AdamW 8-bit优化器。"),
]

def fmt(inst, inp, out):
    user = f"{inst}\n{inp}" if inp else inst
    return f"<|im_start|>system\n你是一个有用的AI助手。<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{out}<|im_end|>"

texts = [fmt(i, p, o) for i, p, o in raw_data]
for idx in range(4):
    if "用Python" in raw_data[idx][0]:
        texts.append(fmt(raw_data[idx][0].replace("用Python","请用Python代码"), raw_data[idx][1], raw_data[idx][2]))

random.seed(42)
random.shuffle(texts)
n = int(len(texts) * 0.85)
train_texts = texts[:n]
eval_texts = texts[n:]
print(f"  Samples: train={len(train_texts)}, eval={len(eval_texts)}")

# Pre-tokenize
MAX_LEN = 256
def encode(text_list):
    return tok(text_list, truncation=True, padding="max_length",
               max_length=MAX_LEN, return_tensors="pt")

train_enc = encode(train_texts)
eval_enc = encode(eval_texts)

class TextDataset(Dataset):
    def __init__(self, enc):
        self.ids = enc["input_ids"]
        self.mask = enc["attention_mask"]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        return {
            "input_ids": self.ids[i],
            "attention_mask": self.mask[i],
            "labels": self.ids[i].clone(),
        }

def collate_fn(batch):
    max_len = max(b["input_ids"].size(0) for b in batch)
    max_len = ((max_len + 7) // 8) * 8  # pad to multiple of 8
    pad_id = tok.pad_token_id
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for b in batch:
        pl = max_len - b["input_ids"].size(0)
        out["input_ids"].append(torch.nn.functional.pad(b["input_ids"], (0, pl), value=pad_id))
        out["attention_mask"].append(torch.nn.functional.pad(b["attention_mask"], (0, pl), value=0))
        out["labels"].append(torch.nn.functional.pad(b["labels"], (0, pl), value=-100))
    return {k: torch.stack(v) for k, v in out.items()}

train_ds = TextDataset(train_enc)
eval_ds = TextDataset(eval_enc)
train_loader = DataLoader(train_ds, batch_size=2, shuffle=True, collate_fn=collate_fn)
eval_loader = DataLoader(eval_ds, batch_size=2, shuffle=False, collate_fn=collate_fn)

# =====================================================
# 3. TRAINING LOOP (custom)
# =====================================================
print("\n[3/3] Training...")
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

EPOCHS = 3
LR = 2e-4
GRAD_ACCUM = 4  # effective batch = 2 * 4 = 8

optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS // GRAD_ACCUM
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=5, num_training_steps=total_steps)

scaler = torch.cuda.amp.GradScaler()
history = {"train_loss": [], "eval_loss": [], "steps": []}
global_step = 0
best_eval_loss = float("inf")

print(f"  Epochs: {EPOCHS} | LR: {LR} | Grad Accum: {GRAD_ACCUM}")
print(f"  Effective batch: {2 * GRAD_ACCUM} | Steps: {total_steps}")
print(f"  Start: {datetime.now().strftime('%H:%M:%S')}")
print("-" * 60)

for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()

    for step, batch in enumerate(train_loader):
        batch = {k: v.cuda() for k, v in batch.items()}

        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss / GRAD_ACCUM

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACCUM == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            epoch_loss += loss.item() * GRAD_ACCUM

            if global_step % 2 == 0:
                history["train_loss"].append(float(loss.item() * GRAD_ACCUM))
                history["steps"].append(global_step)
                print(f"  Epoch {epoch+1}/{EPOCHS} | Step {global_step}/{total_steps} | Loss: {loss.item()*GRAD_ACCUM:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

    # Eval
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for batch in eval_loader:
            batch = {k: v.cuda() for k, v in batch.items()}
            with torch.cuda.amp.autocast():
                outputs = model(**batch)
            eval_loss += outputs.loss.item()
    eval_loss /= len(eval_loader)

    history["eval_loss"].append(float(eval_loss))
    print(f"  >>> Epoch {epoch+1} complete | Eval Loss: {eval_loss:.4f}")

    # Save checkpoint
    ckpt_path = os.path.join(OUTPUT_DIR, "checkpoints", f"epoch_{epoch+1}")
    model.save_pretrained(ckpt_path)
    tok.save_pretrained(ckpt_path)
    print(f"  >>> Checkpoint saved: {ckpt_path}")

elapsed = (time.time() - t0) / 60
print(f"\nTraining complete in {elapsed:.1f} min!")
print(f"Best eval loss: {best_eval_loss:.4f}")

# =====================================================
# SAVE
# =====================================================
print("\nSaving final model...")
final_path = os.path.join(OUTPUT_DIR, "final_lora")
model.save_pretrained(final_path)
tok.save_pretrained(final_path)

metrics = {
    "model": "Qwen2.5-1.5B-Instruct",
    "method": "QLoRA (4-bit NF4 + LoRA r=8 alpha=16)",
    "train_samples": len(train_ds),
    "eval_samples": len(eval_ds),
    "epochs": EPOCHS, "lr": LR,
    "total_minutes": elapsed,
    "final_train_loss": history["train_loss"][-1] if history["train_loss"] else None,
    "final_eval_loss": history["eval_loss"][-1] if history["eval_loss"] else None,
    "best_eval_loss": best_eval_loss,
    "history": history,
}
with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w", encoding="utf-8") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)

print(f"Model: {final_path}")
print(f"Metrics: {os.path.join(OUTPUT_DIR, 'training_history.json')}")
print(f"\n{'='*60}")
print(f"SUCCESS! Total time: {elapsed:.1f} min")
print(f"{'='*60}")
