"""
Generate visualization plots from training history
"""
import os, json
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import numpy as np
import math

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output", "qwen2.5-lora")
IMG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output")
os.makedirs(IMG_DIR, exist_ok=True)

# Load training history
with open(os.path.join(OUT_DIR, "training_history.json"), "r", encoding="utf-8") as f:
    history = json.load(f)

print("Loaded training history:")
print(f"  Train loss: {len(history['history']['train_loss'])} points")
print(f"  Eval loss: {len(history['history']['eval_loss'])} points")
print(f"  Final train loss: {history['final_train_loss']:.4f}")
print(f"  Final eval loss: {history['final_eval_loss']:.4f}")

train_losses = history["history"]["train_loss"]
eval_losses = history["history"]["eval_loss"]
train_steps = history["history"]["steps"]

# =====================================================
# Figure 1: Training & Evaluation Loss Curves
# =====================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Actual losses
ax = axes[0]
ax.plot(train_steps, train_losses, "b-o", linewidth=2, markersize=8, label="Training Loss")
epoch_boundaries = [1]  # step 2, 4, etc.
eval_steps_plot = [train_steps[i*2-1] for i in range(1, len(eval_losses)+1)] if len(train_steps) >= len(eval_losses)*2 else train_steps
# Actually, let's use proper eval step positions
eval_x = [train_steps[min(i*len(train_steps)//len(eval_losses), len(train_steps)-1)] for i in range(len(eval_losses))]
ax.plot(eval_x, eval_losses, "r-s", linewidth=2, markersize=8, label="Evaluation Loss")

# Annotate values
for i, (x, y) in enumerate(zip(train_steps, train_losses)):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=9, color="blue")
for i, (x, y) in enumerate(zip(eval_x, eval_losses)):
    ax.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=9, color="red")

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("QLoRA Fine-Tuning Loss Curves", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

# Right: Loss reduction bar chart
ax2 = axes[1]
epochs = [f"Epoch {i+1}" for i in range(len(eval_losses))]
x = np.arange(len(epochs))
width = 0.35

# Get train losses at epoch ends
train_epoch_losses = []
for e in range(len(eval_losses)):
    end_step = (e + 1) * (len(train_steps) // len(eval_losses))
    idx = min(end_step - 1, len(train_losses) - 1)
    train_epoch_losses.append(train_losses[idx])

bars1 = ax2.bar(x - width/2, train_epoch_losses, width, label="Train Loss", color="#5B9BD5")
bars2 = ax2.bar(x + width/2, eval_losses, width, label="Eval Loss", color="#ED7D31")

for bar, val in zip(bars1, train_epoch_losses):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")
for bar, val in zip(bars2, eval_losses):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, f"{val:.2f}", ha="center", fontsize=10, fontweight="bold")

ax2.set_xticks(x)
ax2.set_xticklabels(epochs)
ax2.set_ylabel("Loss", fontsize=12)
ax2.set_title("Train vs Eval Loss by Epoch", fontsize=14, fontweight="bold")
ax2.legend(fontsize=10)
ax2.grid(axis="y", alpha=0.3)

# Improvement percentage
improvement = (train_epoch_losses[0] - train_epoch_losses[-1]) / train_epoch_losses[0] * 100
eval_improvement = (eval_losses[0] - eval_losses[-1]) / eval_losses[0] * 100
fig.suptitle(f"Training Loss improved {improvement:.1f}% | Eval Loss improved {eval_improvement:.1f}%",
             fontsize=11, fontstyle="italic", y=1.02)

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "training_curves.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: training_curves.png")

# =====================================================
# Figure 2: Perplexity Analysis
# =====================================================
fig, ax = plt.subplots(figsize=(10, 6))

train_ppl = [math.exp(l) for l in train_epoch_losses]
eval_ppl = [math.exp(l) for l in eval_losses]

ax.plot(epochs, train_ppl, "b-o", linewidth=2, markersize=10, label="Train Perplexity")
ax.plot(epochs, eval_ppl, "r-s", linewidth=2, markersize=10, label="Eval Perplexity")

for i, (t, e) in enumerate(zip(train_ppl, eval_ppl)):
    ax.annotate(f"{t:.1f}", (epochs[i], t), textcoords="offset points", xytext=(0, 12), ha="center", fontsize=10, color="blue")
    ax.annotate(f"{e:.1f}", (epochs[i], e), textcoords="offset points", xytext=(0, -18), ha="center", fontsize=10, color="red")

ax.set_ylabel("Perplexity (lower is better)", fontsize=12)
ax.set_title("Model Perplexity During QLoRA Fine-Tuning", fontsize=14, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

import math
ppl_improv = (train_ppl[0] - train_ppl[-1]) / train_ppl[0] * 100
ax.text(0.5, 0.95, f"Perplexity reduced by {ppl_improv:.1f}% over training",
        transform=ax.transAxes, fontsize=11, fontstyle="italic", ha="center",
        bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "perplexity_analysis.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: perplexity_analysis.png")

# =====================================================
# Figure 3: Training Configuration Summary
# =====================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.axis("off")

config_text = (
    "QLoRA Fine-Tuning Configuration\n"
    "=" * 40 + "\n"
    f"Model: Qwen2.5-1.5B-Instruct (1.54B params)\n"
    f"Method: QLoRA (4-bit NF4 quantization + LoRA)\n"
    f"LoRA Config: r=8, alpha=16, dropout=0.05\n"
    f"Target Modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj\n"
    f"Trainable Params: 9,232,384 (0.59% of total)\n"
    f"Learning Rate: 2e-4 with linear warmup (5 steps)\n"
    f"Optimizer: AdamW with weight decay 0.01\n"
    f"Batch Size: 2 × 4 gradient accumulation = effective 8\n"
    f"Training Epochs: 3\n"
    f"Max Sequence Length: 256 tokens\n"
    f"GPU: NVIDIA RTX 4060 Laptop 8GB VRAM\n"
    f"VRAM Used: 1.2 GB\n"
    f"Training Time: {history['total_minutes']:.1f} minutes\n"
    f"\nResults:\n"
    f"  Initial Train Loss: {train_epoch_losses[0]:.2f} → Final: {train_epoch_losses[-1]:.2f} ({improvement:.1f}% improvement)\n"
    f"  Initial Eval Loss: {eval_losses[0]:.2f} → Final: {eval_losses[-1]:.2f} ({eval_improvement:.1f}% improvement)\n"
)

ax.text(0.5, 0.5, config_text, transform=ax.transAxes,
        fontsize=11, verticalalignment="center", ha="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=1", facecolor="lightblue", alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "training_config.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved: training_config.png")

print("\nAll visualizations generated!")
