import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = "PROJECT_ROOT"

# ---- Figure 1: PPL comparison bar chart ----
results = {
    "0.5B zero-shot": 17.44,
    "0.5B LoRA": 7.50,
    "0.5B full finetune": 5.29,
    "1.5B zero-shot": 13.66,
}
fig, ax = plt.subplots(figsize=(10, 6))
names = list(results.keys())
ppls = [results[k] for k in names]
colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
bars = ax.bar(names, ppls, color=colors, edgecolor="black", linewidth=0.8)
for b, v in zip(bars, ppls):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3, f"{v:.2f}", ha="center", fontsize=11)
ax.set_ylabel("Held-out Perplexity (lower is better)")
ax.set_title("Qwen2.5 Instruction Tuning: Perplexity Comparison")
ax.grid(axis="y", alpha=0.3)
plt.xticks(rotation=15)
plt.tight_layout()
out1 = os.path.join(BASE, "output_lora", "ppl_comparison.png")
plt.savefig(out1, dpi=150)
plt.close()
print("Wrote", out1)

# ---- Figure 2: Loss curves ----
lora_hist = json.load(open(os.path.join(BASE, "output_05_lora/history.json"), encoding="utf-8"))
full_hist = json.load(open(os.path.join(BASE, "output_05_full/history.json"), encoding="utf-8"))
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax = axes[0]
ax.plot(range(1, len(lora_hist["train_loss"])+1), lora_hist["train_loss"], "o-", label="train loss")
ax.plot(range(1, len(lora_hist["eval_loss"])+1), lora_hist["eval_loss"], "s--", label="eval loss")
ax.set_title("Qwen2.5-0.5B LoRA: Loss Curve")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-entropy loss")
ax.legend()
ax.grid(alpha=0.3)
ax = axes[1]
ax.plot(range(1, len(full_hist["train_loss"])+1), full_hist["train_loss"], "o-", label="train loss")
ax.plot(range(1, len(full_hist["eval_loss"])+1), full_hist["eval_loss"], "s--", label="eval loss")
ax.set_title("Qwen2.5-0.5B Full Finetune: Loss Curve")
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-entropy loss")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
out2 = os.path.join(BASE, "output_lora", "loss_curve.png")
plt.savefig(out2, dpi=150)
plt.close()
print("Wrote", out2)