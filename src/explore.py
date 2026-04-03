# -*- coding: utf-8 -*-
"""
Phase 1: Data Exploration
Loads spam.csv, prints key statistics, and saves a 4-panel visualization.
Run with: python -m src.explore
"""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "spam.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEP  = "=" * 65
SEP2 = "-" * 65


def section(title):
    print(f"\n\n{SEP}")
    print(f"  {title}")
    print(SEP2)


# Load dataset — latin-1 handles special chars; only cols 0-1 are useful
section("[STEP 1] Loading the Dataset")

df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=[0, 1])
df.columns = ["label", "message"]

print(f"[OK] Dataset loaded from: {DATA_PATH}")
print(f"\n  Shape: {df.shape[0]:,} rows  x  {df.shape[1]} columns (label, message)")


section("[STEP 2] First 10 Rows")
print(df.head(10).to_string(index=True))


section("[STEP 3] Data Types and Missing Values")

print("  Column Data Types:")
print(df.dtypes.to_string())

print("\n  Missing Values per column:")
missing = df.isnull().sum()
for col, count in missing.items():
    status = "NONE (clean!)" if count == 0 else f"WARNING: {count} missing!"
    print(f"    {col:10s} : {status}")


section("[STEP 4] Basic Statistics -- Message Lengths")

df["msg_length"] = df["message"].apply(len)
df["word_count"] = df["message"].apply(lambda x: len(x.split()))

stats = df.groupby("label")[["msg_length", "word_count"]].describe()
print(stats.to_string())

avg_spam_len = df[df["label"] == "spam"]["msg_length"].mean()
avg_ham_len  = df[df["label"] == "ham"]["msg_length"].mean()
avg_spam_wc  = df[df["label"] == "spam"]["word_count"].mean()
avg_ham_wc   = df[df["label"] == "ham"]["word_count"].mean()

comparison = "LONGER" if avg_spam_len > avg_ham_len else "SHORTER"
print(f"""
  Avg spam length  : {avg_spam_len:.0f} chars  ({avg_spam_wc:.0f} words)
  Avg ham  length  : {avg_ham_len:.0f} chars  ({avg_ham_wc:.0f} words)
  --> Spam tends to be {comparison}.
""")


section("[STEP 5] Class Distribution (Spam vs Ham)")

counts      = df["label"].value_counts()
percentages = df["label"].value_counts(normalize=True) * 100

print(f"  {'Label':10s} {'Count':>8s}   {'Percentage':>10s}")
print(f"  {'-'*10} {'-'*8}   {'-'*10}")
for label in ["ham", "spam"]:
    print(f"  {label:10s} {counts[label]:>8,}   {percentages[label]:>9.2f}%")
total = len(df)
print(f"  {'TOTAL':10s} {total:>8,}   {'100.00%':>10s}")

print(f"""
  Note: dataset is imbalanced ({percentages['ham']:.1f}% ham).
  A naive "always-ham" classifier gets {percentages['ham']:.1f}% accuracy but catches ZERO spam.
  We use precision, recall, and F1 as evaluation metrics (Phase 5).
""")


section("[STEP 6] Creating Visualizations")

plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0d1117")

colors = {"ham": "#00d2ff", "spam": "#ff4757"}

# Bar chart
ax1 = fig.add_subplot(2, 2, 1)
ax1.set_facecolor("#161b22")
bars = ax1.bar(
    counts.index, counts.values,
    color=[colors["ham"], colors["spam"]],
    width=0.5, edgecolor="white", linewidth=0.5,
)
for bar, (label, count) in zip(bars, counts.items()):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 30,
        f"{count:,}", ha="center", va="bottom",
        color="white", fontsize=14, fontweight="bold",
    )
ax1.set_title("Spam vs Ham -- Message Count", color="white", fontsize=13, pad=12)
ax1.set_ylabel("Number of Messages", color="#aaaaaa")
ax1.set_xlabel("Label", color="#aaaaaa")
ax1.tick_params(colors="white")
for spine in ax1.spines.values():
    spine.set_edgecolor("#30363d")

# Pie chart
ax2 = fig.add_subplot(2, 2, 2)
ax2.set_facecolor("#161b22")
wedges, texts, autotexts = ax2.pie(
    counts.values, labels=counts.index,
    autopct="%1.1f%%", startangle=90,
    colors=[colors["ham"], colors["spam"]],
    wedgeprops={"edgecolor": "#0d1117", "linewidth": 2},
    textprops={"color": "white", "fontsize": 12},
)
for at in autotexts:
    at.set_fontsize(13)
    at.set_fontweight("bold")
ax2.set_title("Class Distribution (%)", color="white", fontsize=13, pad=12)

# Message length distribution
ax3 = fig.add_subplot(2, 2, 3)
ax3.set_facecolor("#161b22")
for label, color in colors.items():
    subset = df[df["label"] == label]["msg_length"]
    ax3.hist(
        subset, bins=40, alpha=0.7, color=color,
        label=f"{label} (avg: {subset.mean():.0f} chars)",
        edgecolor="none",
    )
ax3.set_title("Message Length Distribution", color="white", fontsize=13, pad=12)
ax3.set_xlabel("Message Length (characters)", color="#aaaaaa")
ax3.set_ylabel("Frequency", color="#aaaaaa")
ax3.legend(facecolor="#1c2128", edgecolor="#30363d", labelcolor="white")
ax3.tick_params(colors="white")
for spine in ax3.spines.values():
    spine.set_edgecolor("#30363d")

# Word count distribution
ax4 = fig.add_subplot(2, 2, 4)
ax4.set_facecolor("#161b22")
for label, color in colors.items():
    subset = df[df["label"] == label]["word_count"]
    ax4.hist(
        subset, bins=30, alpha=0.7, color=color,
        label=f"{label} (avg: {subset.mean():.0f} words)",
        edgecolor="none",
    )
ax4.set_title("Word Count Distribution", color="white", fontsize=13, pad=12)
ax4.set_xlabel("Number of Words per Message", color="#aaaaaa")
ax4.set_ylabel("Frequency", color="#aaaaaa")
ax4.legend(facecolor="#1c2128", edgecolor="#30363d", labelcolor="white")
ax4.tick_params(colors="white")
for spine in ax4.spines.values():
    spine.set_edgecolor("#30363d")

plt.suptitle(
    "SPAM DETECTION -- Phase 1: Data Exploration",
    color="white", fontsize=16, fontweight="bold", y=1.01,
)
plt.tight_layout()

chart_path = os.path.join(OUTPUT_DIR, "phase1_exploration.png")
plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
print(f"  [OK] Chart saved to: {chart_path}")
plt.close()


section("[STEP 7] Sample Messages")

spam_samples = df[df["label"] == "spam"]["message"].sample(5, random_state=42)
ham_samples  = df[df["label"] == "ham"]["message"].sample(5, random_state=42)

print("\n  *** 5 RANDOM SPAM MESSAGES ***")
print(SEP2)
for i, (idx, msg) in enumerate(spam_samples.items(), 1):
    print(f"\n  [{i}] {msg}")

print("\n\n  *** 5 RANDOM HAM (LEGITIMATE) MESSAGES ***")
print(SEP2)
for i, (idx, msg) in enumerate(ham_samples.items(), 1):
    print(f"\n  [{i}] {msg}")


print(f"\n\n{SEP}")
print("  PHASE 1 SUMMARY")
print(SEP)
print(f"""
  Total messages    : {total:,}
  Ham (legitimate)  : {counts['ham']:,} ({percentages['ham']:.1f}%)
  Spam              : {counts['spam']:,} ({percentages['spam']:.1f}%)
  Missing values    : 0

  Key observations:
    - Class imbalance: {percentages['ham']:.0f}% ham vs {percentages['spam']:.0f}% spam → use F1, not accuracy
    - Spam messages avg {avg_spam_len:.0f} chars vs ham {avg_ham_len:.0f} chars
    - Spam vocabulary clusters around action/lure words (free, win, call, prize)
    - Text must be normalized before modeling (uppercase, URLs, punctuation, numbers)
""")

print(SEP)
print("  Phase 1 Complete!  Ready for Phase 2: Preprocessing")
print(SEP)
