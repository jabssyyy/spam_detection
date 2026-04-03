# -*- coding: utf-8 -*-
"""
===================================================================
PHASE 1: Data Exploration -- Spam Detection Project
===================================================================

PURPOSE OF THIS FILE:
    Before building ANY model, we MUST understand our data.
    This is like a doctor examining a patient before prescribing medicine.

    We will answer 5 key questions:
        1. What does our data look like?
        2. What are the column names and types?
        3. Are there missing values?
        4. How many spam vs ham messages are there?
        5. What do real spam/ham messages look like?

WHY DATA EXPLORATION FIRST?
    - You can't clean what you don't understand
    - You can't evaluate a model without knowing your class distribution
    - Patterns in data guide our preprocessing decisions
===================================================================
"""

import sys
import io

# Force UTF-8 output on Windows so special chars print correctly
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")   # non-interactive backend -- saves file without a popup
import matplotlib.pyplot as plt
import os
import warnings

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------
# STEP 0: Set up paths
# ---------------------------------------------------------------
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


# ===================================================================
# STEP 1: LOAD THE DATA
# ===================================================================
# WHY encoding='latin-1'?
#   The file has some special characters (like the British pound sign
#   in spam messages).  latin-1 handles them; UTF-8 would crash.
#
# WHY usecols=[0,1]?
#   The CSV has 5 columns.  Only the first two are useful:
#     Col 0 (v1) = label    --> "spam" or "ham"
#     Col 1 (v2) = message  --> the actual text
#   Columns 3-5 are mostly empty NaN values -- skip them.
# ===================================================================

section("[STEP 1] Loading the Dataset")

df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=[0, 1])
df.columns = ["label", "message"]  # rename from v1/v2 to meaningful names

print(f"[OK] Dataset loaded from: {DATA_PATH}")
print(f"\n  Shape of dataset:")
print(f"    Rows    : {df.shape[0]:,}  (each row = one SMS message)")
print(f"    Columns : {df.shape[1]}    (label + message)")


# ===================================================================
# STEP 2: FIRST LOOK -- First 10 rows
# ===================================================================
# WHY look at the first rows?
#   Like opening a book to page 1 -- you get a feel for the format.
# ===================================================================

section("[STEP 2] First 10 Rows")
print(df.head(10).to_string(index=True))
print("""
  COLUMN EXPLANATION:
    label   --> the 'answer' column: 'ham' (legitimate) or 'spam'
    message --> the actual SMS text

  This is our raw data.  The model will read 'message' and
  learn to predict 'label'.
""")


# ===================================================================
# STEP 3: DATA TYPES & MISSING VALUES
# ===================================================================
section("[STEP 3] Data Types and Missing Values")

print("  Column Data Types:")
print(df.dtypes.to_string())

print("""
  'object' = text/string data (expected -- both columns are text)
  In Phase 3 (Vectorization) we convert text --> numbers.
""")

print("  Missing Values per column:")
missing = df.isnull().sum()
for col, count in missing.items():
    status = "NONE (clean!)" if count == 0 else f"WARNING: {count} missing!"
    print(f"    {col:10s} : {status}")

print("""
  WHY missing values matter:
    If 'label' is missing  --> we can't train on that row
    If 'message' is missing --> nothing to predict from
  This dataset is very clean -- zero missing values!
""")


# ===================================================================
# STEP 4: BASIC STATISTICS -- Message Lengths
# ===================================================================
# WHY compute lengths?
#   Spam messages often pack a lot of words ("FREE PRIZE CALL NOW!")
#   Understanding length patterns helps confirm our intuition.
# ===================================================================

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
  KEY OBSERVATIONS:
    Avg spam message length : {avg_spam_len:.0f} characters
    Avg ham  message length : {avg_ham_len:.0f} characters

    Avg spam word count     : {avg_spam_wc:.0f} words
    Avg ham  word count     : {avg_ham_wc:.0f} words

  --> Spam messages tend to be {comparison}.
      This is a learnable pattern!
""")


# ===================================================================
# STEP 5: CLASS DISTRIBUTION -- The Most Important Step!
# ===================================================================
# WHY? If your data has 87% ham, a model that always says "ham"
# gets 87% accuracy but catches ZERO spam.
# This is the core reason accuracy alone is a bad metric.
# ===================================================================

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
  *** WHY THIS IMBALANCE MATTERS -- CRITICAL LESSON ***

  Imagine a lazy model that ignores the text and always says "ham".

  How accurate would it be?
    --> {percentages['ham']:.1f}% accurate!
        (Because {percentages['ham']:.1f}% of messages ARE legitimately ham.)

  But it catches ZERO spam.
    --> Recall for spam = 0%
    --> That's completely useless!

  This is WHY we cannot rely on accuracy alone.
  We need:
    Precision --> of all predicted spam, how many are actually spam?
    Recall    --> of all actual spam, how many did we catch?
    F1 Score  --> harmonic mean of precision and recall

  We compute all of these in Phase 5 (Evaluation)!
""")


# ===================================================================
# STEP 6: VISUALIZATIONS -- 4-panel chart
# ===================================================================

section("[STEP 6] Creating Visualizations")

plt.style.use("dark_background")
fig = plt.figure(figsize=(18, 14))
fig.patch.set_facecolor("#0d1117")

colors = {"ham": "#00d2ff", "spam": "#ff4757"}

# ---- Chart 1: Bar Chart ----
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

# ---- Chart 2: Pie Chart ----
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

# ---- Chart 3: Message Length Distribution ----
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

# ---- Chart 4: Word Count Distribution ----
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


# ===================================================================
# STEP 7: SAMPLE MESSAGES
# ===================================================================

section("[STEP 7] Sample Messages -- Spot the Patterns!")

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

print("""
  PATTERNS YOU LIKELY NOTICED IN SPAM:
    [*] UPPERCASE words  --> "FREE", "WIN", "URGENT", "CLICK"
    [*] Numbers          --> "1000", "500 prizes", "0800"
    [*] Special chars    --> "!", "http://", "£", "$"
    [*] Action words     --> "call now", "claim", "reply"
    [*] Lure words       --> "free", "winner", "prize", "offer"

  PATTERNS IN HAM:
    [*] Personal pronouns --> "I", "you", "we"
    [*] Casual language   --> "gonna", "ok", "lol"
    [*] Context-specific  --> friends, plans, daily life
    [*] Natural tone      --> no fake urgency or offers

  --> TF-IDF (Phase 3) will automatically assign higher weights
      to words like "free", "win", "prize" because they appear
      disproportionately in spam. The model learns from that!
""")


# ===================================================================
# STEP 8: FINAL SUMMARY
# ===================================================================

print(f"\n\n{SEP}")
print("  PHASE 1 SUMMARY -- WHAT WE LEARNED")
print(SEP)
print(f"""
  DATASET FACTS:
    Total messages    : {total:,}
    Ham (legitimate)  : {counts['ham']:,} ({percentages['ham']:.1f}%)
    Spam              : {counts['spam']:,} ({percentages['spam']:.1f}%)
    Missing values    : 0  (clean dataset!)

  KEY INSIGHTS:

    1. CLASS IMBALANCE
       The dataset heavily favors ham (~{percentages['ham']:.0f}% vs ~{percentages['spam']:.0f}% spam).
       A model predicting "ham" every time would be {percentages['ham']:.1f}% accurate
       but completely useless.  We MUST use precision, recall,
       and F1 as our real evaluation metrics (Phase 5).

    2. MESSAGE LENGTH DIFFERS
       Spam messages are on average {avg_spam_len:.0f} chars vs ham {avg_ham_len:.0f} chars.
       This length difference is a learnable signal.

    3. SPAM HAS DISTINCT VOCABULARY
       Words like "free", "win", "call", "prize" cluster in spam.
       TF-IDF (Phase 3) will weight these words higher.

    4. TEXT NEEDS CLEANING BEFORE MODELING
       ALL-CAPS, URLs, punctuation, phone numbers, and
       number sequences appear in spam.  We must normalize these.

  CHALLENGES WE WILL FACE:
    [!] Imbalanced classes     --> don't trust accuracy alone
    [!] Raw text               --> models need numbers, not strings
    [!] Noisy text             --> URLs, punctuation, casing
    [!] Large vocabulary       --> tens of thousands of unique words

  WHAT'S NEXT -- PHASE 2: TEXT PREPROCESSING:
    Step 1  Convert to lowercase     "FREE NOW"  --> "free now"
    Step 2  Remove punctuation       "free!!!"   --> "free"
    Step 3  Remove numbers           "call 0800" --> "call"
    Step 4  Remove stopwords         "I will go" --> "go"
            (stopwords = common words like "the","is","a"
             that carry no spam-detection signal)
    Step 5  Stemming                 "running"   --> "run"
            (reduces words to their root form)
""")

print(SEP)
print("  Phase 1 Complete!  Ready for Phase 2: Preprocessing")
print(SEP)
