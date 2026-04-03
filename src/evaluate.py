# -*- coding: utf-8 -*-
"""
PHASE 5: Model Evaluation
Loads trained models, evaluates on held-out test set,
generates confusion matrices, computes metrics, and runs error analysis.
Run with: python -m src.evaluate
"""

import sys, io, os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_score, recall_score, f1_score, accuracy_score,
)

from src.preprocessing import preprocess_to_string
from src.vectorizer    import transform_texts, load_vectorizer

BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DATA_PATH  = os.path.join(BASE_DIR, "data", "spam.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEP  = "=" * 65
SEP2 = "-" * 65


# ══════════════════════════════════════════════════════════════════
# 1. LOAD EVERYTHING
# ══════════════════════════════════════════════════════════════════
def load_models_and_data():
    """Load trained models, vectorizer, and the held-out test split."""
    print(f"\n{SEP}\n  STEP 1: LOADING MODELS AND TEST DATA\n{SEP2}")

    # Load models
    models = {
        "Naive Bayes"         : joblib.load(os.path.join(MODELS_DIR, "naive_bayes.pkl")),
        "Logistic Regression" : joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl")),
        "SVM"                 : joblib.load(os.path.join(MODELS_DIR, "svm.pkl")),
    }
    vectorizer   = load_vectorizer()
    split        = joblib.load(os.path.join(MODELS_DIR, "split_indices.pkl"))
    test_idx     = split["test_idx"]

    print(f"  Models loaded      : {list(models.keys())}")
    print(f"  Test indices       : {len(test_idx):,} messages")

    # Reload original data so we can show raw messages in error analysis
    df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "message"]
    df["label_int"] = (df["label"] == "spam").astype(int)
    df["clean"]     = df["message"].apply(preprocess_to_string)

    df_test   = df.iloc[test_idx].reset_index(drop=True)
    X_test    = transform_texts(df_test["clean"].tolist(), vectorizer)
    y_test    = df_test["label_int"].values

    print(f"  Test set size      : {len(y_test):,}")
    print(f"  Spam in test       : {y_test.sum():,} ({100*y_test.mean():.1f}%)")
    print(f"  Ham  in test       : {(y_test==0).sum():,} ({100*(1-y_test.mean()):.1f}%)")

    return models, X_test, y_test, df_test


# ══════════════════════════════════════════════════════════════════
# 2. CONFUSION MATRIX
# ══════════════════════════════════════════════════════════════════
def generate_confusion_matrix(y_true, y_pred, model_name="Model"):
    """Return sklearn confusion matrix and print labelled breakdown."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n  Confusion Matrix -- {model_name}")
    print(f"  {'':30s}  Predicted HAM  Predicted SPAM")
    print(f"  {'Actual HAM  (0)':30s}  {tn:>13,}  {fp:>13,}")
    print(f"  {'Actual SPAM (1)':30s}  {fn:>13,}  {tp:>13,}")
    print(f"""
  TN={tn:>4}  Ham correctly passed through       (good!)
  TP={tp:>4}  Spam correctly caught              (good!)
  FP={fp:>4}  Ham wrongly flagged as spam        (annoying - legit mail blocked)
  FN={fn:>4}  Spam that slipped through as ham   (dangerous - missed spam!)
""")
    return cm, tn, fp, fn, tp


# ══════════════════════════════════════════════════════════════════
# 3. METRICS
# ══════════════════════════════════════════════════════════════════
def calculate_metrics(y_true, y_pred):
    """Calculate and return all key metrics as a dict."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "accuracy"  : accuracy_score(y_true, y_pred),
        "precision" : precision_score(y_true, y_pred, zero_division=0),
        "recall"    : recall_score(y_true, y_pred, zero_division=0),
        "f1"        : f1_score(y_true, y_pred, zero_division=0),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    }


# ══════════════════════════════════════════════════════════════════
# 4. VISUALISE — 4-panel figure (3 confusion matrices + bar chart)
# ══════════════════════════════════════════════════════════════════
def visualize_all(all_cms, all_metrics, model_names):
    """
    Panel 1-3: Confusion matrix heatmaps for each model.
    Panel 4  : Multi-metric bar chart comparison.
    """
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0d1117")
    gs  = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)

    model_colors = ["#ff4757", "#ffa502", "#2ed573"]
    labels       = ["Ham (0)", "Spam (1)"]
    metric_keys  = ["accuracy", "precision", "recall", "f1"]

    # ── Panels 0-2: Confusion matrix heatmaps ─────────────────────
    for i, (name, cm) in enumerate(zip(model_names, all_cms)):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#161b22")
        im = ax.imshow(cm, cmap="YlOrRd", aspect="auto")

        # Annotate cells
        for r in range(2):
            for c in range(2):
                val   = cm[r, c]
                color = "black" if cm[r, c] > cm.max() / 2 else "white"
                ax.text(c, r, f"{val:,}", ha="center", va="center",
                        fontsize=18, fontweight="bold", color=color)

        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Pred Ham", "Pred Spam"], color="white", fontsize=10)
        ax.set_yticklabels(["Act Ham", "Act Spam"],   color="white", fontsize=10)
        ax.set_title(name, color=model_colors[i], fontsize=13, pad=12, fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#30363d")

    # ── Panel 3: Multi-metric bar comparison ──────────────────────
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor("#161b22")
    x   = np.arange(len(metric_keys))
    w   = 0.22
    for i, (name, color) in enumerate(zip(model_names, model_colors)):
        vals = [all_metrics[name][k] for k in metric_keys]
        bars = ax4.bar(x + i*w - w, vals, w, label=name,
                       color=color, alpha=0.85, edgecolor="none")
        for bar, v in zip(bars, vals):
            ax4.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 0.003,
                     f"{v:.3f}", ha="center", va="bottom",
                     fontsize=7, color="white")

    ax4.set_xticks(x)
    ax4.set_xticklabels([m.upper() for m in metric_keys],
                        color="white", fontsize=10)
    ax4.set_ylim(0.9, 1.01)
    ax4.set_ylabel("Score", color="#aaaaaa")
    ax4.set_title("Metric Comparison", color="white", fontsize=13, pad=12)
    ax4.legend(facecolor="#1c2128", edgecolor="#30363d", labelcolor="white", fontsize=9)
    ax4.tick_params(colors="white")
    for spine in ax4.spines.values():
        spine.set_edgecolor("#30363d")

    # ── Panel 4 (bottom wide): F1 / Recall / Precision summary bars ──
    ax5 = fig.add_subplot(gs[1, :])
    ax5.set_facecolor("#161b22")
    metric_display = ["precision", "recall", "f1", "accuracy"]
    n_m = len(metric_display)
    bar_w = 0.2
    x2 = np.arange(len(model_names))
    for j, mk in enumerate(metric_display):
        vals = [all_metrics[n][mk] for n in model_names]
        offset = (j - n_m/2 + 0.5) * bar_w
        ax5.bar(x2 + offset, vals, bar_w,
                label=mk.capitalize(), alpha=0.85, edgecolor="none")

    ax5.set_xticks(x2)
    ax5.set_xticklabels(model_names, color="white", fontsize=12)
    ax5.set_ylim(0.88, 1.01)
    ax5.set_ylabel("Score", color="#aaaaaa")
    ax5.set_title("Full Metric Breakdown per Model", color="white", fontsize=13, pad=12)
    ax5.legend(facecolor="#1c2128", edgecolor="#30363d", labelcolor="white",
               loc="lower right", fontsize=10)
    ax5.tick_params(colors="white")
    for spine in ax5.spines.values():
        spine.set_edgecolor("#30363d")

    plt.suptitle("SPAM DETECTION -- Phase 5: Model Evaluation",
                 color="white", fontsize=16, fontweight="bold", y=1.01)
    plt.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, "phase5_evaluation.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"\n  [OK] Visualization saved: {out_path}")


# ══════════════════════════════════════════════════════════════════
# 5. ERROR ANALYSIS
# ══════════════════════════════════════════════════════════════════
def error_analysis(model, model_name, X_test, y_test, df_test, n=5):
    """Print False Negatives and False Positives with original messages."""
    y_pred = model.predict(X_test)

    fn_mask = (y_test == 1) & (y_pred == 0)   # spam → predicted ham
    fp_mask = (y_test == 0) & (y_pred == 1)   # ham  → predicted spam

    fn_df = df_test[fn_mask].head(n)
    fp_df = df_test[fp_mask].head(n)

    print(f"\n{'~'*65}")
    print(f"  ERROR ANALYSIS: {model_name}")
    print(f"{'~'*65}")
    print(f"  Total FN (missed spam) : {fn_mask.sum():>4}  -- spam that slipped through!")
    print(f"  Total FP (false alarm) : {fp_mask.sum():>4}  -- ham wrongly blocked")

    print(f"\n  --- FALSE NEGATIVES (SPAM missed as HAM) ---")
    for i, (_, row) in enumerate(fn_df.iterrows(), 1):
        print(f"\n  [{i}] {row['message'][:120]}")

    print(f"\n  --- FALSE POSITIVES (HAM flagged as SPAM) ---")
    for i, (_, row) in enumerate(fp_df.iterrows(), 1):
        print(f"\n  [{i}] {row['message'][:120]}")


# ══════════════════════════════════════════════════════════════════
# 6. COMPARE ALL MODELS
# ══════════════════════════════════════════════════════════════════
def compare_models(models, X_test, y_test):
    """Pretty-print comparison table ranked by F1."""
    print(f"\n{SEP}\n  MODEL COMPARISON TABLE\n{SEP2}")
    print(f"  {'Model':22s}  {'Accuracy':>9s}  {'Precision':>9s}  {'Recall':>9s}  {'F1':>9s}  {'FN':>5s}  {'FP':>5s}")
    print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*5}  {'-'*5}")

    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        m      = calculate_metrics(y_test, y_pred)
        results[name] = m
        print(f"  {name:22s}  {m['accuracy']:>9.4f}  {m['precision']:>9.4f}  {m['recall']:>9.4f}  {m['f1']:>9.4f}  {m['FN']:>5}  {m['FP']:>5}")

    best = max(results, key=lambda n: results[n]["f1"])
    print(f"\n  Best model by F1: [{best}]  (F1 = {results[best]['f1']:.4f})")
    return results


# ══════════════════════════════════════════════════════════════════
# 7. MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print(SEP)
    print("  PHASE 5: MODEL EVALUATION -- CONFUSION MATRIX DEEP DIVE")
    print(SEP)

    # ── Load ──────────────────────────────────────────────────────
    models, X_test, y_test, df_test = load_models_and_data()

    # ── Per-model evaluation ──────────────────────────────────────
    print(f"\n{SEP}\n  STEP 2: CONFUSION MATRICES + METRICS\n{SEP2}")

    all_cms     = []
    all_metrics = {}
    model_names = list(models.keys())

    for name, model in models.items():
        print(f"\n  {'='*30}  {name}  {'='*30}")
        y_pred = model.predict(X_test)

        cm, tn, fp, fn, tp = generate_confusion_matrix(y_test, y_pred, name)
        m = calculate_metrics(y_test, y_pred)
        all_cms.append(cm)
        all_metrics[name] = m

        print(f"  Accuracy  : {m['accuracy']:.4f}")
        print(f"  Precision : {m['precision']:.4f}  (of predicted spam, this % IS spam)")
        print(f"  Recall    : {m['recall']:.4f}  (of actual spam, this % was CAUGHT)")
        print(f"  F1 Score  : {m['f1']:.4f}  (harmonic mean of precision & recall)")
        print(f"\n  Full classification report:")
        print(classification_report(y_test, y_pred,
                                    target_names=["Ham(0)", "Spam(1)"],
                                    digits=4))

    # ── Comparison table ──────────────────────────────────────────
    results = compare_models(models, X_test, y_test)

    # ── Visualise ─────────────────────────────────────────────────
    visualize_all(all_cms, all_metrics, model_names)

    # ── Error Analysis ────────────────────────────────────────────
    print(f"\n{SEP}\n  STEP 3: ERROR ANALYSIS\n{SEP2}")
    for name, model in models.items():
        error_analysis(model, name, X_test, y_test, df_test, n=4)

    # ── Final verdict ─────────────────────────────────────────────
    best    = max(results, key=lambda n: results[n]["f1"])
    best_fn = results[best]["FN"]
    best_fp = results[best]["FP"]

    print(f"\n{SEP}")
    print("  PHASE 5 SUMMARY & VERDICT")
    print(SEP2)
    print(f"""
  WINNER: [{best}]

  Accuracy  : {results[best]['accuracy']:.4f}
  Precision : {results[best]['precision']:.4f}
  Recall    : {results[best]['recall']:.4f}  <- most important for spam!
  F1 Score  : {results[best]['f1']:.4f}

  Of {y_test.sum()} actual spam messages in the test set:
    -> Correctly caught (TP) : {y_test.sum() - best_fn}
    -> Slipped through (FN)  : {best_fn}   <-- missed spam (dangerous!)
    -> Ham blocked wrongly   : {best_fp}   <-- false alarm (annoying)

  KEY TAKEAWAYS:
    Recall > Precision for spam:
      We PREFER to block a few legitimate emails (small FP)
      over letting spam through to the inbox (large FN).
      SVM hits the best recall -- catches the most actual spam.

  NEXT: Phase 6 -- FastAPI backend that serves predictions
        via a REST API endpoint /predict
""")
    print(SEP)
    print("  Phase 5 Complete!  Ready for Phase 6: FastAPI Backend")
    print(SEP)


if __name__ == "__main__":
    main()
