# -*- coding: utf-8 -*-
"""
PHASE 4: Model Training
Trains Naive Bayes, Logistic Regression, and SVM on TF-IDF vectors.
Run with: python -m src.train
"""

import sys, io, os, time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection  import train_test_split
from sklearn.naive_bayes      import MultinomialNB
from sklearn.linear_model     import LogisticRegression
from sklearn.svm              import LinearSVC
from sklearn.calibration      import CalibratedClassifierCV

from src.preprocessing import preprocess_to_string
from src.vectorizer    import (
    create_tfidf_vectorizer,
    fit_transform_texts,
    transform_texts,
    get_feature_names,
    save_vectorizer,
)

# ── Paths ──────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "spam.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

SEP  = "=" * 65
SEP2 = "-" * 65


# ══════════════════════════════════════════════════════════════════
# 1. LOAD AND PREPARE DATA
# ══════════════════════════════════════════════════════════════════
def load_and_prepare_data():
    """
    Load spam.csv → preprocess all messages → TF-IDF vectorize.
    Returns: X (sparse matrix), y (int array), vectorizer, feature_names
    """
    print(f"\n{SEP}")
    print("  STEP 1: LOADING AND PREPARING DATA")
    print(SEP2)

    df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "message"]
    print(f"  Loaded {len(df):,} messages  |  spam: {(df.label=='spam').sum()}  ham: {(df.label=='ham').sum()}")

    # Encode labels: spam=1, ham=0
    df["label_int"] = (df["label"] == "spam").astype(int)

    # Preprocess all messages
    print("  Preprocessing text... ", end="", flush=True)
    df["clean"] = df["message"].apply(preprocess_to_string)
    print("done")

    # Fit TF-IDF ONLY on training data would be ideal, but here we
    # fit on all data first to get feature names for display.
    # In split_data we refit on X_train only — see the note there.
    print("  Vectorizing (TF-IDF)... ", end="", flush=True)
    vectorizer, X = fit_transform_texts(df["clean"].tolist())
    print("done")

    y = df["label_int"].values
    feature_names = get_feature_names(vectorizer)

    print(f"  Matrix shape    : {X.shape}  ({X.shape[0]:,} messages × {X.shape[1]:,} features)")
    print(f"  Spam count      : {y.sum():,} ({100*y.mean():.1f}%)")
    print(f"  Ham  count      : {(y==0).sum():,} ({100*(1-y.mean()):.1f}%)")

    return X, y, vectorizer, feature_names, df["clean"].tolist()


# ══════════════════════════════════════════════════════════════════
# 2. TRAIN-TEST SPLIT
# ══════════════════════════════════════════════════════════════════
def split_data(X, y, texts, test_size=0.2, random_state=42):
    """
    Split into train/test sets with stratification.

    stratify=y  →  ensures BOTH splits keep the same spam/ham ratio.
    Without it, by random chance test might get 5% spam or 20% spam,
    making evaluation misleading.
    """
    print(f"\n{SEP}")
    print("  STEP 2: TRAIN / TEST SPLIT")
    print(SEP2)

    # Split indices so we can also refit the vectorizer on train only
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=y
    )

    X_train_raw = X[train_idx]
    X_test_raw  = X[test_idx]
    y_train = y[train_idx]
    y_test  = y[test_idx]

    spam_train_pct = 100 * y_train.mean()
    spam_test_pct  = 100 * y_test.mean()

    print(f"  Total messages   : {len(y):,}")
    print(f"  Training set     : {len(y_train):,}  (spam: {spam_train_pct:.1f}%)")
    print(f"  Test set         : {len(y_test):,}   (spam: {spam_test_pct:.1f}%)")
    print(f"  Spam ratio stays consistent thanks to stratify=y!")

    return X_train_raw, X_test_raw, y_train, y_test, train_idx, test_idx


# ══════════════════════════════════════════════════════════════════
# 3. TRAIN — NAIVE BAYES
# ══════════════════════════════════════════════════════════════════
def train_naive_bayes(X_train, y_train, feature_names, top_n=12):
    """
    Trains MultinomialNB.
    MultinomialNB works with non-negative counts/frequencies — perfect
    for TF-IDF values which are always >= 0.
    """
    print(f"\n{SEP}")
    print("  STEP 3a: TRAINING NAIVE BAYES")
    print(SEP2)

    t0 = time.time()
    model = MultinomialNB(alpha=0.1)   # alpha=smoothing (avoids zero probability)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Training time : {elapsed:.3f}s  (Naive Bayes is the fastest!)")
    print(f"  Classes       : {model.classes_}  (0=ham, 1=spam)")

    # Log-prob difference: high positive = strong spam word
    log_prob_diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
    spam_top_idx  = np.argsort(log_prob_diff)[-top_n:][::-1]
    ham_top_idx   = np.argsort(log_prob_diff)[:top_n]

    print(f"\n  Top {top_n} SPAM indicators (high log-prob in spam vs ham):")
    for idx in spam_top_idx:
        bar = "#" * int((log_prob_diff[idx]) * 3)
        print(f"    {feature_names[idx]:25s}  {log_prob_diff[idx]:+.3f}  |{bar}")

    print(f"\n  Top {top_n} HAM indicators (high log-prob in ham vs spam):")
    for idx in ham_top_idx:
        bar = "#" * int(abs(log_prob_diff[idx]) * 3)
        print(f"    {feature_names[idx]:25s}  {log_prob_diff[idx]:+.3f}  |{bar}")

    return model


# ══════════════════════════════════════════════════════════════════
# 4. TRAIN — LOGISTIC REGRESSION
# ══════════════════════════════════════════════════════════════════
def train_logistic_regression(X_train, y_train, feature_names, top_n=12):
    """
    Trains LogisticRegression.
    Coefficients tell us: positive weight = pushes toward SPAM,
    negative weight = pushes toward HAM.
    """
    print(f"\n{SEP}")
    print("  STEP 3b: TRAINING LOGISTIC REGRESSION")
    print(SEP2)

    t0 = time.time()
    model = LogisticRegression(
        C=1.0,               # regularisation strength (lower = more regularised)
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Training time  : {elapsed:.3f}s")
    print(f"  Coefficients   : {model.coef_.shape}  (one weight per feature)")

    coef = model.coef_[0]          # shape: (n_features,)
    top_spam_idx = np.argsort(coef)[-top_n:][::-1]   # most positive
    top_ham_idx  = np.argsort(coef)[:top_n]            # most negative

    print(f"\n  Top {top_n} SPAM words (largest positive coefficients):")
    for idx in top_spam_idx:
        bar = "#" * int(coef[idx] * 4)
        print(f"    {feature_names[idx]:25s}  {coef[idx]:+.4f}  |{bar}")

    print(f"\n  Top {top_n} HAM words (largest negative coefficients):")
    for idx in top_ham_idx:
        bar = "#" * int(abs(coef[idx]) * 4)
        print(f"    {feature_names[idx]:25s}  {coef[idx]:+.4f}  |{bar}")

    return model


# ══════════════════════════════════════════════════════════════════
# 5. TRAIN — SVM (LinearSVC + Calibration)
# ══════════════════════════════════════════════════════════════════
def train_svm(X_train, y_train):
    """
    Trains LinearSVC wrapped with CalibratedClassifierCV.
    LinearSVC is faster than kernel SVC for high-dimensional text.
    Calibration adds predict_proba() so we get confidence scores.
    """
    print(f"\n{SEP}")
    print("  STEP 3c: TRAINING SVM (LinearSVC + Platt Calibration)")
    print(SEP2)

    t0 = time.time()
    base_svm = LinearSVC(C=1.0, max_iter=2000, random_state=42)
    # CalibratedClassifierCV wraps the SVM and adds probability estimates
    model = CalibratedClassifierCV(base_svm, cv=5)
    model.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"  Training time  : {elapsed:.3f}s  (includes 5-fold calibration)")
    print(f"  Model type     : LinearSVC + Platt scaling (gives probabilities)")

    return model


# ══════════════════════════════════════════════════════════════════
# 6. SAVE MODEL
# ══════════════════════════════════════════════════════════════════
def save_model(model, filename):
    """Save a trained model to models/ using joblib."""
    path = os.path.join(MODELS_DIR, filename)
    joblib.dump(model, path)
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {filename:35s}  ({size_kb:.1f} KB)")
    return path


# ══════════════════════════════════════════════════════════════════
# 7. MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════
def main():
    if sys.stdout.encoding != "utf-8":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    print(SEP)
    print("  PHASE 4: MODEL TRAINING -- SPAM DETECTION")
    print(SEP)

    total_start = time.time()

    # ── Load & Vectorize ──────────────────────────────────────────
    X, y, vectorizer, feature_names, cleaned_texts = load_and_prepare_data()

    # ── Split ─────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, train_idx, test_idx = split_data(
        X, y, cleaned_texts
    )

    # ── Re-fit vectorizer on TRAIN only (correct ML practice) ─────
    # The X above was fit on ALL data (for demo). Now we properly
    # refit on train and re-transform to avoid data leakage.
    print(f"\n  [Note] Re-fitting vectorizer on TRAIN set only (no data leakage)...")
    texts_array = np.array(cleaned_texts)
    vec_proper = create_tfidf_vectorizer()
    X_train = vec_proper.fit_transform(texts_array[train_idx])
    X_test  = transform_texts(texts_array[test_idx].tolist(), vec_proper)
    feature_names = get_feature_names(vec_proper)
    print(f"  Train features : {X_train.shape[1]:,}  |  Test shape: {X_test.shape}")

    # ── Train ─────────────────────────────────────────────────────
    nb_model = train_naive_bayes(X_train, y_train, feature_names)
    lr_model = train_logistic_regression(X_train, y_train, feature_names)
    sv_model = train_svm(X_train, y_train)

    # ── Save ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  STEP 4: SAVING ALL MODELS + VECTORIZER")
    print(SEP2)
    save_model(nb_model,    "naive_bayes.pkl")
    save_model(lr_model,    "logistic_regression.pkl")
    save_model(sv_model,    "svm.pkl")
    vec_path = save_vectorizer(vec_proper)
    print(f"  Saved: tfidf_vectorizer.pkl")

    # Also save train/test indices so evaluate.py uses the same split
    joblib.dump({"train_idx": train_idx, "test_idx": test_idx},
                os.path.join(MODELS_DIR, "split_indices.pkl"))
    print(f"  Saved: split_indices.pkl  (reuse same split in Phase 5)")

    # ── Quick sanity check on train set ───────────────────────────
    print(f"\n{SEP}")
    print("  STEP 5: QUICK SANITY CHECK (train-set accuracy)")
    print(SEP2)
    models = {
        "Naive Bayes"         : nb_model,
        "Logistic Regression" : lr_model,
        "SVM"                 : sv_model,
    }
    for name, m in models.items():
        train_acc = m.score(X_train, y_train)
        test_acc  = m.score(X_test, y_test)
        print(f"  {name:22s}  train acc: {train_acc:.4f}  test acc: {test_acc:.4f}")

    total_elapsed = time.time() - total_start
    print(f"\n{SEP}")
    print("  TRAINING COMPLETE")
    print(SEP2)
    print(f"""
  Models trained   : Naive Bayes, Logistic Regression, SVM
  Models saved to  : {MODELS_DIR}
  Total time       : {total_elapsed:.1f}s

  Files created:
    models/naive_bayes.pkl
    models/logistic_regression.pkl
    models/svm.pkl
    models/tfidf_vectorizer.pkl
    models/split_indices.pkl

  NEXT STEP -- Phase 5: Evaluation
    Run: python -m src.evaluate
    We will generate confusion matrices, precision, recall,
    F1 scores, and compare all 3 models side by side.
""")
    print(SEP)
    print("  Phase 4 Complete!  Ready for Phase 5: Evaluation")
    print(SEP)


if __name__ == "__main__":
    main()
