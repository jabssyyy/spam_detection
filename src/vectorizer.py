# -*- coding: utf-8 -*-
"""
Phase 3: TF-IDF Vectorization
Converts preprocessed text into sparse numeric feature matrices.
TF-IDF weights words by how characteristic they are of a specific message
relative to the full corpus — high weight = rare across corpus but frequent here.
Run with: python -m src.vectorizer
"""

import sys
import io
import os
import numpy as np
import joblib
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocessing import preprocess_to_string

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_FEATURES  = 5000
NGRAM_RANGE   = (1, 2)
MIN_DF        = 2
MAX_DF        = 0.95
SUBLINEAR_TF  = True


def create_tfidf_vectorizer(
    max_features: int = MAX_FEATURES,
    ngram_range: tuple = NGRAM_RANGE,
    min_df: int = MIN_DF,
    max_df: float = MAX_DF,
    sublinear_tf: bool = SUBLINEAR_TF,
) -> TfidfVectorizer:
    """
    Create a configured TF-IDF vectorizer with production defaults.

    Args:
        max_features: Vocabulary size cap (top N by frequency).
        ngram_range:  Unigrams and bigrams by default.
        min_df:       Ignore words appearing in fewer than min_df docs.
        max_df:       Ignore words appearing in more than max_df fraction of docs.
        sublinear_tf: Apply log(1+tf) to dampen high raw counts.

    Returns:
        TfidfVectorizer: Configured but not yet fitted.
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        analyzer="word",
        token_pattern=r"\S+",
    )


def fit_vectorizer(texts: list, vectorizer: TfidfVectorizer = None) -> TfidfVectorizer:
    """
    Fit the vectorizer on training texts to build the vocabulary.

    Must be called only on training data to avoid data leakage.

    Args:
        texts:      List of preprocessed text strings (train set only).
        vectorizer: Optional pre-created vectorizer; creates new if None.

    Returns:
        TfidfVectorizer: Fitted vectorizer with vocabulary built.
    """
    if vectorizer is None:
        vectorizer = create_tfidf_vectorizer()
    vectorizer.fit(texts)
    return vectorizer


def transform_texts(texts: list, vectorizer: TfidfVectorizer):
    """
    Transform preprocessed texts into a TF-IDF sparse matrix.

    Args:
        texts:      List of preprocessed text strings.
        vectorizer: A fitted TfidfVectorizer instance.

    Returns:
        scipy.sparse.csr_matrix: Shape (n_messages, n_features).
    """
    return vectorizer.transform(texts)


def fit_transform_texts(texts: list, vectorizer: TfidfVectorizer = None):
    """
    Fit and transform in one step. Use only on training data.

    Args:
        texts:      List of preprocessed text strings (train).
        vectorizer: Optional pre-created vectorizer.

    Returns:
        tuple: (fitted_vectorizer, sparse_matrix)
    """
    if vectorizer is None:
        vectorizer = create_tfidf_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def get_feature_names(vectorizer: TfidfVectorizer) -> list:
    """Return the vocabulary as a list of words in column order."""
    return vectorizer.get_feature_names_out().tolist()


def get_top_features_for_message(
    vector_row,
    feature_names: list,
    top_n: int = 5
) -> list:
    """
    Return the top N words by TF-IDF score for a single message vector.

    Args:
        vector_row:    A single row from the TF-IDF sparse matrix.
        feature_names: List of all vocabulary words.
        top_n:         Number of top words to return.

    Returns:
        list[tuple]: [(word, score), ...] sorted by score descending.
    """
    if hasattr(vector_row, "toarray"):
        scores = vector_row.toarray().flatten()
    else:
        scores = np.array(vector_row).flatten()

    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(feature_names[i], round(scores[i], 4)) for i in top_indices if scores[i] > 0]


def save_vectorizer(vectorizer: TfidfVectorizer, path: str = None) -> str:
    """
    Save the fitted vectorizer to disk using joblib.

    The saved vectorizer must be reloaded at prediction time to guarantee
    the vocabulary and column ordering match what the models were trained on.

    Args:
        vectorizer: The fitted TfidfVectorizer to save.
        path:       Save path (defaults to models/tfidf_vectorizer.pkl).

    Returns:
        str: The path where the vectorizer was saved.
    """
    if path is None:
        path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, path)
    return path


def load_vectorizer(path: str = None) -> TfidfVectorizer:
    """
    Load a previously saved TF-IDF vectorizer from disk.

    Args:
        path: Path to the .pkl file (defaults to models/tfidf_vectorizer.pkl).

    Returns:
        TfidfVectorizer: Loaded, ready-to-use vectorizer.
    """
    if path is None:
        path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No vectorizer found at '{path}'.\n"
            "Run src/train.py first to train and save the vectorizer."
        )
    return joblib.load(path)


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    SEP  = "=" * 68
    SEP2 = "-" * 68

    print(SEP)
    print("  PHASE 3: TF-IDF VECTORIZATION -- FULL DEMO")
    print(SEP)

    demo_messages = {
        "spam": [
            "FREE ENTRY in 2 a weekly competition to win FA Cup Final tkts "
            "21st May 2005. Text FA to 87121 to receive entry question(std "
            "txt rate) T&C's apply 08452810075over18's",

            "URGENT! You have won a 1 week FREE membership in our 100000 "
            "Prize Jackpot! To claim call 09010 444 festival. Cost 22p/min.",

            "Congratulations ur awarded 500 of bonus points. 2 claim txt "
            "CLAIM to 87131 [150p/msg]. T&C's SAE Box 1144 UK.",
        ],
        "ham": [
            "I'm gonna be home soon and i don't want to talk about this "
            "stuff anymore tonight, k? I've cried enough today.",

            "I've been searching for the right words to thank you for this "
            "breather. I promise i wont take your help for granted.",

            "Ok lar... Joking wif u oni... Have a nice day ahead",
        ],
    }

    all_raw    = demo_messages["spam"] + demo_messages["ham"]
    all_labels = ["SPAM"] * 3 + ["HAM"] * 3

    # Step 1: Preprocess
    print(f"\n{SEP}")
    print("  STEP 1: PREPROCESS ALL MESSAGES")
    print(SEP2)

    preprocessed = [preprocess_to_string(msg) for msg in all_raw]

    for i, (label, raw, clean) in enumerate(zip(all_labels, all_raw, preprocessed), 1):
        print(f"\n  [{i}] [{label}]")
        print(f"  BEFORE : {raw[:75]}{'...' if len(raw)>75 else ''}")
        print(f"  AFTER  : {clean}")

    # Step 2: Fit vectorizer
    print(f"\n{SEP}")
    print("  STEP 2: BUILD TF-IDF VECTORIZER")
    print(SEP2)

    demo_vec = TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"\S+",
    )
    tfidf_matrix  = demo_vec.fit_transform(preprocessed)
    feature_names = demo_vec.get_feature_names_out().tolist()

    print(f"""
  Vocabulary size  : {len(feature_names)} unique words
  Matrix shape     : {tfidf_matrix.shape}  ({tfidf_matrix.shape[0]} msgs x {tfidf_matrix.shape[1]} words)
  Sparsity         : {100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0]*tfidf_matrix.shape[1])):.1f}% zeros
  Non-zero values  : {tfidf_matrix.nnz} / {tfidf_matrix.shape[0]*tfidf_matrix.shape[1]}
  Vocabulary       : {feature_names}
""")

    # Step 3: Top features per message
    print(f"{SEP}")
    print("  STEP 3: TOP TF-IDF FEATURES PER MESSAGE")
    print(SEP2)

    for i, (label, raw) in enumerate(zip(all_labels, all_raw)):
        row       = tfidf_matrix[i]
        top_feats = get_top_features_for_message(row, feature_names, top_n=7)
        print(f"\n  [{i+1}] [{label}]  {raw[:60]}...")
        for word, score in top_feats:
            bar = "#" * int(score * 40)
            print(f"    {word:20s}  {score:.4f}  |{bar}")

    # Step 4: BoW vs TF-IDF comparison
    print(f"\n{SEP}")
    print("  STEP 4: BAG-OF-WORDS vs TF-IDF COMPARISON")
    print(SEP)

    compare_msgs = [
        "free prize win",
        "free call free win",
        "call me tonight",
    ]

    from sklearn.feature_extraction.text import CountVectorizer
    bow_vec   = CountVectorizer(token_pattern=r"\S+")
    bow_mat   = bow_vec.fit_transform(compare_msgs)
    tfidf_cv  = TfidfVectorizer(sublinear_tf=False, token_pattern=r"\S+")
    tfidf_mat = tfidf_cv.fit_transform(compare_msgs)

    vocab   = bow_vec.get_feature_names_out()
    t_vocab = tfidf_cv.get_feature_names_out()

    print(f"\n  Messages: {compare_msgs}")

    print(f"\n  BAG-OF-WORDS (raw counts):")
    print(f"  {'':5s}", end="")
    for w in vocab:
        print(f"  {w:>7s}", end="")
    print()
    for i, row in enumerate(bow_mat.toarray(), 1):
        print(f"  msg{i}: ", end="")
        for v in row:
            print(f"  {v:>7}", end="")
        print()

    print(f"\n  TF-IDF (weighted scores):")
    print(f"  {'':5s}", end="")
    for w in t_vocab:
        print(f"  {w:>7s}", end="")
    print()
    for i, row in enumerate(tfidf_mat.toarray(), 1):
        print(f"  msg{i}: ", end="")
        for v in row:
            print(f"  {v:>7.3f}", end="")
        print()

    # Step 5: Production vectorizer on real dataset
    print(f"\n{SEP}")
    print("  STEP 5: PRODUCTION VECTORIZER -- REAL DATASET")
    print(SEP2)

    import pandas as pd
    DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
    df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "message"]

    print(f"  Preprocessing {len(df):,} messages...")
    df["clean"] = df["message"].apply(preprocess_to_string)

    prod_vec     = create_tfidf_vectorizer()
    prod_matrix  = prod_vec.fit_transform(df["clean"])
    prod_features = get_feature_names(prod_vec)

    density          = prod_matrix.nnz / (prod_matrix.shape[0] * prod_matrix.shape[1]) * 100
    memory_dense_mb  = prod_matrix.shape[0] * prod_matrix.shape[1] * 8 / 1e6

    print(f"""
  Total messages  : {len(df):,}
  Vocabulary size : {len(prod_features):,} features (words + bigrams)
  Matrix shape    : {prod_matrix.shape[0]:,} x {prod_matrix.shape[1]:,}
  Non-zero cells  : {prod_matrix.nnz:,}
  Sparsity        : {100 - density:.2f}% zeros
  Memory (dense)  : {memory_dense_mb:.0f} MB  vs  sparse: ~{prod_matrix.nnz * 12 / 1e6:.1f} MB  (~{memory_dense_mb / (prod_matrix.nnz * 12 / 1e6):.0f}x smaller)

  Sample vocab (first 20): {prod_features[:20]}
""")

    spam_like = [f for f in prod_features if any(
        w in f for w in ["free", "win", "prize", "call", "claim", "urgent", "reply"]
    )][:15]
    print("  Spam-flavored bigrams found:")
    for f in spam_like:
        print(f"    {f}")

    # Step 6: Top words per class
    print(f"\n{SEP}")
    print("  STEP 6: TOP PREDICTIVE WORDS -- SPAM vs HAM")
    print(SEP2)

    spam_mask = df["label"] == "spam"
    ham_mask  = df["label"] == "ham"

    spam_scores = np.asarray(prod_matrix[spam_mask.values].mean(axis=0)).flatten()
    ham_scores  = np.asarray(prod_matrix[ham_mask.values].mean(axis=0)).flatten()

    top_spam_idx = np.argsort(spam_scores)[::-1][:15]
    top_ham_idx  = np.argsort(ham_scores)[::-1][:15]

    print(f"\n  TOP 15 WORDS BY AVG TF-IDF IN SPAM:")
    print(f"  {'Rank':5s}  {'Feature':25s}  {'Avg TF-IDF':>10s}  Bar")
    print(f"  {'-'*5}  {'-'*25}  {'-'*10}  {'-'*30}")
    for rank, idx in enumerate(top_spam_idx, 1):
        bar = "#" * int(spam_scores[idx] * 600)
        print(f"  {rank:5d}  {prod_features[idx]:25s}  {spam_scores[idx]:>10.4f}  {bar}")

    print(f"\n  TOP 15 WORDS BY AVG TF-IDF IN HAM:")
    print(f"  {'Rank':5s}  {'Feature':25s}  {'Avg TF-IDF':>10s}  Bar")
    print(f"  {'-'*5}  {'-'*25}  {'-'*10}  {'-'*30}")
    for rank, idx in enumerate(top_ham_idx, 1):
        bar = "#" * int(ham_scores[idx] * 600)
        print(f"  {rank:5d}  {prod_features[idx]:25s}  {ham_scores[idx]:>10.4f}  {bar}")

    # Step 7: Visualization
    print(f"\n{SEP}")
    print("  STEP 7: GENERATING VISUALIZATION...")
    print(SEP2)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0d1117")
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # Panel 1: Top spam words
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161b22")
    top15_spam_words  = [prod_features[i] for i in top_spam_idx]
    top15_spam_scores = [spam_scores[i]   for i in top_spam_idx]
    colors_spam = plt.cm.Reds(np.linspace(0.4, 0.9, 15))[::-1]
    ax1.barh(range(15), top15_spam_scores[::-1], color=colors_spam, edgecolor="none")
    ax1.set_yticks(range(15))
    ax1.set_yticklabels(top15_spam_words[::-1], color="white", fontsize=9)
    ax1.set_xlabel("Avg TF-IDF Score in SPAM", color="#aaaaaa")
    ax1.set_title("Top 15 Spam Indicator Words", color="#ff4757", fontsize=13, pad=10)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # Panel 2: Top ham words
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22")
    top15_ham_words  = [prod_features[i] for i in top_ham_idx]
    top15_ham_scores = [ham_scores[i]    for i in top_ham_idx]
    colors_ham = plt.cm.Blues(np.linspace(0.4, 0.9, 15))[::-1]
    ax2.barh(range(15), top15_ham_scores[::-1], color=colors_ham, edgecolor="none")
    ax2.set_yticks(range(15))
    ax2.set_yticklabels(top15_ham_words[::-1], color="white", fontsize=9)
    ax2.set_xlabel("Avg TF-IDF Score in HAM", color="#aaaaaa")
    ax2.set_title("Top 15 Ham Indicator Words", color="#00d2ff", fontsize=13, pad=10)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    # Panel 3: Spam vs Ham score comparison
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#161b22")
    compare_words = top15_spam_words[:10]
    compare_idx   = top_spam_idx[:10]
    compare_spam  = [spam_scores[i] for i in compare_idx]
    compare_ham   = [ham_scores[i]  for i in compare_idx]
    x = np.arange(len(compare_words))
    w = 0.35
    ax3.bar(x - w/2, compare_spam, w, label="Spam", color="#ff4757", alpha=0.85)
    ax3.bar(x + w/2, compare_ham,  w, label="Ham",  color="#00d2ff", alpha=0.85)
    ax3.set_xticks(x)
    ax3.set_xticklabels(compare_words, rotation=35, ha="right", color="white", fontsize=8)
    ax3.set_ylabel("Avg TF-IDF Score", color="#aaaaaa")
    ax3.set_title("Spam vs Ham: Key Word Scores", color="white", fontsize=13, pad=10)
    ax3.legend(facecolor="#1c2128", edgecolor="#30363d", labelcolor="white")
    ax3.tick_params(colors="white")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#30363d")

    # Panel 4: Matrix sparsity heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161b22")
    sample_matrix = prod_matrix[:50, :100].toarray()
    im = ax4.imshow(sample_matrix, aspect="auto", cmap="magma", interpolation="nearest")
    ax4.set_title(
        "TF-IDF Matrix Heatmap\n(first 50 messages x 100 words)",
        color="white", fontsize=12, pad=10
    )
    ax4.set_xlabel("Word Index (of top 100 features)", color="#aaaaaa")
    ax4.set_ylabel("Message Index", color="#aaaaaa")
    ax4.tick_params(colors="white")
    for spine in ax4.spines.values():
        spine.set_edgecolor("#30363d")
    cbar = plt.colorbar(im, ax=ax4)
    cbar.set_label("TF-IDF Score", color="white")
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.suptitle(
        "SPAM DETECTION -- Phase 3: TF-IDF Vectorization",
        color="white", fontsize=16, fontweight="bold", y=1.01
    )

    chart_path = os.path.join(OUTPUT_DIR, "phase3_tfidf.png")
    plt.savefig(chart_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    print(f"  [OK] Visualization saved to: {chart_path}")

    # Step 8: Save production vectorizer
    print(f"\n{SEP}")
    print("  STEP 8: SAVING PRODUCTION VECTORIZER")
    print(SEP2)

    saved_path = save_vectorizer(prod_vec)
    print(f"  [OK] Vectorizer saved to: {saved_path}")

    print(f"\n{SEP}")
    print("  Phase 3 Complete!  Ready for Phase 4: Model Training")
    print(SEP)
