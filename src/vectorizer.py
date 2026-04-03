# -*- coding: utf-8 -*-
"""
===================================================================
PHASE 3: Text Vectorization (TF-IDF) -- Spam Detection Project
===================================================================

PURPOSE OF THIS FILE:
    After Phase 2 we have CLEAN text tokens:
        ["free", "entry", "win", "prize", "call"]

    But ML models are pure math -- they only understand NUMBERS.
    This file converts those clean tokens into numeric vectors
    using the TF-IDF technique.

WHAT THIS FILE BUILDS:
    create_tfidf_vectorizer()    -- configure the TF-IDF object
    fit_vectorizer(texts)        -- learn vocabulary from training data
    transform_texts(texts, vec)  -- convert messages to numeric matrix
    fit_transform(texts)         -- fit + transform in one step
    get_feature_names(vec)       -- retrieve vocabulary words
    get_top_features(vec, mat)   -- top words per message
    save_vectorizer(vec, path)   -- persist vectorizer to disk
    load_vectorizer(path)        -- reload saved vectorizer

===================================================================
CONCEPT PRIMER (read this before the code!)
===================================================================

--- What is a VOCABULARY? ---

    A vocabulary is the complete set of unique words across ALL
    messages in our training dataset.

    Example (tiny dataset of 3 messages):
        msg 1: "free prize win now"
        msg 2: "call me tonight please"
        msg 3: "free entry win prize"

    Vocabulary = {"free", "prize", "win", "now", "call",
                  "me", "tonight", "please", "entry"}
    Size = 9 words

    In our REAL dataset (~5500 messages, after preprocessing)
    the vocabulary will be roughly 5,000-8,000 unique words.
    We'll cap it at MAX_FEATURES = 5000 to control complexity.

--- What is BAG OF WORDS (BoW)? ---

    Each message becomes a vector of word COUNTS.
    Vocabulary position maps to a vector index.

    vocab:     free  prize  win  now  call  me  tonight  please  entry
    index:      [0]   [1]   [2]  [3]  [4]  [5]   [6]     [7]    [8]

    msg 1: "free prize win now"     --> [1, 1, 1, 1, 0, 0, 0, 0, 0]
    msg 3: "free entry win prize"   --> [1, 1, 1, 0, 0, 0, 0, 0, 1]

    PROBLEM WITH BoW:
        Frequent words like "the", "a", "is" get high counts in
        EVERY message and dominate the vector -- even after removing
        stopwords, common domain words (e.g., "call") still appear
        in both spam and ham.  BoW treats all words equally.

--- What is TF-IDF? ---

    TF-IDF = Term Frequency × Inverse Document Frequency

    It weights each word by TWO factors:

    TF (Term Frequency):
        How often does this word appear in THIS message?
        If "free" appears 3 times in a 10-word message: TF = 3/10 = 0.3
        More occurrences in a message = higher TF.

    IDF (Inverse Document Frequency):
        How RARE is this word across ALL messages?
        IDF = log( N / df ) where:
            N  = total number of messages
            df = number of messages that contain this word

        If "free" appears in 50 of 5000 messages:
            IDF = log(5000/50) = log(100) ≈ 4.6   (HIGH -- rare = important)

        If "call" appears in 2000 of 5000 messages:
            IDF = log(5000/2000) = log(2.5) ≈ 0.9  (LOW -- common = less useful)

    TF-IDF = TF × IDF
        A word that appears OFTEN in THIS message BUT RARELY across
        all messages gets a HIGH score -- that's the "signature" word.

    INTUITION for spam detection:
        "free" is rare across all legitimate SMS messages BUT
        appears very frequently in spam -> HIGH TF-IDF in spam.
        "prize", "win", "claim" work the same way.

        Meanwhile "call" appears in everyday messages too,
        so its IDF is lower -> less discriminating.

--- What does the OUTPUT look like? ---

    After vectorization, we get a 2D MATRIX:

        Rows    = messages    (one row per SMS)
        Columns = vocabulary  (one column per unique word)
        Values  = TF-IDF scores

    Shape: (5572 messages, 5000 words)
    That's 5572 × 5000 = 27,860,000 cells!

    MOST cells are ZERO (a message uses ~20 words from 5000 vocab).
    Storing 27 million mostly-zero floats wastes memory.

    Solution: SPARSE MATRIX
        Only stores the NON-ZERO values and their positions.
        A message with 20 words only stores 20 values -- not 5000.
        Memory drops from ~220 MB to ~2-3 MB. Huge difference!

===================================================================
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
from sklearn.preprocessing import normalize

# Import our Phase 2 preprocessing pipeline
from src.preprocessing import preprocess_to_string, preprocess_text

# ---------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, "models")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------
# These are the parameters we'll pass to TfidfVectorizer.
# Each one is explained in create_tfidf_vectorizer() below.
MAX_FEATURES  = 5000    # only keep top 5000 most frequent words
NGRAM_RANGE   = (1, 2)  # use single words AND two-word pairs
MIN_DF        = 2       # ignore words appearing in fewer than 2 docs
MAX_DF        = 0.95    # ignore words appearing in >95% of docs
SUBLINEAR_TF  = True    # apply log(1+tf) instead of raw tf


# ===================================================================
# FUNCTION 1: create_tfidf_vectorizer
# ===================================================================
def create_tfidf_vectorizer(
    max_features: int = MAX_FEATURES,
    ngram_range: tuple = NGRAM_RANGE,
    min_df: int = MIN_DF,
    max_df: float = MAX_DF,
    sublinear_tf: bool = SUBLINEAR_TF,
) -> TfidfVectorizer:
    """
    Create and configure a TF-IDF vectorizer with sensible defaults.

    PARAMETERS EXPLAINED:

    max_features=5000
        Keep only the top 5000 words by term frequency across all docs.
        WHY? Our vocab could be 8000+ words. Many are rare typos or
        abbreviations that appear once. Keeping 5000 gives the model
        enough signal without overfitting to rare noise.

    ngram_range=(1,2)
        Use both UNIGRAMS (single words) and BIGRAMS (word pairs).
        WHY? Some spam signals are TWO words together:
            "free entry"  -->  "free" + "entry" + "free entry"
            "call now"    -->  "call" + "now"   + "call now"
            "win prize"   -->  "win"  + "prize" + "win prize"
        Bigrams capture these compound patterns that unigrams miss.

    min_df=2
        Ignore words that appear in fewer than 2 documents.
        WHY? A word appearing in only 1 document is probably a
        typo or name-specific. It has zero generalization power.
        Removing it shrinks the vocabulary and reduces noise.

    max_df=0.95
        Ignore words that appear in more than 95% of documents.
        WHY? A word in 95% of all messages is basically a stopword
        that slipped through. It carries no discriminating signal.

    sublinear_tf=True
        Replace raw term frequency with log(1 + tf).
        WHY? Without this, a word appearing 10 times gets exactly
        10x the score of a word appearing once. But that's not how
        language works -- the 10th occurrence adds less meaning
        than the 1st. log(1+10)=2.4 vs log(1+1)=0.69  is more
        realistic. This helps the model focus on presence vs absence.

    analyzer='word'
        Tokenize at word level (not character level).
        Our preprocessor already handled tokenization.

    token_pattern
        Since we already preprocessed, set to match any non-space
        sequence. Default sklearn pattern requires 2+ chars which
        could drop some short meaningful tokens.

    Returns:
        TfidfVectorizer: A configured (but not yet fitted) vectorizer
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df,
        sublinear_tf=sublinear_tf,
        analyzer="word",
        token_pattern=r"\S+",   # match any non-whitespace sequence
    )
    return vectorizer


# ===================================================================
# FUNCTION 2: fit_vectorizer
# ===================================================================
def fit_vectorizer(texts: list, vectorizer: TfidfVectorizer = None) -> TfidfVectorizer:
    """
    Fit the vectorizer on a list of preprocessed text strings.
    Fitting BUILDS the vocabulary from the training data.

    *** CRITICAL RULE: Only fit on TRAINING data, never on test data! ***

    WHY ONLY TRAINING DATA?
        Imagine you study for an exam using a textbook (training).
        On exam day, you're asked about topics NOT in your textbook (test leakage).

        If we fit on ALL data including the test set, the vectorizer
        will learn vocabulary words that "came from the future" --
        words that appear only in test messages.

        This inflates performance metrics and makes your model look
        better than it really is.  In production, new messages won't
        have been seen before -- same as test data.

        The correct flow:
            1. Split data: 80% train, 20% test
            2. .fit(train_texts)        --> build vocabulary from train
            3. .transform(train_texts)  --> convert train to vectors
            4. .transform(test_texts)   --> convert test using SAME vocab
               (any word in test not seen in train gets score of 0 -- correct!)

    Args:
        texts (list[str]): List of preprocessed text strings (from train set)
        vectorizer:        Optional pre-created vectorizer (creates new if None)

    Returns:
        TfidfVectorizer: Fitted vectorizer with vocabulary built
    """
    if vectorizer is None:
        vectorizer = create_tfidf_vectorizer()
    vectorizer.fit(texts)
    return vectorizer


# ===================================================================
# FUNCTION 3: transform_texts
# ===================================================================
def transform_texts(texts: list, vectorizer: TfidfVectorizer):
    """
    Transform a list of preprocessed text strings into a TF-IDF matrix.

    WHAT YOU GET BACK:
        A scipy sparse matrix of shape (len(texts), vocab_size).

        Each ROW = one message
        Each COL = one word in the vocabulary
        Value    = TF-IDF score (0.0 to ~1.0 after normalization)

    WHY A SPARSE MATRIX?
        Dense matrix for 5572 messages x 5000 vocab:
            5572 x 5000 x 8 bytes = 223 MB!

        Most cells are ZERO (a 20-word message has 4980 zeros).
        Sparse format only stores non-zero values + their indices.
        Memory: ~3 MB instead of 223 MB. 74x smaller!

        Scikit-learn's models (NB, LR, SVM) all accept sparse
        matrices natively -- no conversion needed.

    NOTE: The vectorizer must be FITTED before calling this.
          Any word not seen during fitting gets a score of 0.

    Args:
        texts (list[str]): List of preprocessed text strings
        vectorizer:        A FITTED TfidfVectorizer instance

    Returns:
        scipy.sparse.csr_matrix: Shape (n_messages, n_features)
    """
    return vectorizer.transform(texts)


# ===================================================================
# FUNCTION 4: fit_transform_texts
# ===================================================================
def fit_transform_texts(texts: list, vectorizer: TfidfVectorizer = None):
    """
    Fit the vectorizer AND transform texts in one step.

    Use this ONLY on training data.
    For test/validation data, use transform_texts() with the
    already-fitted vectorizer.

    Args:
        texts (list[str]): List of preprocessed text strings (train)
        vectorizer:        Optional pre-created vectorizer

    Returns:
        tuple: (fitted_vectorizer, sparse_matrix)
    """
    if vectorizer is None:
        vectorizer = create_tfidf_vectorizer()
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


# ===================================================================
# FUNCTION 5: get_feature_names
# ===================================================================
def get_feature_names(vectorizer: TfidfVectorizer) -> list:
    """
    Get the list of words (features) the vectorizer knows about.

    This is the VOCABULARY -- one word per column in the TF-IDF matrix.
    The i-th word corresponds to the i-th column in the matrix.

    WHY USEFUL?
        - Debugging: see what words the model uses
        - Interpretation: see which words have high weights for spam
        - Feature importance: find the most predictive spam words

    Args:
        vectorizer: A fitted TfidfVectorizer

    Returns:
        list[str]: All vocabulary words in column order
    """
    return vectorizer.get_feature_names_out().tolist()


# ===================================================================
# FUNCTION 6: get_top_features_for_message
# ===================================================================
def get_top_features_for_message(
    vector_row,
    feature_names: list,
    top_n: int = 5
) -> list:
    """
    For a single message's TF-IDF vector, return the top N words
    with the highest TF-IDF scores.

    Use this to understand WHY the model thinks a message is spam:
        "Message flagged because: 'free'(0.72), 'win'(0.61), 'prize'(0.58)"

    Args:
        vector_row:       A single row from the TF-IDF sparse matrix
        feature_names:    List of all vocabulary words
        top_n:            How many top words to return

    Returns:
        list[tuple]: [(word, score), ...] sorted by score descending
    """
    # Convert sparse row to a flat numpy array
    if hasattr(vector_row, "toarray"):
        scores = vector_row.toarray().flatten()
    else:
        scores = np.array(vector_row).flatten()

    # Get indices of top-n non-zero scores
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(feature_names[i], round(scores[i], 4)) for i in top_indices if scores[i] > 0]


# ===================================================================
# FUNCTION 7: save_vectorizer
# ===================================================================
def save_vectorizer(vectorizer: TfidfVectorizer, path: str = None) -> str:
    """
    Save the fitted vectorizer to disk using joblib.

    *** WHY SAVING THE VECTORIZER IS CRITICAL ***

    When you deploy the spam API (Phase 6), a user sends a new message.
    The API must:
        1. Preprocess the message   (Phase 2 functions)
        2. Convert to TF-IDF vector (needs the SAME vectorizer!)
        3. Pass vector to the model (Phase 4)

    If you create a NEW vectorizer at prediction time, it would have
    a DIFFERENT vocabulary order -- the columns would mean different
    words!  The model was trained on column 47 = "free", but a new
    vectorizer might put "free" at column 312.  Disaster!

    By saving the vectorizer, you guarantee the vocabulary is identical
    between training time and prediction time.

    joblib vs pickle:
        Both serialize Python objects. joblib is optimized for large
        numpy arrays (like our vocabulary) -- it's faster and smaller.

    Args:
        vectorizer: The fitted TfidfVectorizer to save
        path:       Where to save (defaults to models/tfidf_vectorizer.pkl)

    Returns:
        str: The path where the vectorizer was saved
    """
    if path is None:
        path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    joblib.dump(vectorizer, path)
    return path


# ===================================================================
# FUNCTION 8: load_vectorizer
# ===================================================================
def load_vectorizer(path: str = None) -> TfidfVectorizer:
    """
    Load a previously saved TF-IDF vectorizer from disk.

    Use this in the API (Phase 6) and at prediction time to ensure
    the vocabulary matches exactly what the model was trained on.

    Args:
        path: Path to the .pkl file (defaults to models/tfidf_vectorizer.pkl)

    Returns:
        TfidfVectorizer: The loaded, ready-to-use vectorizer
    """
    if path is None:
        path = os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No vectorizer found at '{path}'.\n"
            "Run src/train.py first to train and save the vectorizer."
        )
    return joblib.load(path)


# ===================================================================
# DEMONSTRATION -- runs when you execute this file directly
# ===================================================================
if __name__ == "__main__":
    # Fix Windows terminal encoding only when running directly
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    SEP  = "=" * 68
    SEP2 = "-" * 68

    print(SEP)
    print("  PHASE 3: TF-IDF VECTORIZATION -- FULL DEMO")
    print(SEP)

    # -----------------------------------------------------------
    # Sample messages for the demo
    # -----------------------------------------------------------
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

    all_raw   = demo_messages["spam"] + demo_messages["ham"]
    all_labels = ["SPAM"] * 3 + ["HAM"] * 3

    # -----------------------------------------------------------
    # STEP 1: Preprocess all messages
    # -----------------------------------------------------------
    print(f"\n\n{SEP}")
    print("  STEP 1: PREPROCESS ALL MESSAGES (Phase 2 pipeline)")
    print(SEP2)

    preprocessed = [preprocess_to_string(msg) for msg in all_raw]

    for i, (label, raw, clean) in enumerate(zip(all_labels, all_raw, preprocessed), 1):
        print(f"\n  [{i}] [{label}]")
        print(f"  BEFORE : {raw[:75]}{'...' if len(raw)>75 else ''}")
        print(f"  AFTER  : {clean}")

    # -----------------------------------------------------------
    # STEP 2: Build TF-IDF Vectorizer on these 6 messages
    # -----------------------------------------------------------
    print(f"\n\n{SEP}")
    print("  STEP 2: BUILD TF-IDF VECTORIZER")
    print(SEP2)

    # For demo, use minimal settings so we can see the full vocab
    demo_vec = TfidfVectorizer(
        ngram_range=(1, 1),
        min_df=1,
        sublinear_tf=True,
        token_pattern=r"\S+",
    )
    tfidf_matrix = demo_vec.fit_transform(preprocessed)
    feature_names = demo_vec.get_feature_names_out().tolist()

    print(f"""
  Vectorizer fitted on {len(preprocessed)} messages.

  VOCABULARY SIZE  : {len(feature_names)} unique words
  MATRIX SHAPE     : {tfidf_matrix.shape}
                     ({tfidf_matrix.shape[0]} messages x {tfidf_matrix.shape[1]} unique words)

  SPARSITY         : {100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0]*tfidf_matrix.shape[1])):.1f}% of matrix cells are ZERO
  NON-ZERO VALUES  : {tfidf_matrix.nnz} out of {tfidf_matrix.shape[0]*tfidf_matrix.shape[1]} total cells
  MATRIX TYPE      : {type(tfidf_matrix).__name__} (sparse -- efficient!)

  FULL VOCABULARY  : {feature_names}
""")

    # -----------------------------------------------------------
    # STEP 3: Show TF-IDF scores for spam vs ham
    # -----------------------------------------------------------
    print(f"\n{SEP}")
    print("  STEP 3: TOP TF-IDF FEATURES PER MESSAGE")
    print(SEP2)

    for i, (label, raw) in enumerate(zip(all_labels, all_raw)):
        row      = tfidf_matrix[i]
        top_feats = get_top_features_for_message(row, feature_names, top_n=7)
        print(f"\n  [{i+1}] [{label}]")
        print(f"  MSG  : {raw[:60]}...")
        print(f"  TOP FEATURES:")
        for word, score in top_feats:
            bar = "#" * int(score * 40)
            print(f"    {word:20s}  {score:.4f}  |{bar}")

    # -----------------------------------------------------------
    # STEP 4: Show a Bag-of-Words vs TF-IDF comparison (tiny example)
    # -----------------------------------------------------------
    print(f"\n\n{SEP}")
    print("  STEP 4: BAG-OF-WORDS vs TF-IDF COMPARISON")
    print(SEP)

    compare_msgs = [
        "free prize win",        # spam-like
        "free call free win",    # spam-like with repetition
        "call me tonight",       # ham-like
    ]

    from sklearn.feature_extraction.text import CountVectorizer
    bow_vec   = CountVectorizer(token_pattern=r"\S+")
    bow_mat   = bow_vec.fit_transform(compare_msgs)
    tfidf_cv  = TfidfVectorizer(sublinear_tf=False, token_pattern=r"\S+")
    tfidf_mat = tfidf_cv.fit_transform(compare_msgs)

    vocab    = bow_vec.get_feature_names_out()
    t_vocab  = tfidf_cv.get_feature_names_out()

    print(f"\n  MESSAGES:")
    for i, msg in enumerate(compare_msgs, 1):
        print(f"    [{i}] {msg}")

    print(f"\n  BAG-OF-WORDS MATRIX (raw counts):")
    print(f"  {'':5s}", end="")
    for w in vocab:
        print(f"  {w:>7s}", end="")
    print()
    for i, row in enumerate(bow_mat.toarray(), 1):
        print(f"  msg{i}: ", end="")
        for v in row:
            print(f"  {v:>7}", end="")
        print()

    print(f"\n  TF-IDF MATRIX (weighted scores):")
    print(f"  {'':5s}", end="")
    for w in t_vocab:
        print(f"  {w:>7s}", end="")
    print()
    for i, row in enumerate(tfidf_mat.toarray(), 1):
        print(f"  msg{i}: ", end="")
        for v in row:
            print(f"  {v:>7.3f}", end="")
        print()

    print(f"""
  INSIGHT:
    In BoW, "free" appears twice in msg2 -- score doubles (2 vs 1).
    In TF-IDF, repetition is dampened (sublinear_tf log scales it).

    "call" appears in msg3 ONLY -- so its IDF is high (rare = important).
    "free" + "win" appear in MULTIPLE messages -- lower IDF.

    The KEY insight: TF-IDF rewards words that are CHARACTERISTIC
    of a specific message, not words that appear everywhere.
""")

    # -----------------------------------------------------------
    # STEP 5: Production vectorizer on real dataset
    # -----------------------------------------------------------
    print(f"\n{SEP}")
    print("  STEP 5: PRODUCTION VECTORIZER -- THE REAL DATASET")
    print(SEP2)

    import pandas as pd
    DATA_PATH = os.path.join(BASE_DIR, "data", "spam.csv")
    df = pd.read_csv(DATA_PATH, encoding="latin-1", usecols=[0, 1])
    df.columns = ["label", "message"]

    print(f"  Preprocessing {len(df):,} messages... (this takes a few seconds)")
    df["clean"] = df["message"].apply(preprocess_to_string)

    # Create production-grade vectorizer
    prod_vec = create_tfidf_vectorizer()
    prod_matrix = prod_vec.fit_transform(df["clean"])
    prod_features = get_feature_names(prod_vec)

    density = prod_matrix.nnz / (prod_matrix.shape[0] * prod_matrix.shape[1]) * 100
    memory_dense_mb = prod_matrix.shape[0] * prod_matrix.shape[1] * 8 / 1e6

    print(f"""
  PRODUCTION VECTORIZER RESULTS:
    Total messages      : {len(df):,}
    Vocabulary size     : {len(prod_features):,} features (words + bigrams)
    Matrix shape        : {prod_matrix.shape[0]:,} x {prod_matrix.shape[1]:,}
    Non-zero cells      : {prod_matrix.nnz:,}
    Sparsity            : {100 - density:.2f}% zeros
    Density             : {density:.2f}% non-zero

    Memory (dense)      : {memory_dense_mb:.0f} MB  (if we stored ALL values)
    Memory (sparse)     : ~{prod_matrix.nnz * 12 / 1e6:.1f} MB  (sparse stores only non-zeros)
    Compression ratio   : ~{memory_dense_mb / (prod_matrix.nnz * 12 / 1e6):.0f}x smaller!

  SAMPLE VOCABULARY (first 20 features):
    {prod_features[:20]}

  SAMPLE VOCABULARY (some spam-flavored features -- bigrams!):
""")
    spam_like = [f for f in prod_features if any(
        w in f for w in ["free", "win", "prize", "call", "claim", "urgent", "reply"]
    )][:15]
    for f in spam_like:
        print(f"    {f}")

    # -----------------------------------------------------------
    # STEP 6: Analyze top spam vs ham words
    # -----------------------------------------------------------
    print(f"\n\n{SEP}")
    print("  STEP 6: TOP PREDICTIVE WORDS -- SPAM vs HAM")
    print(SEP2)

    spam_mask = df["label"] == "spam"
    ham_mask  = df["label"] == "ham"

    spam_matrix = prod_matrix[spam_mask.values]
    ham_matrix  = prod_matrix[ham_mask.values]

    spam_scores = np.asarray(spam_matrix.mean(axis=0)).flatten()
    ham_scores  = np.asarray(ham_matrix.mean(axis=0)).flatten()

    # Words with highest average TF-IDF in SPAM
    top_spam_idx = np.argsort(spam_scores)[::-1][:15]
    top_ham_idx  = np.argsort(ham_scores)[::-1][:15]

    print(f"\n  TOP 15 WORDS BY AVERAGE TF-IDF IN SPAM MESSAGES:")
    print(f"  {'Rank':5s}  {'Feature':25s}  {'Avg TF-IDF':>10s}  Bar")
    print(f"  {'-'*5}  {'-'*25}  {'-'*10}  {'-'*30}")
    for rank, idx in enumerate(top_spam_idx, 1):
        bar = "#" * int(spam_scores[idx] * 600)
        print(f"  {rank:5d}  {prod_features[idx]:25s}  {spam_scores[idx]:>10.4f}  {bar}")

    print(f"\n  TOP 15 WORDS BY AVERAGE TF-IDF IN HAM MESSAGES:")
    print(f"  {'Rank':5s}  {'Feature':25s}  {'Avg TF-IDF':>10s}  Bar")
    print(f"  {'-'*5}  {'-'*25}  {'-'*10}  {'-'*30}")
    for rank, idx in enumerate(top_ham_idx, 1):
        bar = "#" * int(ham_scores[idx] * 600)
        print(f"  {rank:5d}  {prod_features[idx]:25s}  {ham_scores[idx]:>10.4f}  {bar}")

    print(f"""
  KEY INSIGHT:
    Notice how SPAM top words are very ACTION/LURE oriented:
      "free", "call", "txt", "reply", "claim", "prize", "win", "urgent"

    HAM top words are CONVERSATIONAL and PERSONAL:
      "ok", "come", "know", "think", "good", "time", "want", "need"

    TF-IDF has surfaced the REAL discriminating signal automatically.
    The model will learn: high "free"/"win"/"prize" score --> SPAM.
""")

    # -----------------------------------------------------------
    # STEP 7: Visualize
    # -----------------------------------------------------------
    print(f"\n{SEP}")
    print("  STEP 7: GENERATING VISUALIZATION...")
    print(SEP2)

    plt.style.use("dark_background")
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#0d1117")

    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # ---- Panel 1: Top Spam Features ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor("#161b22")
    top15_spam_words  = [prod_features[i] for i in top_spam_idx]
    top15_spam_scores = [spam_scores[i]   for i in top_spam_idx]
    colors_spam = plt.cm.Reds(np.linspace(0.4, 0.9, 15))[::-1]
    bars = ax1.barh(
        range(15), top15_spam_scores[::-1],
        color=colors_spam, edgecolor="none"
    )
    ax1.set_yticks(range(15))
    ax1.set_yticklabels(top15_spam_words[::-1], color="white", fontsize=9)
    ax1.set_xlabel("Avg TF-IDF Score in SPAM", color="#aaaaaa")
    ax1.set_title("Top 15 Spam Indicator Words", color="#ff4757", fontsize=13, pad=10)
    ax1.tick_params(colors="white")
    for spine in ax1.spines.values():
        spine.set_edgecolor("#30363d")

    # ---- Panel 2: Top Ham Features ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor("#161b22")
    top15_ham_words  = [prod_features[i] for i in top_ham_idx]
    top15_ham_scores = [ham_scores[i]    for i in top_ham_idx]
    colors_ham = plt.cm.Blues(np.linspace(0.4, 0.9, 15))[::-1]
    ax2.barh(
        range(15), top15_ham_scores[::-1],
        color=colors_ham, edgecolor="none"
    )
    ax2.set_yticks(range(15))
    ax2.set_yticklabels(top15_ham_words[::-1], color="white", fontsize=9)
    ax2.set_xlabel("Avg TF-IDF Score in HAM", color="#aaaaaa")
    ax2.set_title("Top 15 Ham Indicator Words", color="#00d2ff", fontsize=13, pad=10)
    ax2.tick_params(colors="white")
    for spine in ax2.spines.values():
        spine.set_edgecolor("#30363d")

    # ---- Panel 3: Spam vs Ham score comparison for TOP spam words ----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor("#161b22")
    compare_words  = top15_spam_words[:10]
    compare_idx    = top_spam_idx[:10]
    compare_spam   = [spam_scores[i] for i in compare_idx]
    compare_ham    = [ham_scores[i]  for i in compare_idx]
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

    # ---- Panel 4: Sparsity visualization (sample 50 messages) ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor("#161b22")
    sample_matrix = prod_matrix[:50, :100].toarray()
    im = ax4.imshow(
        sample_matrix, aspect="auto", cmap="magma",
        interpolation="nearest"
    )
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

    # -----------------------------------------------------------
    # STEP 8: Save the production vectorizer
    # -----------------------------------------------------------
    print(f"\n{SEP}")
    print("  STEP 8: SAVING PRODUCTION VECTORIZER")
    print(SEP2)

    saved_path = save_vectorizer(prod_vec)
    print(f"  [OK] Vectorizer saved to: {saved_path}")
    print(f"""
  WHY WE SAVE IT:
    When the API (Phase 6) receives a new message like:
        "You've won a free prize! Call now!"

    It must convert it using the EXACT SAME vocabulary and
    column ordering as when the model was trained.

    Loading the saved vectorizer guarantees this consistency.
    Without it, the model's column 47 might mean a completely
    different word than what the API sends.
""")

    # -----------------------------------------------------------
    # FINAL DATA FLOW SUMMARY
    # -----------------------------------------------------------
    print(f"\n{SEP}")
    print("  DATA FLOW SUMMARY -- PHASES 1 TO 3")
    print(SEP)
    print(f"""
  PHASE 1: Raw Data
    spam.csv --> 5,572 rows x 2 columns (label, message)
    "FREE ENTRY Win a prize call 0800 now!"    label: spam

  PHASE 2: Preprocessing
    Raw text
      --> lowercase        : "free entry win a prize call 0800 now!"
      --> remove URLs      : (no URLs in this one)
      --> remove numbers   : "free entry win a prize call  now!"
      --> remove punct     : "free entry win a prize call  now "
      --> remove stopwords : "free entry win prize call now"
      --> tokenize         : ["free","entry","win","prize","call","now"]
      --> join to string   : "free entry win prize call now"

  PHASE 3: Vectorization
    "free entry win prize call now"
      --> TF-IDF matrix row: [0.0, 0.72, 0.0, 0.61, 0.0, 0.58, ...]
          ^col=free(0.72)        col=win(0.61)    col=prize(0.58)

    Final matrix shape: (5,572 x {len(prod_features):,})
    This matrix is what the model TRAINS on in Phase 4!

  PHASE 4 PREVIEW: Model Training
    X = tfidf_matrix     (shape: 5572 x {len(prod_features):,})
    y = label_array      (shape: 5572, values: 0=ham, 1=spam)

    model.fit(X_train, y_train)
    --> Model learns: rows with high "free", "win", "prize" = spam
    --> Stores this as learned weights/parameters internally
""")

    print(SEP)
    print("  Phase 3 Complete!  Ready for Phase 4: Model Training")
    print(SEP)
