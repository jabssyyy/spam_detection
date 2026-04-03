# 📩 Spam Detection with Machine Learning

An end-to-end SMS spam detection system built from scratch as a learning project.
Covers data exploration, text preprocessing, TF-IDF vectorization, model training,
evaluation with confusion matrices, a REST API backend, and a modern frontend UI.

---

## 🚀 What This Project Does

- Classifies any SMS message as **SPAM** or **HAM** (not spam) in real time
- Trains three ML models: Naive Bayes, Logistic Regression, and SVM
- Deep-dives into model evaluation — confusion matrix, precision, recall, F1
- Serves predictions via a **FastAPI** REST API
- Provides a **modern browser UI** to interact with the classifier

---

## 🛠 Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| ML / NLP | scikit-learn, NLTK, NumPy, pandas |
| Vectorization | TF-IDF (`TfidfVectorizer`) |
| Models | MultinomialNB, LogisticRegression, LinearSVC |
| API | FastAPI + Uvicorn |
| Serialization | joblib |
| Visualization | Matplotlib, Seaborn |
| Frontend | HTML5, CSS3, Vanilla JS |

---

## 📁 Project Structure

```
spam_detection/
│
├── data/
│   └── spam.csv                   # SMS Spam Collection dataset (5,572 messages)
│
├── src/
│   ├── __init__.py
│   ├── explore.py                 # Phase 1: Data exploration & visualization
│   ├── preprocessing.py           # Phase 2: Text cleaning pipeline
│   ├── vectorizer.py              # Phase 3: TF-IDF vectorization
│   ├── train.py                   # Phase 4: Model training (NB, LR, SVM)
│   └── evaluate.py                # Phase 5: Confusion matrix & metrics
│
├── api/
│   ├── __init__.py
│   └── main.py                    # Phase 6: FastAPI backend
│
├── frontend/
│   ├── index.html                 # Phase 7: UI structure
│   ├── style.css                  # Styling (clean, professional light mode)
│   ├── about.html                 # Phase 7: About the Models page (metrics)
│   └── script.js                  # API calls, result rendering
│
├── models/                        # Saved trained models (auto-created)
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   ├── tfidf_vectorizer.pkl
│   └── split_indices.pkl
│
├── outputs/                       # Generated visualizations
│   ├── phase1_exploration.png
│   └── phase5_evaluation.png
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/jabssyyy/spam_detection.git
cd spam_detection

# 2. Create and activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download NLTK stopwords (one-time)
python -c "import nltk; nltk.download('stopwords')"
```

---

## ▶️ How to Run

### Step 1 — Train the models (if `models/` is empty)

```bash
python -m src.train
```

This will:
- Load and preprocess all 5,572 messages
- Build the TF-IDF vocabulary on training data only
- Train Naive Bayes, Logistic Regression, and SVM
- Save all models + vectorizer to `models/`

### Step 2 — Start the API server

```bash
uvicorn api.main:app --reload --port 8000
```

Server starts at: `http://127.0.0.1:8000`  
Interactive docs: `http://127.0.0.1:8000/docs`

### Step 3 — Open the frontend

The frontend UI is served directly by the FastAPI server!
Simply open your browser and visit:

**http://127.0.0.1:8000/ui**

---

## 🔌 API Documentation

### `GET /`
Health check — confirms server and models are ready.

```json
{
  "status": "healthy",
  "models_loaded": ["naive_bayes", "logistic_regression", "svm"],
  "default_model": "svm",
  "uptime_seconds": 42.1
}
```

### `POST /predict`
Classify a single message.

**Request:**
```json
{
  "text": "FREE ENTRY! You have WON a prize. Call NOW!",
  "model": "svm"
}
```

**Response:**
```json
{
  "prediction": "spam",
  "label": 1,
  "confidence": 0.9928,
  "spam_prob": 0.9928,
  "ham_prob": 0.0072,
  "model_used": "svm",
  "pipeline": {
    "raw_text": "FREE ENTRY! ...",
    "cleaned_text": "free entry prize call",
    "token_count": 4
  }
}
```

### `POST /predict/batch`
Classify multiple messages in one call.

**Request:**
```json
{
  "texts": ["Free prize call now!", "See you tomorrow"],
  "model": "svm"
}
```

### `GET /predict/compare?text=<message>`
Run the same message through all 3 models simultaneously.

### `GET /models`
List all available models.

---

## 📊 Model Performance

All models evaluated on a held-out test set (20% of data, 1,115 messages).
The same 80/20 stratified split is used for all comparisons.

| Model | Accuracy | Precision | Recall | F1 Score | Missed Spam (FN) |
|-------|----------|-----------|--------|----------|-----------------|
| Naive Bayes | 98.21% | 99.24% | 87.25% | 92.86% | 19 |
| Logistic Regression | 96.86% | 98.31% | 77.85% | 86.89% | 33 |
| **SVM ⭐** | **98.65%** | 97.18% | **92.62%** | **94.85%** | **11** |

### Confusion Matrix — SVM (Best Model)

```
                  Predicted HAM   Predicted SPAM
Actual HAM  (0)       962               4          ← 4 false alarms
Actual SPAM (1)        11             138          ← 11 missed spams
```

> **Why Recall matters most for spam:**
> A missed spam (FN) reaches the inbox — dangerous.
> A false alarm (FP) blocks a legitimate message — simply annoying.
> SVM has the highest recall (92.62%) = catches the most spam.

---

## 🧠 ML Pipeline — How It Works

```
Raw SMS Text
    │
    ▼  Phase 2: Preprocessing
lowercase → remove URLs → remove numbers → remove punctuation
    → remove stopwords → tokenize
    │
    ▼  Phase 3: TF-IDF Vectorization
Each message → sparse vector of 5,000 TF-IDF weighted features
(fit vocabulary on TRAINING data only — no data leakage!)
    │
    ▼  Phase 4: Model Training
Naive Bayes  → models word probability per class
Logistic Reg → learns weights for each word
SVM          → finds max-margin hyperplane in 5000-D space
    │
    ▼  Phase 5: Evaluation
Confusion matrix, precision, recall, F1 — for all 3 models
    │
    ▼  Phase 6–7: API + Frontend
FastAPI serves predictions at /predict
Browser UI calls the API and renders results
```

---

## 📈 Integration Test Results (Phase 8)

17 / 17 tests passed ✅

| Group | Tests | Result |
|-------|-------|--------|
| Spam messages | 6 | 6/6 PASS |
| Ham messages | 6 | 6/6 PASS |
| Edge cases | 5 | 5/5 PASS |

---

## 💡 Key Concepts Learned

| Concept | Where used |
|---------|-----------|
| Class imbalance | Phase 1 (87% ham, 13% spam) |
| Text preprocessing | Phase 2 (NLTK stopwords, regex) |
| TF-IDF | Phase 3 (why it beats Bag of Words) |
| Train-test split (stratified) | Phase 4 |
| Confusion matrix | Phase 5 (TP, TN, FP, FN) |
| Precision vs Recall trade-off | Phase 5 |
| REST API design | Phase 6 (FastAPI, Pydantic) |
| CORS | Phase 6 (browser security) |
| async/await + fetch | Phase 7 (JavaScript) |
| Data leakage | Phase 4 (fit vectorizer on train only) |

---

## 🔮 Future Improvements

- [ ] Add stemming / lemmatization (Porter Stemmer)
- [ ] Try deep learning: LSTM or DistilBERT for context-aware detection
- [ ] Add confidence threshold for "uncertain" predictions
- [ ] Add user feedback loop (mark predictions as wrong → retrain)
- [ ] Deploy to cloud (Render / Railway / HuggingFace Spaces)
- [ ] Add email header analysis (sender, subject line features)

---

## 📄 Dataset

**SMS Spam Collection v.1**  
Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
Size: 5,572 messages (4,825 ham + 747 spam)

---

*Built as a hands-on ML learning project — Phase 1 through Phase 8*
