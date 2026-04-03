# 🛡️ Spam Detection with Machine Learning

> An end-to-end spam classifier built with Python, scikit-learn, FastAPI, and a slick frontend UI — designed not just to *work* but to *teach* you how every piece fits together.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.7-orange?style=flat-square&logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/FastAPI-0.135-green?style=flat-square&logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-purple?style=flat-square)

---

## 📌 Project Overview

This project builds a **binary text classifier** that detects whether an SMS message is **spam** or **ham** (legitimate). It covers the full ML pipeline — from raw data to a live web API with a frontend.

| Label | Meaning | Value |
|-------|---------|-------|
| Ham | Legitimate message | 0 |
| Spam | Unwanted/junk message | 1 |

**Key focus areas:**
- Understanding *why* each step is done, not just *what*
- Deep-diving into model evaluation (confusion matrix, precision, recall, F1)
- Analyzing the impact of class imbalance on model selection

---

## 📁 Project Structure

```
spam_detection/
│
├── README.md                    # You are here
├── requirements.txt             # Python dependencies
│
├── data/
│   └── spam.csv                 # SMS Spam Collection Dataset (~5,500 messages)
│
├── src/                         # Core ML logic
│   ├── __init__.py
│   ├── explore.py               # Phase 1: Data exploration & visualization
│   ├── preprocessing.py         # Phase 2: Text cleaning pipeline
│   ├── vectorizer.py            # Phase 3: TF-IDF transformation
│   ├── train.py                 # Phase 4: Model training (NB, LR, SVM)
│   └── evaluate.py              # Phase 5: Metrics & confusion matrix
│
├── api/
│   └── main.py                  # Phase 6: FastAPI /predict endpoint
│
├── frontend/
│   ├── index.html               # Phase 7: UI
│   ├── style.css
│   └── script.js
│
├── models/                      # Saved trained models (generated)
│   ├── naive_bayes.pkl
│   ├── logistic_regression.pkl
│   ├── svm.pkl
│   └── tfidf_vectorizer.pkl
│
└── outputs/                     # Charts & visualizations (generated)
    └── phase1_exploration.png
```

---

## 🧠 End-to-End Pipeline

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Raw Text    │ →  │ Preprocessed │ →  │  Numerical   │ →  │   Trained    │
│  Messages    │    │    Text      │    │   Vectors    │    │    Model     │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
   spam.csv          lowercase,          TF-IDF              NB, LR, SVM
                     no stopwords        vectors
```

---

## 🚀 8 Phases Explained

| Phase | Name | What Happens |
|-------|------|-------------|
| 1 | Data Exploration | Load data, check distribution, visualize |
| 2 | Preprocessing | Clean text (lowercase, remove noise, stopwords) |
| 3 | Vectorization | Convert text → TF-IDF numeric vectors |
| 4 | Model Training | Train Naive Bayes, Logistic Regression, SVM |
| 5 | Evaluation | Confusion matrix, precision, recall, F1 |
| 6 | API Backend | FastAPI `/predict` endpoint |
| 7 | Frontend | HTML/CSS/JS user interface |
| 8 | Integration | Connect frontend ↔ backend, end-to-end test |

---

## 📊 Dataset

- **Source:** [SMS Spam Collection — Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** ~5,572 messages
- **Class Distribution:** ~87% Ham, ~13% Spam (imbalanced!)

> ⚠️ **Why imbalance matters:** A model that always predicts "ham" would be 87% accurate — but catches ZERO spam. This is why we use **Precision, Recall, and F1** instead of accuracy alone.

---

## 🔬 Models Used

| Model | Why Good for Text? |
|-------|-------------------|
| **Naive Bayes** | Fast, works great with word frequencies |
| **Logistic Regression** | Interpretable weights, solid baseline |
| **SVM** | Works well with high-dimensional data |

---

## 📐 Evaluation Metrics — Confusion Matrix

```
                    Predicted
                 Spam    Ham
Actual  Spam  [  TP  |  FN  ]   ← Missing spam is dangerous!
        Ham   [  FP  |  TN  ]   ← Blocking ham is annoying
```

| Metric | Formula | What It Tells You |
|--------|---------|-------------------|
| **Accuracy** | (TP+TN) / Total | Overall correctness (misleading on imbalanced data!) |
| **Precision** | TP / (TP+FP) | Of predicted spam, how many are actually spam? |
| **Recall** | TP / (TP+FN) | Of actual spam, how many did we catch? |
| **F1 Score** | 2×(P×R)/(P+R) | Harmonic mean — best single metric for imbalanced data |

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10+ |
| ML Library | scikit-learn |
| Data Handling | pandas, numpy |
| Text Processing | NLTK |
| API Backend | FastAPI + Uvicorn |
| Frontend | HTML + CSS + JavaScript |
| Model Storage | joblib |
| Visualization | matplotlib, seaborn |

---

## 🛠️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/jabssyyy/spam_detection.git
cd spam_detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK stopwords (first time only)

```python
python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"
```

---

## ▶️ How to Run

### Phase 1 — Explore the data
```bash
python -m src.explore
```
> Outputs a 4-panel chart to `outputs/phase1_exploration.png`

### Phase 4 — Train all models
```bash
python -m src.train
```
> Saves trained models to `models/`

### Phase 6 — Start the API server
```bash
uvicorn api.main:app --reload
```
> API available at `http://localhost:8000`
> Auto-docs at `http://localhost:8000/docs`

### Phase 7 — Open the frontend
```
Open frontend/index.html in your browser
```

---

## 🌐 API Usage

**POST** `/predict`

```json
{
  "message": "Congratulations! You've won a FREE prize. Call now!"
}
```

**Response:**
```json
{
  "prediction": "spam",
  "confidence": 0.97,
  "model": "naive_bayes"
}
```

---

## 📈 Phase 1 Results (Data Exploration)

| Stat | Value |
|------|-------|
| Total Messages | 5,572 |
| Ham Messages | 4,825 (86.6%) |
| Spam Messages | 747 (13.4%) |
| Avg Spam Length | 139 characters |
| Avg Ham Length | 71 characters |
| Missing Values | 0 |

**Key Finding:** Spam messages are nearly **2× longer** than ham and contain distinctive vocabulary: `"free"`, `"win"`, `"claim"`, `"prize"`, `"call now"`, etc.

---

## 💡 Key Concepts to Understand

### Why TF-IDF over Bag of Words?
- **BoW** just counts word frequency
- **TF-IDF** weights words by *importance* — rare words get higher scores
- The word `"free"` appearing in spam a lot gets a high TF-IDF weight

### Why Stemming?
- `"running"`, `"runs"`, `"runner"` → all become `"run"`
- Reduces vocabulary size, helps the model generalize

### Why NOT just use accuracy?
- With 87% ham, predicting "ham" always = 87% accurate but 0% useful
- **F1 Score** penalizes models that ignore the minority class (spam)

---

## 📝 Learning Notes

This project is structured as a **learning journey**. Every file is heavily commented with:
- `WHY` we do each step
- `WHAT` goes in and comes out
- Real examples from the dataset
- Connections to ML theory

---

## 🤝 Contributing

Pull requests are welcome! If you find a bug or want to add a new model, open an issue first.

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

*Built as a learning project to deeply understand spam detection, text classification, and model evaluation with confusion matrices.*
