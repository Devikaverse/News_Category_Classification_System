# News Category Classification System

[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B.svg?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![Python](https://img.shields.io/badge/Python-3.9+-3776AB.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-3.8+-154f5b.svg?style=for-the-badge)](https://www.nltk.org/)
[![Plotly](https://img.shields.io/badge/Plotly-5.18+-3F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)](https://plotly.com/)
[![Linear SVM Accuracy](https://img.shields.io/badge/Linear%20SVM-98.36%25-7c3aed.svg?style=for-the-badge)](https://github.com)
[![Naive Bayes Accuracy](https://img.shields.io/badge/Naive%20Bayes-97.65%25-2563eb.svg?style=for-the-badge)](https://github.com)

A machine learning and natural language processing (NLP) application built with Python, scikit-learn, and Streamlit to classify news articles into five categories: **Business**, **Entertainment**, **Politics**, **Sports**, and **Technology**.

---

## Table of Contents

- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Model Performance & Benchmarks](#-model-performance--benchmarks)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Local Setup](#-installation--local-setup)
- [Deployment Guide](#-deployment-guide)
  - [Option 1: Streamlit Community Cloud (Recommended)](#option-1-streamlit-community-cloud-recommended)
  - [Option 2: Render / Heroku (Using Procfile)](#option-2-render--heroku-using-procfile)
  - [Option 3: Hugging Face Spaces](#option-3-hugging-face-spaces)
- [Application Modules Walkthrough](#-application-modules-walkthrough)
  - [1. Live Article Classifier](#1-live-article-classifier)
  - [2. Bulk Batch CSV Processing](#2-bulk-batch-csv-processing)
  - [3. Dataset & Model Analytics](#3-dataset--model-analytics)
  - [4. NLP Pipeline Debugger](#4-nlp-pipeline-debugger)
  - [5. Architecture & Methodology](#5-architecture--methodology)
- [Handling Regional & International Content](#-handling-regional--international-content)
- [Technology Stack](#-technology-stack)
- [Configuration & Retraining](#-configuration--retraining)
- [Troubleshooting](#-troubleshooting)
- [License](#-license)

---

## Key Features

- **Real-Time Single-Article Inference**: Instant classification with confidence percentages, sub-millisecond latency monitoring, and probabilistic distributions across all 5 categories.
- **Uncertainty & Out-of-Domain Guardrails**: Protects against gibberish, non-news text, or unseen topics by flagging them as `Uncertain / Other` (via calibrated thresholding and out-of-vocabulary detection).
- **Explainable AI (XAI) Feature Attribution**: Dynamically extracts and displays the top 12 TF-IDF weighted terms driving the model's classification for any article.
- **Local CSV Article Filtering**: Automatically classify uploaded articles with the locally trained model, display all articles in a selected category, and download the filtered results without connecting to an online AI service.
- **Interactive NLP Pipeline Debugger**: Trace text transformation across 5 distinct stages (Regex Sanitization -> NLTK Tokenization -> Stopword Filtration -> WordNet Lemmatization -> TF-IDF Vector Projection).
- **Executive Analytics Dashboard**: Inspect model performance across Linear SVM and Naive Bayes, interactive Plotly confusion matrices (with row-normalization), and per-category precision/recall/F1 metrics.
- **Intuitive Architecture Explanations**: Accessible, plain-English conceptual analogies (*"Finding the Widest Highway"* for SVM vs. *"Multiplying Word Odds"* for Naive Bayes) with zero visual clutter.
- **Custom Design System**: Bespoke enterprise theme in `src/styles.css` with Plus Jakarta Sans typography, dark SaaS sidebar, responsive cards, and zero distracting emojis.

---

## System Architecture

```
Raw Input Text
      │
      ▼
┌────────────────────────────────────────────────────────┐
│               NLP Preprocessing Engine                 │
│  • HTML strip, lowercase, regex symbol removal         │
│  • Tokenization (NLTK punkt)                           │
│  • Stopword Removal (NLTK stopwords, tokens > 2 chars) │
│  • Lemmatization (NLTK WordNetLemmatizer)              │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│              TF-IDF Feature Extractor                  │
│  • 5,000 max features (sublinear TF scaling)           │
│  • N-gram range: (1, 2) unigrams & bigrams             │
│  • Minimum document frequency: min_df = 2              │
└────────────────────────────────────────────────────────┘
      │
      ▼
┌────────────────────────────────────────────────────────┐
│            Calibrated Inference Classifiers            │
│  • Calibrated Linear Support Vector Machine (SVC)      │
│  • Multinomial Naive Bayes (Laplace alpha = 0.1)       │
└────────────────────────────────────────────────────────┘
      │
      ├── Prob < Threshold / Out of Vocab ──► [ Uncertain / Other ]
      └── High Confidence Classification ───► [ Business | Entertainment |
                                                Politics | Sports | Technology ]
```

---

## Model Performance & Benchmarks

The classification models are trained on the standard **BBC News Corpus** (2,225 total articles). Before training, 98 exact duplicate articles were dropped to prevent train/test data leakage, leaving 2,127 deduplicated articles evaluated under a stratified 80/20 train/test split (1,701 train / 426 test):

| Evaluation Metric | Calibrated Linear SVM | Multinomial Naive Bayes | Advantage |
| :--- | :---: | :---: | :--- |
| **Model Paradigm** | Discriminative (Hyperplane Boundary) | Generative (Word Likelihood) | Linear SVM optimizes separation margin |
| **Test Accuracy** | **98.36%** | **97.65%** | Linear SVM (+0.71%) |
| **Weighted Precision** | **98.37%** | **97.70%** | Linear SVM (+0.67%) |
| **Weighted Recall** | **98.36%** | **97.65%** | Linear SVM (+0.71%) |
| **Weighted F1-Score** | **0.9835** | **0.9766** | Linear SVM (+0.0069) |
| **Training Duration** | ~1.20s | **~0.15s** | Naive Bayes (~8x faster training) |
| **Average Inference** | **< 2.0 ms / doc** | **< 1.0 ms / doc** | Both execute with sub-millisecond latency |
| **Primary Production Role** | **Selected Default Classifier** | **Fast Baseline & Dual Verification** | SVM offers highest boundary fidelity |

### Per-Class Performance Breakdown (Linear SVM on Held-Out Test Set)

| Category | Precision | Recall | F1-Score | Support (Test Docs) |
| :--- | :---: | :---: | :---: | :---: |
| **Business** | 98.98% | 96.04% | 0.9749 | 101 |
| **Entertainment** | 97.33% | 98.65% | 0.9799 | 74 |
| **Politics** | 98.77% | 98.77% | 0.9877 | 81 |
| **Sports** | 99.01% | 99.01% | 0.9901 | 101 |
| **Technology** | 97.18% | 100.00% | 0.9857 | 69 |

---

## Project Directory Structure

```text
news-category-classification/
│
├── .streamlit/
│   └── config.toml           # Streamlit theme configuration (light mode, #2563eb primary)
│
├── models/                   # Serialized model artifacts & evaluation metrics
│   ├── nb_model.joblib       # Serialized Multinomial Naive Bayes model
│   ├── svm_model.joblib      # Serialized Calibrated Linear SVM model
│   ├── vectorizer.joblib     # Fitted TF-IDF Vectorizer (5,000 vocabulary)
│   ├── metrics.json          # Complete evaluation metrics, confusion matrices, and corpus stats
│   └── processed_dataset.csv # Preprocessed corpus sample for dashboard stats
│
├── src/                      # Source modules
│   ├── __init__.py           # Package marker
│   ├── model.py              # Model loading, training pipeline, and inference logic
│   ├── preprocessing.py      # NLP pipeline (cleaning, tokenization, lemmatization)
│   └── styles.css            # Custom UI styling and responsive typography tokens
│
├── .gitignore                # Git ignore rules for Python, virtual environments, and caches
├── Procfile                  # Process file for Render / Heroku cloud deployment
├── app.py                    # Main Streamlit web application entry point
├── news_dataset.csv          # Raw BBC news dataset (2,225 articles)
├── requirements.txt          # Python dependencies specification
└── README.md                 # Complete project documentation
```

---

## Installation & Local Setup

### 1. Prerequisites
- **Python 3.9, 3.10, or 3.11** installed.
- Git (optional, for version control).

### 2. Clone or Download
```bash
git clone https://github.com/your-username/news-category-classification.git
cd news-category-classification
```

### 3. Set Up Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

> **Automated NLTK Setup**: The application automatically checks and downloads necessary NLTK packages (`stopwords`, `punkt`, `punkt_tab`, `wordnet`, `omw-1.4`) upon first startup.

### 5. Launch the Web Application
```bash
streamlit run app.py
```
Open your browser and navigate to:
```
http://localhost:8501
```

---

## Deployment Guide

The project is structured and pre-configured for seamless 1-click cloud deployment.

### Option 1: Streamlit Community Cloud (Recommended)
1. Push your repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/<your-username>/news-category-classification.git
   git push -u origin main
   ```
2. Log into [share.streamlit.io](https://share.streamlit.io/).
3. Click **"New app"** and select your repository:
   - **Repository**: `<your-username>/news-category-classification`
   - **Branch**: `main`
   - **Main file path**: `app.py`
4. Click **"Deploy!"**. Streamlit Cloud will install `requirements.txt` and launch the app in seconds.

### Option 2: Render / Heroku (Using Procfile)
The included `Procfile` specifies:
```text
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```
1. Create a new Web Service on [Render](https://render.com/) or Heroku.
2. Connect your GitHub repository.
3. Build command: `pip install -r requirements.txt`
4. Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`

### Option 3: Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces).
2. Choose **Streamlit** as the Space SDK.
3. Push the repository contents to the Space repository.

---

## Application Modules Walkthrough

### 1. Live Article Classifier
- **Single Article Inference**:
  - Select one of 5 pre-configured editorial presets (Business, Technology, Sports, Politics, Entertainment) or paste any custom article text.
  - Generates:
    - Target Category badge and color indicator.
    - Confidence percentage and inference latency (in milliseconds).
    - Top 12 discriminative TF-IDF keywords that influenced the classification.
    - Horizontal probability distribution chart across all 5 classes.
- **Uncertainty Guard**:
  - Rejects out-of-domain text, random characters, or low-confidence samples (< 45%), marking them as `Uncertain / Other` rather than returning a misleading classification.

### 2. Bulk Batch CSV Processing
- Upload a CSV file containing news articles. Classification uses the model trained from the included `news_dataset.csv`; no online AI service is required.
- Select the column to classify.
- The uploaded articles are automatically classified into Business, Entertainment, Politics, Sports, Technology, or Uncertain / Other.
- Filter the displayed articles by predicted category, or show all categories.
- Download the displayed category results as a CSV.

### 3. Dataset & Model Analytics
- Executive KPI cards for Linear SVM Accuracy (98.20%), Naive Bayes Accuracy (97.53%), Total Corpus Size (2,225 articles), and TF-IDF Vocabulary (5,000 terms).
- Side-by-side grouped bar charts comparing Accuracy, Precision, Recall, and F1-Score.
- Category balance distribution charts across the BBC corpus.
- Interactive Plotly Confusion Matrix with raw counts and normalized true-class percentages.
- Complete per-class classification report table.

### 4. NLP Pipeline Debugger
- Step through the 5 internal transformation stages of the NLP pipeline in real-time:
  1. **Stage 1: Normalization & Sanitization** (HTML/URL removal, lowercasing, regex filtering).
  2. **Stage 2: Tokenization** (NLTK word tokenization into discrete lexical tokens).
  3. **Stage 3: Stopword Elimination** (Removal of high-frequency non-informative English stopwords).
  4. **Stage 4: Morphological Lemmatization** (WordNet root lemma extraction).
  5. **Stage 5: High-Dimensional TF-IDF Vector Projection** (Matched vocabulary indices and relative importance weights).
- Lexical metrics panel tracking raw word count, token reduction percentage, and vocabulary density.

### 5. Architecture & Methodology
- **End-to-End Workflow**: Interactive Mermaid diagram of the ingestion, preprocessing, vectorization, and inference architecture.
- **Core Intuition Explanations**:
  - **Linear SVM**: Explained as *"Finding the Widest Highway"* across a 5,000-dimensional space, focusing on borderline support vectors.
  - **Multinomial Naive Bayes**: Explained as *"Multiplying Word Odds"* with Laplace (+1) smoothing to guarantee mathematical stability on rare words.
- **Executive Comparison Matrix**: Comprehensive evaluation breakdown contrasting model paradigms, decision rules, speed, and real-world deployment tradeoffs.
- **Taxonomy Specifications**: Full category breakdown of the BBC News corpus with sample discriminative keywords.

---

## Handling Regional & International Content

The models were trained on the benchmark **BBC News Corpus** (UK editorial coverage from 2004–2005). When classifying contemporary international political news (such as Indian political events featuring names like *Delhi*, *Mumbai*, *DMK*, *CJP*):

- **Linear SVM Behavior**: SVM evaluates combinations of high-weight terms. If words like *"pressure"* or *"resignation"* have stronger weights in business articles within the historical UK corpus, SVM may lean toward Business if the political entities (*Delhi*, *DMK*) are outside the 2004 UK vocabulary.
- **Naive Bayes Behavior**: Multinomial Naive Bayes tallies individual word probabilities. Because political terms like *"protest"*, *"electoral"*, and *"opposition"* appear frequently in political articles, Naive Bayes often correctly flags the piece as **Politics** (e.g., 70.7% confidence).
- **Dual-Model Verification**: The application allows instant switching between **Support Vector Machine (Linear SVM)** and **Multinomial Naive Bayes** from the sidebar so users can verify how both discriminative and generative paradigms interpret challenging texts.

---

## Technology Stack

| Layer | Component | Description |
| :--- | :--- | :--- |
| **User Interface** | [Streamlit](https://streamlit.io/) | Interactive dashboard framework |
| **Typography & Theme** | [Vanilla CSS](src/styles.css) | Custom styling, Plus Jakarta Sans, dark SaaS sidebar |
| **NLP Pipeline** | [NLTK](https://www.nltk.org/) | Tokenization, English stopwords, WordNet Lemmatizer |
| **Machine Learning** | [Scikit-Learn](https://scikit-learn.org/) | `LinearSVC`, `MultinomialNB`, `TfidfVectorizer`, `CalibratedClassifierCV` |
| **Data Processing** | [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/) | Data manipulation and numerical operations |
| **Visualizations** | [Plotly](https://plotly.com/) | Interactive charts, heatmaps, and distribution plots |
| **Serialization** | [Joblib](https://joblib.readthedocs.io/) | Serialization of trained models and vectorizer pipelines |

---

## Configuration & Retraining

### Re-training the Pipeline
To re-train both models and serialize updated artifacts:
```python
from src.model import train_and_save_pipeline

# Trains models, computes evaluation metrics, and saves joblib artifacts to models/
metrics = train_and_save_pipeline(
    csv_path="news_dataset.csv",
    max_features=5000
)
```

### Tuning Uncertainty Threshold
The default confidence threshold for out-of-domain detection is configured in [`src/model.py`](src/model.py):
```python
DEFAULT_CONFIDENCE_THRESHOLD = 0.45  # 45% minimum class probability
```
Inputs whose highest predicted probability falls below this threshold are tagged as `Uncertain / Other`.

---

## Troubleshooting

<details>
<summary><b>1. NLTK Resource Download Errors</b></summary>

If your environment has restricted internet access, manually download NLTK data before running:
```python
import nltk
nltk.download(['stopwords', 'punkt', 'punkt_tab', 'wordnet', 'omw-1.4'])
```
</details>

<details>
<summary><b>2. Port Already in Use</b></summary>

If port 8501 is occupied, launch Streamlit on an alternate port:
```bash
streamlit run app.py --server.port 8502
```
</details>

<details>
<summary><b>3. Missing Model Artifacts</b></summary>

If `.joblib` files are not found in `models/`, the application will automatically run `train_and_save_pipeline()` on first launch using `news_dataset.csv`.
</details>

---

## License

This project is licensed under the [MIT License](LICENSE).
