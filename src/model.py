import os
import json
import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report

from src.preprocessing import preprocess_text

CATEGORY_MAPPING = {
    'business': 'Business',
    'entertainment': 'Entertainment',
    'politics': 'Politics',
    'sport': 'Sports',
    'tech': 'Technology'
}

UNCERTAIN_LABEL = 'Uncertain / Other'
DEFAULT_CONFIDENCE_THRESHOLD = 0.45

CATEGORY_ICONS = {
    UNCERTAIN_LABEL: '',
    'Business': '',
    'Entertainment': '',
    'Politics': '',
    'Sports': '',
    'Technology': ''
}

CATEGORY_COLORS = {
    UNCERTAIN_LABEL: '#f59e0b',  # Amber (rejected / out-of-domain)
    'Business': '#3b82f6',       # Blue
    'Entertainment': '#ec4899',  # Pink
    'Politics': '#eab308',       # Amber
    'Sports': '#10b981',         # Emerald
    'Technology': '#8b5cf6'      # Purple
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DEFAULT_DATASET_PATH = os.path.join(BASE_DIR, 'news_dataset.csv')

def train_and_save_pipeline(csv_path: str = DEFAULT_DATASET_PATH, max_features: int = 5000):
    """Loads dataset, trains MultinomialNB and Calibrated LinearSVC, evaluates and saves artifacts."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    print(f"[+] Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Standardize column names
    text_col = 'text' if 'text' in df.columns else df.columns[0]
    cat_col = 'category' if 'category' in df.columns else ('Category' if 'Category' in df.columns else df.columns[1])
    
    df = df[[text_col, cat_col]].dropna()
    df['clean_category'] = df[cat_col].astype(str).str.lower().str.strip().map(
        lambda c: CATEGORY_MAPPING.get(c, c.capitalize())
    )
    
    # Drop exact duplicate articles BEFORE splitting to avoid train/test leakage
    before = len(df)
    df = df.drop_duplicates(subset=[text_col]).reset_index(drop=True)
    print(f"[+] Removed {before - len(df)} duplicate articles (leakage prevention)")

    print(f"[+] Preprocessing {len(df)} documents with NLP pipeline...")
    start_prep = time.time()
    df['processed_text'] = df[text_col].apply(preprocess_text)
    print(f"[OK] Preprocessing finished in {time.time() - start_prep:.2f}s")
    
    # Remove empty processed strings if any
    df = df[df['processed_text'].str.strip().str.len() > 0].copy()
    
    X = df['processed_text']
    y = df['clean_category']
    categories = sorted(list(y.unique()))
    
    # Stratified Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"[+] Extracting TF-IDF features (max_features={max_features}, ngram_range=(1, 2))...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=max_features,
        sublinear_tf=True,
        min_df=2
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Model 1: Multinomial Naive Bayes
    print("[+] Training Multinomial Naive Bayes...")
    nb_model = MultinomialNB(alpha=0.1)
    nb_model.fit(X_train_tfidf, y_train)
    nb_pred = nb_model.predict(X_test_tfidf)
    nb_probs = nb_model.predict_proba(X_test_tfidf)
    
    # Model 2: Linear SVM (Calibrated for probability estimates)
    print("[+] Training Calibrated Linear Support Vector Machine...")
    base_svm = LinearSVC(C=1.0, random_state=42, max_iter=2000)
    svm_model = CalibratedClassifierCV(estimator=base_svm, method='sigmoid', cv=3)
    svm_model.fit(X_train_tfidf, y_train)
    svm_pred = svm_model.predict(X_test_tfidf)
    svm_probs = svm_model.predict_proba(X_test_tfidf)
    
    # Evaluation Metrics Computation
    def evaluate(y_true, y_pred, y_prob):
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred, labels=categories).tolist()
        report = classification_report(y_true, y_pred, labels=categories, output_dict=True)
        return {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "confusion_matrix": cm,
            "classification_report": report
        }
    
    metrics = {
        "dataset_info": {
            "raw_samples": before,
            "deduplicated_samples": len(df),
            "duplicates_removed": before - len(df),
            "dedup_rate_pct": round(((before - len(df)) / before) * 100, 2) if before else 0,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "feature_count": len(vectorizer.get_feature_names_out()),
            "categories": categories,
            "category_distribution": df['clean_category'].value_counts().to_dict()
        },
        "categories": categories,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "total_samples": before,
        "feature_count": len(vectorizer.get_feature_names_out()),
        "Multinomial Naive Bayes": evaluate(y_test, nb_pred, nb_probs),
        "Support Vector Machine (Linear SVM)": evaluate(y_test, svm_pred, svm_probs)
    }
    
    # Save artifacts
    print(f"[+] Serializing artifacts into {MODEL_DIR}...")
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, "vectorizer.joblib"))
    joblib.dump(nb_model, os.path.join(MODEL_DIR, "nb_model.joblib"))
    joblib.dump(svm_model, os.path.join(MODEL_DIR, "svm_model.joblib"))
    
    with open(os.path.join(MODEL_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        
    # Save preprocessed sample data for dashboard analytics
    sample_df = df[[text_col, 'clean_category', 'processed_text']].copy()
    sample_df.to_csv(os.path.join(MODEL_DIR, "processed_dataset.csv"), index=False, encoding="utf-8")
    
    print("[OK] Pipeline trained and artifacts serialized successfully!")
    return metrics

def load_artifacts():
    """Loads vectorizer, models, and metadata metrics."""
    vec_path = os.path.join(MODEL_DIR, "vectorizer.joblib")
    nb_path = os.path.join(MODEL_DIR, "nb_model.joblib")
    svm_path = os.path.join(MODEL_DIR, "svm_model.joblib")
    metrics_path = os.path.join(MODEL_DIR, "metrics.json")
    
    if not (os.path.exists(vec_path) and os.path.exists(nb_path) and os.path.exists(svm_path)):
        print("[!] Artifacts not found. Initiating model training...")
        train_and_save_pipeline()
        
    vectorizer = joblib.load(vec_path)
    nb_model = joblib.load(nb_path)
    svm_model = joblib.load(svm_path)
    
    metrics = {}
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
            
    return vectorizer, {"Multinomial Naive Bayes": nb_model, "Support Vector Machine (Linear SVM)": svm_model}, metrics

def predict_single_article(text: str, model_name: str, vectorizer, models_dict,
                           confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
    """End-to-end inference on one article. Rejects low-confidence / out-of-vocabulary
    inputs as 'Uncertain / Other' instead of forcing one of the five categories."""
    start_time = time.perf_counter()

    model = models_dict.get(model_name)
    if not model:
        raise ValueError(f"Unknown model: {model_name}")

    # 1. Preprocessing
    processed = preprocess_text(text)
    if not processed:
        return {
            "error": "The input text does not contain any valid vocabulary words after NLP filtering."
        }

    # 2. Vectorization
    tfidf_vec = vectorizer.transform([processed])

    # 3. Model inference & probabilities
    classes = list(model.classes_)
    probs = model.predict_proba(tfidf_vec)[0]

    pred_idx = int(np.argmax(probs))
    raw_category = classes[pred_idx]
    confidence_score = float(probs[pred_idx])

    # 4. Uncertainty / out-of-domain detection
    is_uncertain = False
    uncertain_reason = None
    if tfidf_vec.nnz == 0:
        is_uncertain = True
        uncertain_reason = "None of the words in this text appear in the model's vocabulary."
    elif confidence_score < confidence_threshold:
        is_uncertain = True
        uncertain_reason = (
            f"Highest class probability ({confidence_score * 100:.1f}%) is below the "
            f"{confidence_threshold * 100:.0f}% acceptance threshold."
        )

    final_category = UNCERTAIN_LABEL if is_uncertain else raw_category

    prob_distribution = {
        cls: round(float(prob) * 100, 2)
        for cls, prob in zip(classes, probs)
    }
    prob_distribution = dict(sorted(prob_distribution.items(), key=lambda item: item[1], reverse=True))

    # 5. Top TF-IDF terms of the input document
    feature_names = np.array(vectorizer.get_feature_names_out())
    row = tfidf_vec.tocoo()
    sorted_items = sorted(zip(row.col, row.data), key=lambda x: x[1], reverse=True)
    top_keywords = [
        {"word": feature_names[col_idx], "tfidf_score": round(float(weight), 4)}
        for col_idx, weight in sorted_items[:12]
    ]

    latency_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "category": final_category,
        "raw_category": raw_category,
        "is_uncertain": is_uncertain,
        "uncertain_reason": uncertain_reason,
        "icon": CATEGORY_ICONS.get(final_category, ""),
        "color": CATEGORY_COLORS.get(final_category, "#3b82f6"),
        "confidence": round(confidence_score * 100, 2),
        "latency_ms": latency_ms,
        "probabilities": prob_distribution,
        "top_keywords": top_keywords,
        "processed_text": processed,
        "raw_text": text
    }

def batch_predict(df: pd.DataFrame, text_column: str, model_name: str, vectorizer, models_dict,
                  confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD):
    """Batch inference. Empty cells and low-confidence rows are flagged 'Uncertain / Other'."""
    model = models_dict.get(model_name)
    start_time = time.perf_counter()

    if len(df) == 0:
        res_df = df.copy()
        res_df['Predicted_Category'] = []
        res_df['Raw_Top_Category'] = []
        res_df['Confidence_Score'] = []
        return res_df, {
            "total_count": 0,
            "total_time_s": 0.0,
            "avg_latency_ms": 0.0,
            "uncertain_count": 0,
            "category_distribution": {}
        }

    # Treat NaN / blank cells as empty text (not the literal string "nan")
    raw_texts = df[text_column].fillna("").astype(str)
    clean_texts = raw_texts.apply(preprocess_text)
    empty_mask = clean_texts.str.strip().str.len() == 0

    tfidf_matrix = vectorizer.transform(clean_texts)
    probs = model.predict_proba(tfidf_matrix)
    classes = np.array(model.classes_)
    raw_preds = classes[np.argmax(probs, axis=1)]
    confidences = np.max(probs, axis=1)

    uncertain_mask = (confidences < confidence_threshold) | empty_mask.values
    final_preds = np.where(uncertain_mask, UNCERTAIN_LABEL, raw_preds)

    total_time = time.perf_counter() - start_time
    avg_latency = round((total_time / max(len(df), 1)) * 1000, 2)

    res_df = df.copy()
    res_df['Predicted_Category'] = final_preds
    res_df['Raw_Top_Category'] = raw_preds
    res_df['Confidence_Score'] = [round(float(c) * 100, 2) for c in confidences]

    return res_df, {
        "total_count": len(df),
        "total_time_s": round(total_time, 2),
        "avg_latency_ms": avg_latency,
        "uncertain_count": int(uncertain_mask.sum()),
        "category_distribution": res_df['Predicted_Category'].value_counts().to_dict()
    }
