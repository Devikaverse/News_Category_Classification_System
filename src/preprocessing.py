import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Auto-download NLTK resources if missing
def ensure_nltk_resources():
    resource_map = {
        'stopwords': ['corpora/stopwords', 'corpora/stopwords.zip'],
        'punkt': ['tokenizers/punkt', 'tokenizers/punkt.zip'],
        'punkt_tab': ['tokenizers/punkt_tab', 'tokenizers/punkt_tab.zip'],
        'wordnet': ['corpora/wordnet', 'corpora/wordnet.zip'],
        'omw-1.4': ['corpora/omw-1.4', 'corpora/omw-1.4.zip']
    }
    for pkg, paths in resource_map.items():
        found = False
        for p in paths:
            try:
                nltk.data.find(p)
                found = True
                break
            except LookupError:
                continue
        if not found:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

ensure_nltk_resources()

# Initialize NLP tools
try:
    STOP_WORDS = set(stopwords.words('english'))
except Exception:
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()

def clean_raw_text(text: str) -> str:
    """Removes HTML tags, URLs, special characters, and excessive whitespace."""
    if not isinstance(text, str):
        return ""
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', ' ', text)
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphabetical characters except spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def lemmatize_token(token: str) -> str:
    """Morphological lemmatizer: reduces nouns to singular and verbs to base infinitive."""
    noun_lemma = lemmatizer.lemmatize(token, pos='n')
    return lemmatizer.lemmatize(noun_lemma, pos='v')

def preprocess_text(text: str) -> str:
    """Full preprocessing pipeline: Clean -> Tokenize -> Stopword Removal -> Morphological Lemmatization."""
    if not text or not isinstance(text, str):
        return ""
    
    cleaned = clean_raw_text(text)
    if not cleaned:
        return ""
    
    try:
        tokens = word_tokenize(cleaned)
    except Exception:
        tokens = cleaned.split()
    
    # Filter stopwords and tokens shorter than 2 characters, then lemmatize nouns and verbs
    filtered_tokens = [
        lemmatize_token(token)
        for token in tokens
        if token not in STOP_WORDS and len(token) > 2
    ]
    
    return " ".join(filtered_tokens)

def preprocess_pipeline_steps(text: str) -> dict:
    """Returns detailed intermediate representations for visual debugging."""
    if not text or not isinstance(text, str):
        text = ""
        
    raw_text = text
    
    # 1. True Raw Tokenization (NLTK Tokenizer on original text)
    try:
        raw_tokens = word_tokenize(raw_text)
    except Exception:
        raw_tokens = raw_text.split() if raw_text else []
        
    # 2. Text Normalization & Sanitization
    cleaned_text = clean_raw_text(raw_text)
    try:
        cleaned_tokens = word_tokenize(cleaned_text)
    except Exception:
        cleaned_tokens = cleaned_text.split() if cleaned_text else []
        
    # 3. Stopword & Noise Elimination
    stopwords_removed = [t for t in cleaned_tokens if t not in STOP_WORDS and len(t) > 2]
    removed_stopwords = [t for t in cleaned_tokens if t in STOP_WORDS]
    short_noise = [t for t in cleaned_tokens if t not in STOP_WORDS and len(t) <= 2]
    
    # 4. Morphological Lemmatization (nouns to singular, verbs to base infinitive)
    lemmatized_tokens = [lemmatize_token(t) for t in stopwords_removed]
    
    # Track unique word-level transformations with arrows (before -> after)
    transformations = []
    seen_transformations = set()
    for orig, lem in zip(stopwords_removed, lemmatized_tokens):
        if orig != lem and (orig, lem) not in seen_transformations:
            seen_transformations.add((orig, lem))
            transformations.append({"original": orig, "lemma": lem})
            
    final_output = " ".join(lemmatized_tokens)
    
    raw_char_len = len(raw_text)
    final_char_len = len(final_output)
    reduction_pct = round(((raw_char_len - final_char_len) / raw_char_len * 100), 2) if raw_char_len > 0 else 0
    
    return {
        "raw_text": raw_text,
        "raw_tokens": raw_tokens,
        "cleaned_text": cleaned_text,
        "cleaned_tokens": cleaned_tokens,
        "stopwords_removed": stopwords_removed,
        "removed_stopwords": list(dict.fromkeys(removed_stopwords)),
        "short_noise": list(dict.fromkeys(short_noise)),
        "lemmatized_tokens": lemmatized_tokens,
        "transformations": transformations,
        "final_output": final_output,
        "stats": {
            "raw_word_count": len(raw_text.split()),
            "token_count": len(raw_tokens),
            "cleaned_word_count": len(cleaned_tokens),
            "retained_token_count": len(lemmatized_tokens),
            "removed_stopword_count": len(removed_stopwords) + len(short_noise),
            "lemmatized_count": len(transformations),
            "reduction_percentage": reduction_pct,
            "vocabulary_density": round((len(set(lemmatized_tokens)) / max(len(lemmatized_tokens), 1)) * 100, 2)
        }
    }
