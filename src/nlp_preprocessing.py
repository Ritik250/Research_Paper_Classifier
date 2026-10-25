"""NLTK-based preprocessing for academic abstracts.

No NLTK corpora need to be downloaded: tokenization and Snowball stemming use
NLTK's code, while a stable English stop-word set comes from scikit-learn.
"""
import re
from nltk.stem import SnowballStemmer
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_STEMMER = SnowballStemmer("english")
_STOP_WORDS = set(ENGLISH_STOP_WORDS)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b")

def nltk_tokenize_and_stem(text):
    """Return normalized word tokens, excluding stopwords and bare numbers."""
    text = _URL_RE.sub(" ", str(text))
    text = _EMAIL_RE.sub(" ", text)
    tokens = []
    for token in wordpunct_tokenize(text.lower()):
        token = token.strip("_-'’")
        if not token or len(token) < 2 or token in _STOP_WORDS:
            continue
        if token.isdigit():
            continue
        if not any(ch.isalpha() for ch in token):
            continue
        tokens.append(_STEMMER.stem(token))
    return tokens

def preprocess_texts(texts):
    """Preprocess a batch of documents into whitespace-separated normalized tokens."""
    return [" ".join(nltk_tokenize_and_stem(text)) for text in texts]
