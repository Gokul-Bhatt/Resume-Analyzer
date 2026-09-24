import re
import nltk

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

STOP_WORDS = set(stopwords.words('english'))

def clean_text(text):
    if not text or not text.strip():
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    words = [w for w in text.split() if w not in STOP_WORDS and len(w) > 1]
    return " ".join(words)

def get_matched_missing(resume_text, jd_text):
    resume_words = set(clean_text(resume_text).split())
    jd_words     = set(clean_text(jd_text).split())
    matched = sorted(resume_words & jd_words)
    missing = sorted(jd_words - resume_words)
    return matched, missing
