from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from utils import clean_text, get_matched_missing


def calculate_ats_score(resume_text, jd_text):
    """
    WHY WE USE CountVectorizer INSTEAD OF TfidfVectorizer:
    -------------------------------------------------------
    TF-IDF calculates IDF (Inverse Document Frequency) across ALL documents.
    When you only have 2 documents (resume + JD), any word that appears in
    BOTH gets IDF = log(2/2) = 0 — meaning shared words score ZERO.
    This makes cosine similarity return 0.0% even when many words match.

    CountVectorizer simply counts word frequencies (no IDF penalty),
    so shared words correctly contribute to the similarity score.
    This gives accurate, meaningful ATS scores.
    """

    # Step 1: Clean both texts
    resume_clean = clean_text(resume_text)
    jd_clean     = clean_text(jd_text)

    if not resume_clean.strip() or not jd_clean.strip():
        return 0.0, [], []

    # Step 2: Convert to word count vectors (CountVectorizer = TF without IDF)
    vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
    vectors    = vectorizer.fit_transform([resume_clean, jd_clean])

    # Step 3: Cosine similarity between resume vector and JD vector
    similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]

    # Step 4: Convert to percentage
    score = round(similarity * 100, 1)

    # Step 5: Get matched and missing keywords
    matched, missing = get_matched_missing(resume_text, jd_text)

    return score, matched, missing
