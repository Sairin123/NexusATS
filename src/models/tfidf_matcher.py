"""TF-IDF lexical matcher using n-gram vectorization and Cosine Similarity."""
from typing import Dict, List, Optional, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TfidfMatcher:
    """Computes lexical matching scores and keyword importance between resumes and job descriptions using TF-IDF and Cosine Similarity."""

    def __init__(
        self,
        ngram_range: Tuple[int, int] = (1, 2),
        max_features: Optional[int] = 5000,
        sublinear_tf: bool = True,
    ):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.sublinear_tf = sublinear_tf

    def compute_similarity(self, resume_text: str, job_text: str) -> float:
        """Computes pairwise cosine similarity between preprocessed resume and job description."""
        if not resume_text.strip() or not job_text.strip():
            return 0.0

        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_text])
            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            return float(np.clip(similarity, 0.0, 1.0))
        except ValueError:
            # Can happen if text contains only unknown words or stopwords
            return 0.0

    def compute_batch_similarity(self, resume_texts: List[str], job_text: str) -> List[float]:
        """Computes cosine similarity for multiple resumes against a single job description."""
        if not resume_texts or not job_text.strip():
            return [0.0] * len(resume_texts)

        corpus = [job_text] + resume_texts
        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(corpus)
            job_vector = tfidf_matrix[0:1]
            resume_vectors = tfidf_matrix[1:]

            similarities = cosine_similarity(resume_vectors, job_vector).flatten()
            return [float(np.clip(s, 0.0, 1.0)) for s in similarities]
        except ValueError:
            return [0.0] * len(resume_texts)

    def extract_top_matching_keywords(
        self, resume_text: str, job_text: str, top_n: int = 10
    ) -> List[Dict[str, float]]:
        """Identifies top contributing shared keywords based on element-wise TF-IDF product."""
        if not resume_text.strip() or not job_text.strip():
            return []

        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            sublinear_tf=self.sublinear_tf,
        )

        try:
            matrix = vectorizer.fit_transform([resume_text, job_text]).toarray()
            feature_names = np.array(vectorizer.get_feature_names_out())

            # Element-wise product of resume and job vector weights
            importance_scores = matrix[0] * matrix[1]

            # Filter non-zero overlaps
            non_zero_indices = np.where(importance_scores > 0)[0]
            if len(non_zero_indices) == 0:
                return []

            # Sort by score descending
            sorted_indices = non_zero_indices[np.argsort(importance_scores[non_zero_indices])[::-1]]

            top_keywords = []
            for idx in sorted_indices[:top_n]:
                top_keywords.append({
                    "keyword": str(feature_names[idx]),
                    "importance_score": round(float(importance_scores[idx]), 4),
                    "resume_weight": round(float(matrix[0][idx]), 4),
                    "job_weight": round(float(matrix[1][idx]), 4),
                })
            return top_keywords
        except ValueError:
            return []

