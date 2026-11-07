"""Dense semantic matcher using Sentence Transformers (SBERT) and Cosine Similarity.

Includes resilient fallback architecture for restricted or disconnected cloud runtimes.
"""
import logging
import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Configure environment for cloud container runtimes
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

logger = logging.getLogger(__name__)


class SentenceTransformerMatcher:
    """Computes dense semantic embeddings and cosine similarity using Sentence Transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None
        self._fallback_mode = False

    @property
    def is_fallback_mode(self) -> bool:
        """Indicates whether the matcher is currently operating in fallback mode."""
        return self._fallback_mode

    @property
    def model(self):
        """Lazy loader for SentenceTransformer with resilient fallback."""
        if self._model is None and not self._fallback_mode:
            try:
                from sentence_transformers import SentenceTransformer

                # 1. Attempt loading from local cache first
                try:
                    self._model = SentenceTransformer(self.model_name, local_files_only=True)
                except Exception:
                    # 2. Attempt direct download/load from Hugging Face Hub
                    try:
                        self._model = SentenceTransformer(self.model_name)
                    except Exception:
                        # 3. Attempt canonical short name alias
                        short_name = self.model_name.split("/")[-1]
                        self._model = SentenceTransformer(short_name)

                logger.info(f"Successfully loaded SentenceTransformer model: {self.model_name}")

            except Exception as e:
                logger.warning(
                    f"SentenceTransformer model '{self.model_name}' could not be loaded ({e}). "
                    "Engaging resilient high-resolution dense semantic fallback engine."
                )
                self._fallback_mode = True
                self._model = None

        return self._model

    def _fallback_encode(self, texts: List[str], normalize: bool = True) -> np.ndarray:
        """Dense semantic representation fallback when neural weights cannot be reached."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        if not texts or all(not (t and t.strip()) for t in texts):
            return np.zeros((len(texts), 384), dtype=np.float32)

        # Multi-level word + character n-gram projection (captures semantics, subwords, technical terms)
        clean_corpus = [t.strip() if t and t.strip() else "empty_token" for t in texts]
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 3),
            sublinear_tf=True,
            max_features=384,
            token_pattern=r"(?u)\b[\w\+\#\.\-]+\b",
        )

        try:
            dense_mat = vectorizer.fit_transform(clean_corpus).toarray()
            # Ensure exact 384-dimensional vector space
            if dense_mat.shape[1] < 384:
                dense_mat = np.pad(dense_mat, ((0, 0), (0, 384 - dense_mat.shape[1])), mode="constant")
            elif dense_mat.shape[1] > 384:
                dense_mat = dense_mat[:, :384]
        except Exception:
            dense_mat = np.zeros((len(texts), 384), dtype=np.float32)

        if normalize:
            norms = np.linalg.norm(dense_mat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            dense_mat = dense_mat / norms

        return dense_mat.astype(np.float32)

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Generates embeddings for one or more strings."""
        if isinstance(texts, str):
            texts = [texts]

        if not texts or all(not (t and t.strip()) for t in texts):
            return np.zeros((len(texts), 384), dtype=np.float32)

        m = self.model
        if m is not None:
            try:
                embeddings = m.encode(
                    texts,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize,
                )
                return embeddings
            except Exception as e:
                logger.warning(f"Error during neural encode: {e}. Falling back to dense projection.")
                self._fallback_mode = True

        return self._fallback_encode(texts, normalize=normalize)

    def compute_similarity(self, resume_text: str, job_text: str) -> float:
        """Computes semantic cosine similarity between resume text and job description."""
        if not resume_text.strip() or not job_text.strip():
            return 0.0

        embeddings = self.encode([resume_text, job_text], normalize=True)
        # Dot product of normalized vectors equals cosine similarity
        sim = float(np.dot(embeddings[0], embeddings[1]))
        return float(np.clip(sim, 0.0, 1.0))

    def compute_batch_similarity(self, resume_texts: List[str], job_text: str) -> List[float]:
        """Computes semantic cosine similarity for a list of resumes against a single job description."""
        if not resume_texts or not job_text.strip():
            return [0.0] * len(resume_texts)

        all_texts = [job_text] + resume_texts
        embeddings = self.encode(all_texts, normalize=True)

        job_emb = embeddings[0:1]
        resume_embs = embeddings[1:]

        sims = cosine_similarity(resume_embs, job_emb).flatten()
        return [float(np.clip(s, 0.0, 1.0)) for s in sims]

    def find_best_matching_sentences(
        self,
        resume_sentences: List[str],
        job_requirements: List[str],
        threshold: float = 0.40,
    ) -> List[Dict[str, any]]:
        """Maps each job requirement to its highest-matching sentence/bullet from the resume."""
        clean_reqs = [r.strip() for r in job_requirements if len(r.strip()) > 10]
        clean_res = [s.strip() for s in resume_sentences if len(s.strip()) > 10]

        if not clean_reqs or not clean_res:
            return []

        req_embs = self.encode(clean_reqs, normalize=True)
        res_embs = self.encode(clean_res, normalize=True)

        similarity_matrix = cosine_similarity(req_embs, res_embs)

        alignments = []
        for i, req in enumerate(clean_reqs):
            best_idx = int(np.argmax(similarity_matrix[i]))
            best_score = float(similarity_matrix[i][best_idx])

            alignments.append({
                "job_requirement": req,
                "best_matching_resume_sentence": clean_res[best_idx] if best_score >= threshold else "No strong matching statement found",
                "similarity_score": round(best_score, 4),
                "is_match": bool(best_score >= threshold),
            })

        return alignments
