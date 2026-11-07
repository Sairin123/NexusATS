"""Dense semantic matcher using Sentence Transformers (SBERT) and Cosine Similarity."""
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SentenceTransformerMatcher:
    """Computes dense semantic embeddings and cosine similarity using Sentence Transformers."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    @property
    def model(self):
        """Lazy loader for SentenceTransformer to optimize startup time."""
        if self._model is None:
            import os
            from sentence_transformers import SentenceTransformer
            # Try loading from local cache first to prevent network retry timeouts in offline/sandboxed environments
            try:
                self._model = SentenceTransformer(self.model_name, local_files_only=True)
            except Exception:
                try:
                    self._model = SentenceTransformer(self.model_name)
                except Exception as e:
                    raise RuntimeError(f"Failed to load embedding model {self.model_name}: {e}")
        return self._model

    def encode(self, texts: Union[str, List[str]], normalize: bool = True) -> np.ndarray:
        """Generates embeddings for one or more strings."""
        if isinstance(texts, str):
            texts = [texts]

        if not texts or all(not t.strip() for t in texts):
            # Return zero vector with expected dimension (384 for MiniLM)
            return np.zeros((len(texts), 384), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=normalize,
        )
        return embeddings

    def compute_similarity(self, resume_text: str, job_text: str) -> float:
        """Computes semantic cosine similarity between resume text and job description."""
        if not resume_text.strip() or not job_text.strip():
            return 0.0

        embeddings = self.encode([resume_text, job_text], normalize=True)
        # With normalized vectors, dot product equals cosine similarity
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
