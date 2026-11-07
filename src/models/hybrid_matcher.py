"""Hybrid matching engine combining dense semantic embeddings, TF-IDF lexical matching, and skill coverage."""
from typing import Dict, List, Optional
import numpy as np

from src.config import DEFAULT_WEIGHTS, SCORE_THRESHOLDS
from src.models.tfidf_matcher import TfidfMatcher
from src.models.embedding_matcher import SentenceTransformerMatcher
from src.skills_extractor import SkillsExtractor


class HybridMatcher:
    """Ensemble matcher combining semantic embeddings, TF-IDF lexical similarity, and structured skill coverage."""

    def __init__(
        self,
        tfidf_matcher: Optional[TfidfMatcher] = None,
        embedding_matcher: Optional[SentenceTransformerMatcher] = None,
        skills_extractor: Optional[SkillsExtractor] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.tfidf = tfidf_matcher or TfidfMatcher()
        self.embedder = embedding_matcher or SentenceTransformerMatcher()
        self.skills_extractor = skills_extractor or SkillsExtractor()

        raw_weights = weights or DEFAULT_WEIGHTS
        # Normalize weights to sum to 1.0
        total = sum(raw_weights.values())
        self.weights = {k: v / total for k, v in raw_weights.items()}

    def evaluate_pair(
        self,
        resume_raw: str,
        resume_clean: str,
        job_raw: str,
        job_clean: str,
        resume_sections: Optional[Dict[str, str]] = None,
    ) -> Dict[str, any]:
        """Calculates individual component scores and final weighted hybrid matching score."""
        # 1. Semantic Cosine Similarity (using raw / full text to preserve contextual semantics)
        semantic_sim = self.embedder.compute_similarity(resume_raw, job_raw)

        # 2. Lexical TF-IDF Cosine Similarity (using preprocessed tokens)
        tfidf_sim = self.tfidf.compute_similarity(resume_clean, job_clean)

        # 3. Skill Overlap & Coverage Analysis
        skill_analysis = self.skills_extractor.calculate_skill_overlap(resume_raw, job_raw)
        skill_coverage = skill_analysis["coverage_score"]

        # 4. Experience / Section Specific Similarity
        exp_text = ""
        if resume_sections:
            exp_text = (
                resume_sections.get("experience", "")
                + " "
                + resume_sections.get("projects", "")
            ).strip()

        if exp_text:
            exp_sim = self.embedder.compute_similarity(exp_text, job_raw)
        else:
            exp_sim = semantic_sim

        # 5. Composite Hybrid Score
        hybrid_score = (
            self.weights["semantic"] * semantic_sim
            + self.weights["tfidf"] * tfidf_sim
            + self.weights["skills"] * skill_coverage
            + self.weights["experience"] * exp_sim
        )
        hybrid_score = float(np.clip(hybrid_score, 0.0, 1.0))

        # 6. Determine Match Tier
        if hybrid_score >= SCORE_THRESHOLDS["strong_match"]:
            tier = "Strong Match"
            badge = "success"
        elif hybrid_score >= SCORE_THRESHOLDS["moderate_match"]:
            tier = "Moderate Match"
            badge = "warning"
        else:
            tier = "Weak Match"
            badge = "danger"

        # 7. Extract Top Overlapping Keywords
        top_keywords = self.tfidf.extract_top_matching_keywords(resume_clean, job_clean, top_n=8)

        return {
            "overall_score": round(hybrid_score * 100, 2),  # In percentage 0-100%
            "match_tier": tier,
            "badge": badge,
            "components": {
                "semantic_similarity": round(semantic_sim * 100, 2),
                "tfidf_similarity": round(tfidf_sim * 100, 2),
                "skill_coverage": round(skill_coverage * 100, 2),
                "experience_relevance": round(exp_sim * 100, 2),
            },
            "weights_used": self.weights,
            "skill_analysis": skill_analysis,
            "top_matching_keywords": top_keywords,
        }

