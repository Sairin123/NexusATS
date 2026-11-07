"""End-to-end Pipeline orchestrating Resume Screening, Job Matching, and Candidate Ranking."""
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.parser import DocumentParser
from src.preprocessor import TextPreprocessor
from src.skills_extractor import SkillsExtractor
from src.models.tfidf_matcher import TfidfMatcher
from src.models.embedding_matcher import SentenceTransformerMatcher
from src.models.hybrid_matcher import HybridMatcher
from src.explainability import ScreeningExplainer
from src.ats_checker import ATSChecker


class ResumeScreeningPipeline:
    """Production-ready pipeline for screening resumes, matching jobs, and ranking candidates."""

    def __init__(
        self,
        embedding_model_name: Optional[str] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.preprocessor = TextPreprocessor(remove_stopwords=True, lemmatize=False)
        self.skills_extractor = SkillsExtractor()
        self.tfidf_matcher = TfidfMatcher()

        if embedding_model_name:
            self.embedding_matcher = SentenceTransformerMatcher(model_name=embedding_model_name)
        else:
            self.embedding_matcher = SentenceTransformerMatcher()

        self.hybrid_matcher = HybridMatcher(
            tfidf_matcher=self.tfidf_matcher,
            embedding_matcher=self.embedding_matcher,
            skills_extractor=self.skills_extractor,
            weights=weights,
        )
        self.explainer = ScreeningExplainer()
        self.ats_checker = ATSChecker()

    def _resolve_text(self, source: Union[str, Path]) -> Tuple[str, Optional[str]]:
        """Resolves file path or raw text into string content and optional title."""
        title = None
        if isinstance(source, (str, Path)) and (isinstance(source, Path) or ("\n" not in source and len(source) < 300)):
            p = Path(source)
            if p.exists() and p.is_file():
                if p.suffix.lower() == ".json":
                    data = DocumentParser.parse_json(p)
                    if isinstance(data, dict):
                        title = data.get("title")
                return DocumentParser.load_document(p), title
        return str(source).strip(), None

    def screen_single(
        self,
        resume_source: Union[str, Path],
        job_source: Union[str, Path],
        job_title: Optional[str] = None,
        candidate_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Performs deep screening and gap analysis of a single resume against a job description."""
        start_time = time.time()

        resume_raw, _ = self._resolve_text(resume_source)
        job_raw, detected_title = self._resolve_text(job_source)

        # Parse contact information and sections
        contact_info = DocumentParser.extract_contact_info(resume_raw)
        actual_name = candidate_name or contact_info["name"] or "Candidate"
        actual_job_title = job_title or detected_title or "Target Role"

        sections = self.preprocessor.extract_sections(resume_raw)

        # Clean texts for lexical analysis
        resume_clean = self.preprocessor.clean_text(resume_raw)
        job_clean = self.preprocessor.clean_text(job_raw)

        # Multi-model evaluation
        match_result = self.hybrid_matcher.evaluate_pair(
            resume_raw=resume_raw,
            resume_clean=resume_clean,
            job_raw=job_raw,
            job_clean=job_clean,
            resume_sections=sections,
        )

        # Sentence-level alignment (matching key job bullets to resume bullets)
        job_lines = [line.strip() for line in job_raw.split("\n") if len(line.strip()) > 15]
        resume_lines = [line.strip() for line in resume_raw.split("\n") if len(line.strip()) > 15]
        alignments = self.embedding_matcher.find_best_matching_sentences(
            resume_sentences=resume_lines,
            job_requirements=job_lines[:6],
        )

        # Explainability & Insights
        summary = self.explainer.generate_executive_summary(match_result, actual_name, actual_job_title)
        recommendations = self.explainer.generate_recommendations(match_result)
        categorized_missing = self.explainer.categorize_missing_skills(
            match_result["skill_analysis"]["missing_skills"]
        )

        # ATS Compliance & Readiness Check
        required_job_skills = self.skills_extractor.get_all_skills_flat(job_raw)
        ats_report = self.ats_checker.evaluate_ats_readiness(
            resume_text=resume_raw,
            job_text=job_raw,
            matched_skills=match_result["skill_analysis"]["matched_skills"],
            required_skills=required_job_skills,
            semantic_similarity=match_result["components"]["semantic_similarity"],
            tfidf_similarity=match_result["components"]["tfidf_similarity"],
        )

        elapsed_ms = round((time.time() - start_time) * 1000, 2)

        return {
            "candidate": {
                "name": actual_name,
                "email": contact_info["email"],
                "phone": contact_info["phone"],
                "linkedin": contact_info["linkedin"],
                "github": contact_info["github"],
            },
            "job_title": actual_job_title,
            "overall_score": match_result["overall_score"],
            "match_tier": match_result["match_tier"],
            "badge": match_result["badge"],
            "components": match_result["components"],
            "skill_analysis": match_result["skill_analysis"],
            "categorized_missing_skills": categorized_missing,
            "top_matching_keywords": match_result["top_matching_keywords"],
            "sentence_alignments": alignments,
            "executive_summary": summary,
            "recommendations": recommendations,
            "sections_extracted": {k: bool(v) for k, v in sections.items() if v},
            "ats_report": ats_report,
            "latency_ms": elapsed_ms,
        }

    def screen_batch(
        self,
        resumes: List[Union[str, Path, Dict[str, Any]]],
        job_source: Union[str, Path],
        job_title: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Screens multiple resumes against a job description, ranking them by score."""
        job_raw, detected_title = self._resolve_text(job_source)
        actual_job_title = job_title or detected_title
        results = []

        for item in resumes:
            if isinstance(item, dict):
                r_source = item.get("source") or item.get("text") or item.get("path")
                c_name = item.get("name")
            else:
                r_source = item
                c_name = None

            res = self.screen_single(
                resume_source=r_source,
                job_source=job_raw,
                job_title=actual_job_title,
                candidate_name=c_name,
            )
            results.append(res)

        # Sort descending by overall score
        results.sort(key=lambda x: x["overall_score"], reverse=True)

        # Assign ranks
        for rank, res in enumerate(results, start=1):
            res["rank"] = rank

        return results

    def match_jobs_for_candidate(
        self,
        resume_source: Union[str, Path],
        jobs: List[Union[str, Path, Dict[str, Any]]],
    ) -> List[Dict[str, Any]]:
        """Finds and ranks the best-matching job openings for a given candidate resume."""
        resume_raw, _ = self._resolve_text(resume_source)
        results = []

        for job_item in jobs:
            if isinstance(job_item, dict):
                j_source = job_item.get("description") or job_item.get("text") or job_item.get("path")
                j_title = job_item.get("title", "Open Position")
            else:
                j_source = job_item
                raw_text, detected_title = self._resolve_text(j_source)
                j_title = detected_title or (Path(job_item).stem.replace("_", " ").title() if isinstance(job_item, (str, Path)) else "Role")
                j_source = raw_text

            res = self.screen_single(
                resume_source=resume_raw,
                job_source=j_source,
                job_title=j_title,
            )
            results.append(res)

        results.sort(key=lambda x: x["overall_score"], reverse=True)
        for rank, res in enumerate(results, start=1):
            res["rank"] = rank

        return results

    def benchmark_methods(
        self,
        resume_source: Union[str, Path],
        job_source: Union[str, Path],
    ) -> Dict[str, Any]:
        """Compares TF-IDF, Sentence Transformers, and Hybrid method performance and score differences."""
        resume_raw, _ = self._resolve_text(resume_source)
        job_raw, _ = self._resolve_text(job_source)

        resume_clean = self.preprocessor.clean_text(resume_raw)
        job_clean = self.preprocessor.clean_text(job_raw)

        # 1. TF-IDF benchmark
        t0 = time.time()
        tfidf_score = self.tfidf_matcher.compute_similarity(resume_clean, job_clean)
        tfidf_time = (time.time() - t0) * 1000

        # 2. Sentence Transformers benchmark
        t1 = time.time()
        sbert_score = self.embedding_matcher.compute_similarity(resume_raw, job_raw)
        sbert_time = (time.time() - t1) * 1000

        # 3. Hybrid benchmark
        t2 = time.time()
        hybrid_res = self.hybrid_matcher.evaluate_pair(
            resume_raw=resume_raw,
            resume_clean=resume_clean,
            job_raw=job_raw,
            job_clean=job_clean,
        )
        hybrid_time = (time.time() - t2) * 1000

        return {
            "tfidf": {
                "score_percent": round(tfidf_score * 100, 2),
                "latency_ms": round(tfidf_time, 2),
                "type": "Lexical / N-Gram Vector Space",
                "advantage": "Ultra-fast, captures exact technical acronyms and rare tokens",
            },
            "sentence_transformers": {
                "score_percent": round(sbert_score * 100, 2),
                "latency_ms": round(sbert_time, 2),
                "type": "Dense Neural Semantic Embeddings (SBERT)",
                "advantage": "Captures contextual meaning, synonyms, and conceptual experience",
            },
            "hybrid": {
                "score_percent": hybrid_res["overall_score"],
                "latency_ms": round(hybrid_time + tfidf_time, 2),
                "type": "Multi-Faceted Weighted Ensemble",
                "advantage": "Best of both worlds: balances conceptual alignment with hard skill compliance",
            },
        }
