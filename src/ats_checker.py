"""AI-Powered ATS (Applicant Tracking System) Compatibility & Scoring Engine.

Combines Neural Semantic Embeddings (SBERT all-MiniLM-L6-v2), Lexical TF-IDF Vectorization,
Taxonomy Skill Coverage, and Document Parseability into an authentic, highly dynamic ATS compatibility score.
"""
import re
from typing import Any, Dict, List, Optional
from src.parser import DocumentParser
from src.preprocessor import TextPreprocessor


class ATSChecker:
    """Evaluates resume compliance against enterprise Applicant Tracking Systems using NLP and ML."""

    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def evaluate_ats_readiness(
        self,
        resume_text: str,
        job_text: str,
        matched_skills: List[str],
        required_skills: List[str],
        semantic_similarity: float = 0.50,
        tfidf_similarity: float = 0.20,
    ) -> Dict[str, Any]:
        """Calculates dynamic ATS Score (0-100) powered by SBERT, TF-IDF, and structural parsing."""
        clean_res = resume_text.strip()
        words = clean_res.split()
        word_count = len(words)

        # 1. Contact Information Audit
        contacts = DocumentParser.extract_contact_info(clean_res)
        has_email = bool(contacts.get("email"))
        has_phone = bool(contacts.get("phone"))
        has_link = bool(contacts.get("linkedin") or contacts.get("github"))
        contact_points = (1 if has_email else 0) + (1 if has_phone else 0) + (1 if has_link else 0)
        contact_score = round((contact_points / 3) * 100, 1)

        # 2. Section Structure Audit
        sections = self.preprocessor.extract_sections(clean_res)
        found_sections = [k for k, v in sections.items() if v and k != "other"]
        essential_sections = ["experience", "education", "skills"]
        sections_matched = [s for s in essential_sections if s in found_sections]
        has_summary = "summary" in found_sections

        structure_ratio = len(sections_matched) / len(essential_sections)
        if has_summary:
            structure_ratio = min(1.0, structure_ratio + 0.1)
        structure_score = round(min(100.0, structure_ratio * 100), 1)

        # 3. Hard Skills Coverage Ratio (0 - 100)
        if required_skills:
            skill_score = round((len(matched_skills) / len(required_skills)) * 100, 1)
        else:
            # If no taxonomy skills were extracted, estimate based on lexical overlap
            skill_score = round(min(100.0, (tfidf_similarity if tfidf_similarity > 1.0 else tfidf_similarity * 100.0) * 2.0), 1)

        # 4. Quantifiable Metric / Impact Density Audit
        metric_patterns = [
            r"\b\d+%\b",                  # 25%, 100%
            r"\$[\d,]+(?:\.\d+)?[kKmMbB]?", # $150K, $1.2M, $500
            r"\b\d+[kKmMbB]\b",           # 50K, 15M
            r"\b\d+\+\s*(?:years|months|users|clients|projects|engineers|teams)\b",
            r"\b(?:reduced|increased|improved|boosted|optimized|cut|slashed)\s+(?:by\s+)?\d+%",
        ]
        impact_matches = []
        for pat in metric_patterns:
            matches = re.findall(pat, clean_res, flags=re.IGNORECASE)
            impact_matches.extend(matches)

        impact_count = len(impact_matches)
        if impact_count >= 5:
            impact_score = 100.0
        elif impact_count >= 3:
            impact_score = 80.0
        elif impact_count >= 1:
            impact_score = 60.0
        else:
            impact_score = 30.0

        # 5. Length & Word Count Check (optimal: 350 - 950 words)
        if 350 <= word_count <= 950:
            length_score = 100.0
            length_status = "Optimal (1-2 pages)"
        elif 200 <= word_count < 350:
            length_score = 75.0
            length_status = "Slightly Brief"
        elif 950 < word_count <= 1400:
            length_score = 80.0
            length_status = "Slightly Long"
        else:
            length_score = 50.0
            length_status = "Suboptimal Length"

        # Document Format Integrity Score (0 - 100)
        format_integrity = round(
            0.30 * contact_score
            + 0.30 * structure_score
            + 0.20 * impact_score
            + 0.20 * length_score,
            1,
        )

        # Normalize incoming similarities to percentage scale (0 - 100)
        semantic_pct = round(semantic_similarity * 100.0 if semantic_similarity <= 1.0 else semantic_similarity, 1)
        raw_tfidf = tfidf_similarity if tfidf_similarity <= 1.0 else tfidf_similarity / 100.0
        # Scale TF-IDF by 2.5 because unigram/bigram cosine similarity on raw text rarely exceeds 0.40
        tfidf_scaled_pct = round(min(100.0, raw_tfidf * 250.0), 1)

        # Core Machine Learning & NLP Requisition Fit (0 - 100)
        # 50% SBERT Semantic Alignment + 30% Skill Coverage + 20% TF-IDF Keyword Match
        ml_requisition_fit = (
            0.50 * semantic_pct
            + 0.30 * skill_score
            + 0.20 * tfidf_scaled_pct
        )

        # Format Modifier: formats modify score between 0.75x and 1.05x
        format_multiplier = 0.75 + 0.30 * (format_integrity / 100.0)

        # Final Machine Learning Driven ATS Score
        composite_ats = round(max(5.0, min(99.0, ml_requisition_fit * format_multiplier)), 1)

        # Readiness Tier
        if composite_ats >= 72.0:
            readiness_tier = "ATS Optimized (High Pass Probability)"
            tier_badge = "success"
        elif composite_ats >= 48.0:
            readiness_tier = "Moderate Fit (Review Needed)"
            tier_badge = "warning"
        else:
            readiness_tier = "High Risk of ATS Auto-Rejection"
            tier_badge = "danger"

        # Detailed 5-Point Checklist Items
        checklist = [
            {
                "check": "Semantic Requisition Alignment (SBERT Neural Model)",
                "status": "PASS" if semantic_pct >= 65 else ("WARN" if semantic_pct >= 45 else "FAIL"),
                "detail": f"{semantic_pct}% contextual understanding of job requirements",
                "score": semantic_pct,
            },
            {
                "check": "Crucial Requisition Skills Coverage",
                "status": "PASS" if skill_score >= 65 else ("WARN" if skill_score >= 35 else "FAIL"),
                "detail": f"{len(matched_skills)} of {len(required_skills) if required_skills else 'N/A'} required skills matched",
                "score": skill_score,
            },
            {
                "check": "Exact Lexical Keyword Overlap (TF-IDF)",
                "status": "PASS" if tfidf_scaled_pct >= 40 else ("WARN" if tfidf_scaled_pct >= 15 else "FAIL"),
                "detail": f"{tfidf_scaled_pct}% match of exact role terminology & keywords",
                "score": tfidf_scaled_pct,
            },
            {
                "check": "Quantified Achievements & Metrics",
                "status": "PASS" if impact_count >= 3 else ("WARN" if impact_count >= 1 else "FAIL"),
                "detail": f"{impact_count} measurable impact metrics detected (%, $, volume)",
                "score": impact_score,
            },
            {
                "check": "Structure, Contact & Formatting Integrity",
                "status": "PASS" if format_integrity >= 80 else "WARN",
                "detail": f"{word_count} words ({length_status}), {len(found_sections)} standard headings",
                "score": format_integrity,
            },
        ]

        # Actionable Optimization Steps
        recommendations = []
        if semantic_pct < 60.0:
            recommendations.append("Align work experience descriptions more closely with the responsibilities in the job posting.")
        if skill_score < 70.0 and required_skills:
            missing_top = [s for s in required_skills if s not in matched_skills][:4]
            if missing_top:
                recommendations.append(f"Add critical missing requisition keywords: {', '.join([f'`{k}`' for k in missing_top])}.")
        if impact_count < 3:
            recommendations.append("Add measurable performance metrics (e.g. 'reduced latency by 40%', 'managed $2M budget').")
        if not has_phone or not has_email:
            recommendations.append("Ensure both professional email and phone number are clearly visible in the header.")
        if "summary" not in found_sections:
            recommendations.append("Add a concise 3-4 sentence 'Professional Summary' highlighting target role keywords.")

        return {
            "ats_score": composite_ats,
            "readiness_tier": readiness_tier,
            "tier_badge": tier_badge,
            "word_count": word_count,
            "impact_metrics_count": impact_count,
            "checklist": checklist,
            "recommendations": recommendations,
            "found_sections": found_sections,
            "contact_audit": contacts,
            "components": {
                "semantic_sbert": semantic_pct,
                "tfidf_lexical": tfidf_scaled_pct,
                "skills_coverage": skill_score,
                "format_integrity": format_integrity,
            }
        }
