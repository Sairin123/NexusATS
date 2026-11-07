"""Explainability and gap analysis generator for candidate resume screening."""
from typing import Any, Dict, List, Optional
from src.skills_extractor import SKILL_TO_CATEGORY


class ScreeningExplainer:
    """Provides actionable feedback, skill gap diagnostics, and recommendations for recruiters and job seekers."""

    @staticmethod
    def generate_executive_summary(match_result: Dict[str, Any], candidate_name: str, job_title: str) -> str:
        """Produces a concise recruiter executive summary."""
        score = match_result["overall_score"]
        tier = match_result["match_tier"]
        matched_count = match_result["skill_analysis"]["matched_count"]
        req_count = match_result["skill_analysis"]["required_count"]
        semantic = match_result["components"]["semantic_similarity"]

        if tier == "Strong Match":
            return (
                f"**{candidate_name}** is a **{tier}** ({score}%) for the position of **{job_title}**. "
                f"Demonstrates strong conceptual alignment (Semantic Score: {semantic}%) and covers "
                f"{matched_count} of {req_count} essential requirements."
            )
        elif tier == "Moderate Match":
            return (
                f"**{candidate_name}** demonstrates **{tier}** ({score}%) for **{job_title}**. "
                f"Foundational competencies match {matched_count}/{req_count} required skills, "
                f"with minor technical gaps that can be addressed via targeted onboarding."
            )
        else:
            return (
                f"**{candidate_name}** has a **{tier}** ({score}%) for **{job_title}**. "
                f"Significant alignment gaps detected with only {matched_count}/{req_count} required domain skills."
            )

    @staticmethod
    def generate_recommendations(match_result: Dict[str, Any]) -> List[str]:
        """Generates specific, constructive recommendations for candidates to boost match rate."""
        recommendations = []
        missing_skills = match_result["skill_analysis"].get("missing_skills", [])
        components = match_result["components"]

        # Skill gap specific advice
        if missing_skills:
            # Group missing skills by category
            grouped_missing: Dict[str, List[str]] = {}
            for s in missing_skills:
                cat = SKILL_TO_CATEGORY.get(s, "general")
                grouped_missing.setdefault(cat, []).append(s)

            for cat, skills in grouped_missing.items():
                cat_name = cat.replace("_", " ").title()
                skill_list = ", ".join([f"`{s}`" for s in skills[:4]])
                recommendations.append(
                    f"**Acquire {cat_name} Competencies**: Target {skill_list}. Showcase these through projects or open-source work."
                )

        # TF-IDF Lexical advice
        if components["tfidf_similarity"] < 35.0:
            recommendations.append(
                "**Keyword & Phrasing Optimization**: Incorporate standard industry terminology and action verbs matching the job description."
            )

        # Experience advice
        if components["experience_relevance"] < 50.0:
            recommendations.append(
                "**Quantify Work Impact**: Frame experience with measurable outcomes (e.g. 'Optimized model inference latency by 35%')."
            )

        if not recommendations:
            recommendations.append(
                "**Profile Excellence**: Resume is well-aligned with the target role. Highlight leadership and architectural achievements in the interview."
            )

        return recommendations

    @staticmethod
    def categorize_missing_skills(missing_skills: List[str]) -> Dict[str, List[str]]:
        """Groups missing skills into domains (AI, Web, Cloud, DB, etc.)."""
        categorized: Dict[str, List[str]] = {}
        for s in missing_skills:
            cat = SKILL_TO_CATEGORY.get(s, "other").replace("_", " ").title()
            categorized.setdefault(cat, []).append(s)
        return categorized

