"""Unit tests for ATS Checker and scoring system."""
import pytest
from src.ats_checker import ATSChecker


def test_ats_readiness_evaluation():
    checker = ATSChecker()
    resume_text = """
Jane Doe
jane.doe@email.com | +1 (555) 123-4567 | linkedin.com/in/janedoe

SUMMARY
Experienced software engineer with 6+ years building scalable systems.

EXPERIENCE
Senior Developer at Acme Corp
- Optimized system latency by 45%, processing 10M+ transactions daily.
- Managed a $1.5M budget and led a team of 8 engineers.

EDUCATION
B.S. in Computer Science

SKILLS
Python, Docker, Kubernetes, AWS, SQL
"""
    job_text = "Looking for software engineer with Python, Docker, Kubernetes, AWS, and CI/CD."
    matched_skills = ["python", "docker", "kubernetes", "aws"]
    required_skills = ["python", "docker", "kubernetes", "aws", "ci/cd"]

    report = checker.evaluate_ats_readiness(
        resume_text=resume_text,
        job_text=job_text,
        matched_skills=matched_skills,
        required_skills=required_skills,
        semantic_similarity=0.82,
        tfidf_similarity=0.40,
    )

    assert report["ats_score"] > 70.0
    assert report["impact_metrics_count"] >= 3
    assert len(report["checklist"]) == 5
    assert report["readiness_tier"] != ""

