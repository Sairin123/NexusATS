"""Unit tests for skills extraction and taxonomy matching."""
import pytest
from src.skills_extractor import SkillsExtractor


def test_extract_skills_categories():
    extractor = SkillsExtractor()
    sample_text = "Proficient in Python, PyTorch, Docker, Kubernetes, React, and PostgreSQL."

    skills = extractor.extract_skills(sample_text)

    assert "python" in skills["machine_learning_ai"] or "python" in skills["programming_languages"]
    assert "pytorch" in skills["machine_learning_ai"]
    assert "docker" in skills["cloud_devops"]
    assert "kubernetes" in skills["cloud_devops"]
    assert "react" in skills["web_frameworks"]
    assert "postgresql" in skills["databases_data_eng"]


def test_skill_overlap_calculation():
    extractor = SkillsExtractor()
    resume_text = "Experienced in Python, PyTorch, Docker, and AWS."
    job_text = "Requirements: Python, PyTorch, Kubernetes, and AWS."

    overlap = extractor.calculate_skill_overlap(resume_text, job_text)

    assert "python" in overlap["matched_skills"]
    assert "pytorch" in overlap["matched_skills"]
    assert "aws" in overlap["matched_skills"]
    assert "kubernetes" in overlap["missing_skills"]
    assert overlap["required_count"] == 4
    assert overlap["matched_count"] == 3
    assert overlap["coverage_score"] == 0.75

