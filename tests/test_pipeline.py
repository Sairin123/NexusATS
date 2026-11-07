"""End-to-end integration tests for the Resume Screening Pipeline."""
from pathlib import Path
import pytest

from src.config import JOBS_DIR, RESUMES_DIR
from src.pipeline import ResumeScreeningPipeline


@pytest.fixture(scope="module")
def pipeline():
    return ResumeScreeningPipeline()


def test_pipeline_single_screening(pipeline):
    resume_path = RESUMES_DIR / "resume_ml_engineer_alice.txt"
    job_path = JOBS_DIR / "job_senior_ml_engineer.json"

    result = pipeline.screen_single(resume_path, job_path)

    assert result["candidate"]["name"] == "Alice Chen"
    assert result["overall_score"] > 60.0
    assert result["components"]["semantic_similarity"] > 70.0
    assert result["skill_analysis"]["matched_count"] > 10
    assert len(result["recommendations"]) > 0


def test_pipeline_batch_screening_ranking(pipeline):
    job_path = JOBS_DIR / "job_senior_ml_engineer.json"
    resumes = [
        RESUMES_DIR / "resume_ml_engineer_alice.txt",
        RESUMES_DIR / "resume_frontend_bob.txt",
    ]

    ranked = pipeline.screen_batch(resumes, job_path)

    assert len(ranked) == 2
    # Alice (ML) should be ranked higher than Bob (Frontend) for an ML Engineer job
    assert ranked[0]["candidate"]["name"] == "Alice Chen"
    assert ranked[0]["rank"] == 1
    assert ranked[1]["candidate"]["name"] == "Bob Miller"
    assert ranked[1]["rank"] == 2
    assert ranked[0]["overall_score"] > ranked[1]["overall_score"]


def test_pipeline_job_recommendation(pipeline):
    resume_path = RESUMES_DIR / "resume_frontend_bob.txt"
    jobs = [
        JOBS_DIR / "job_senior_ml_engineer.json",
        JOBS_DIR / "job_frontend_engineer.json",
    ]

    recommended = pipeline.match_jobs_for_candidate(resume_path, jobs)

    # Bob should have Frontend Engineer ranked #1
    assert recommended[0]["rank"] == 1
    assert "Frontend" in recommended[0]["job_title"]
    assert recommended[0]["overall_score"] > recommended[1]["overall_score"]

