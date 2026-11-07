"""Unit tests for TF-IDF matcher and Cosine Similarity."""
import pytest
from src.models.tfidf_matcher import TfidfMatcher


def test_tfidf_identical_texts():
    matcher = TfidfMatcher()
    text = "python machine learning deep learning neural networks"
    sim = matcher.compute_similarity(text, text)
    assert pytest.approx(sim, 0.01) == 1.0


def test_tfidf_orthogonal_texts():
    matcher = TfidfMatcher()
    t1 = "react javascript css html frontend"
    t2 = "kubernetes docker terraform cloud infrastructure"
    sim = matcher.compute_similarity(t1, t2)
    assert sim == 0.0


def test_tfidf_batch_similarity():
    matcher = TfidfMatcher()
    job = "python machine learning engineer"
    r1 = "python machine learning specialist"
    r2 = "frontend react developer"

    scores = matcher.compute_batch_similarity([r1, r2], job)
    assert len(scores) == 2
    assert scores[0] > scores[1]


def test_top_matching_keywords():
    matcher = TfidfMatcher()
    resume = "expert python developer with deep learning and pytorch"
    job = "looking for python developer with deep learning"

    keywords = matcher.extract_top_matching_keywords(resume, job, top_n=6)
    kw_names = [k["keyword"] for k in keywords]

    assert any("python" in k for k in kw_names)
    assert any("learning" in k for k in kw_names)
    assert len(keywords) > 0
