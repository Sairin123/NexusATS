"""Unit tests for Sentence Transformers embeddings and semantic cosine similarity."""
import numpy as np
import pytest
from src.models.embedding_matcher import SentenceTransformerMatcher


def test_embedding_dimensions():
    matcher = SentenceTransformerMatcher()
    emb = matcher.encode("Natural language processing with transformers.")
    assert emb.shape == (1, 384)
    # Norm of normalized vector should be approx 1.0
    norm = np.linalg.norm(emb[0])
    assert pytest.approx(norm, 0.01) == 1.0


def test_semantic_similarity_paraphrase():
    matcher = SentenceTransformerMatcher()
    s1 = "Developed deep neural networks for natural language understanding."
    s2 = "Built deep learning architectures for NLP tasks."
    s3 = "Cooked delicious pasta and baked chocolate chip cookies."

    sim_related = matcher.compute_similarity(s1, s2)
    sim_unrelated = matcher.compute_similarity(s1, s3)

    assert sim_related > 0.65
    assert sim_unrelated < 0.35
    assert sim_related > sim_unrelated


def test_best_matching_sentences():
    matcher = SentenceTransformerMatcher()
    res_sentences = [
        "Architected enterprise search engine with Sentence Transformers and Pinecone.",
        "Managed team sprint retrospectives and agile planning.",
    ]
    job_reqs = [
        "Experience building semantic vector search applications.",
    ]

    alignments = matcher.find_best_matching_sentences(res_sentences, job_reqs)
    assert len(alignments) == 1
    assert alignments[0]["is_match"] is True
    assert "Sentence Transformers" in alignments[0]["best_matching_resume_sentence"]

