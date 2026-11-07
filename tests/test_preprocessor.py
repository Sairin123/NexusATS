"""Unit tests for text preprocessing and section segmentation."""
import pytest
from src.preprocessor import TextPreprocessor


def test_text_cleaning_basic():
    preprocessor = TextPreprocessor(remove_stopwords=True)
    raw = "Senior Software Engineer with experience in Python and Machine Learning! Contact: john@test.com"
    clean = preprocessor.clean_text(raw)

    assert "python" in clean
    assert "machine" in clean
    assert "learning" in clean
    # Email should be stripped
    assert "john@test.com" not in clean


def test_protected_technical_tokens():
    preprocessor = TextPreprocessor(remove_stopwords=True)
    raw = "Skills include C++, C#, .NET core, and CI/CD automation pipelines."
    clean = preprocessor.clean_text(raw)

    assert "c++" in clean
    assert "c#" in clean
    assert ".net" in clean
    assert "ci/cd" in clean


def test_extract_sections():
    preprocessor = TextPreprocessor()
    resume_text = """
John Doe
Software Engineer

SUMMARY
Passionate developer building scalable systems.

EXPERIENCE
Senior Engineer at Tech Corp
- Built backend services in Python.

EDUCATION
B.S. in Computer Science, MIT

SKILLS
Python, Docker, Kubernetes
"""
    sections = preprocessor.extract_sections(resume_text)

    assert "Passionate developer" in sections["summary"]
    assert "Tech Corp" in sections["experience"]
    assert "MIT" in sections["education"]
    assert "Kubernetes" in sections["skills"]

