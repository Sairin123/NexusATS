"""Document parser supporting TXT, PDF, DOCX, and JSON files for resumes and job descriptions."""
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Union


class DocumentParser:
    """Parses text, PDF, DOCX, and JSON files and extracts metadata."""

    @staticmethod
    def parse_txt(file_path: Union[str, Path]) -> str:
        """Reads plain text file."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()

    @staticmethod
    def parse_pdf(file_path: Union[str, Path]) -> str:
        """Extracts text from PDF file using pypdf."""
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path))
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text_parts.append(page_text)
            return "\n".join(text_parts).strip()
        except ImportError:
            # Fallback reading with raw stream decode if pypdf is not available
            with open(file_path, "rb") as f:
                content = f.read()
                # Basic ASCII/UTF-8 string extraction
                text_chars = re.findall(rb"[\x20-\x7E\r\n\t]{4,}", content)
                return "\n".join(t.decode("latin1", errors="ignore") for t in text_chars)

    @staticmethod
    def parse_docx(file_path: Union[str, Path]) -> str:
        """Extracts text from DOCX file using python-docx."""
        try:
            import docx
            doc = docx.Document(str(file_path))
            return "\n".join([p.text for p in doc.paragraphs if p.text.strip()]).strip()
        except ImportError:
            # Fallback or error
            raise ImportError("python-docx is required to parse .docx files.")

    @staticmethod
    def parse_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """Parses a structured JSON file."""
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @classmethod
    def load_document(cls, file_path: Union[str, Path]) -> str:
        """Loads and extracts text according to file extension."""
        path = Path(file_path)
        ext = path.suffix.lower()

        if ext in [".txt", ".md"]:
            return cls.parse_txt(path)
        elif ext == ".pdf":
            return cls.parse_pdf(path)
        elif ext in [".docx", ".doc"]:
            return cls.parse_docx(path)
        elif ext == ".json":
            data = cls.parse_json(path)
            if isinstance(data, dict):
                # Combine standard job description fields if present
                title = data.get("title", "")
                description = data.get("description", "")
                requirements = " ".join(data.get("requirements", [])) if isinstance(data.get("requirements"), list) else data.get("requirements", "")
                responsibilities = " ".join(data.get("responsibilities", [])) if isinstance(data.get("responsibilities"), list) else data.get("responsibilities", "")
                skills = " ".join(data.get("skills", [])) if isinstance(data.get("skills"), list) else data.get("skills", "")
                return f"{title}\n{description}\nRequirements:\n{requirements}\nResponsibilities:\n{responsibilities}\nSkills:\n{skills}".strip()
            return str(data)
        else:
            # Try reading as text by default
            return cls.parse_txt(path)

    @staticmethod
    def extract_contact_info(text: str) -> Dict[str, Optional[str]]:
        """Extracts candidate contact details (email, phone, linkedin, github) from resume text."""
        # Email pattern
        email_match = re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", text)
        email = email_match.group(0) if email_match else None

        # Phone pattern
        phone_match = re.search(r"\(?\+?[0-9]{1,3}\)?[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{0,4}", text)
        phone = phone_match.group(0).strip() if phone_match else None

        # LinkedIn & GitHub
        linkedin_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/in/([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
        linkedin = linkedin_match.group(0) if linkedin_match else None

        github_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/([a-zA-Z0-9_-]+)", text, re.IGNORECASE)
        github = github_match.group(0) if github_match else None

        # Name extraction heuristic (first line or first non-empty line)
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        candidate_name = lines[0] if lines else "Candidate"
        # If candidate name has email or url, sanitize
        if "@" in candidate_name or "http" in candidate_name or len(candidate_name) > 40:
            candidate_name = "Candidate"

        return {
            "name": candidate_name,
            "email": email,
            "phone": phone,
            "linkedin": linkedin,
            "github": github,
        }

