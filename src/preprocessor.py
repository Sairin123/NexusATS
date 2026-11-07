"""Text preprocessing and normalization module for resumes and job descriptions."""
import re
import string
from typing import Dict, List, Optional

# Standard English stopwords fallback in case NLTK data is not pre-downloaded
DEFAULT_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her",
    "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's",
    "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it",
    "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my",
    "myself", "no", "nor", "not", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same",
    "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so",
    "some", "such", "than", "that", "that's", "the", "their", "theirs", "them",
    "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll",
    "they're", "they've", "this", "those", "through", "to", "too", "under", "until",
    "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}

# Preserve programming language terms and domain symbols before punctuation stripping
TECH_PROTECT_MAP = {
    r"(?<!\w)c\+\+(?!\w)": "cpp",
    r"(?<!\w)c#(?!\w)": "csharp",
    r"(?<!\w)\.net(?!\w)": "dotnet",
    r"(?<!\w)node\.js(?!\w)": "nodejs",
    r"(?<!\w)react\.js(?!\w)": "reactjs",
    r"(?<!\w)vue\.js(?!\w)": "vuejs",
    r"(?<!\w)ci/cd(?!\w)": "cicd",
}

TECH_REVERT_MAP = {
    "cpp": "c++",
    "csharp": "c#",
    "dotnet": ".net",
    "nodejs": "node.js",
    "reactjs": "react.js",
    "vuejs": "vue.js",
    "cicd": "ci/cd",
}


class TextPreprocessor:
    """Preprocesses raw text for NLP pipelines, TF-IDF vectorization, and embedding models."""

    def __init__(self, remove_stopwords: bool = True, lemmatize: bool = False):
        self.remove_stopwords = remove_stopwords
        self.lemmatize = lemmatize
        self.stopwords = set(DEFAULT_STOPWORDS)

        # Attempt to load NLTK stopwords and lemmatizer if available
        self._lemmatizer = None
        if self.lemmatize:
            try:
                import nltk
                from nltk.corpus import stopwords
                from nltk.stem import WordNetLemmatizer
                try:
                    self.stopwords = set(stopwords.words("english"))
                except LookupError:
                    nltk.download("stopwords", quiet=True)
                    self.stopwords = set(stopwords.words("english"))

                try:
                    self._lemmatizer = WordNetLemmatizer()
                except LookupError:
                    nltk.download("wordnet", quiet=True)
                    nltk.download("omw-1.4", quiet=True)
                    self._lemmatizer = WordNetLemmatizer()
            except Exception:
                self._lemmatizer = None

    def clean_text(self, text: str, preserve_case: bool = False) -> str:
        """Cleans and standardizes raw text."""
        if not text:
            return ""

        # Normalize linebreaks and unicode spaces
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[\t\u00a0]+", " ", text)

        # Protect known technical terms (e.g., C++, C#, .NET, CI/CD)
        working_text = text if preserve_case else text.lower()
        for pattern, token in TECH_PROTECT_MAP.items():
            working_text = re.sub(pattern, f" {token} ", working_text, flags=re.IGNORECASE)

        # Remove email addresses and URLs
        working_text = re.sub(r"\S+@\S+\.\S+", " ", working_text)
        working_text = re.sub(r"https?://\S+|www\.\S+", " ", working_text)

        # Remove phone numbers (various formats)
        working_text = re.sub(r"\(?\+?[0-9]{1,3}\)?[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{3,4}[-.\s]?[0-9]{0,4}", " ", working_text)

        # Replace hyphens, slashes, and bullet points with spaces
        working_text = re.sub(r"[\u2022\u2023\u25E6\u2043\u2219•*▪►✔-]", " ", working_text)
        working_text = re.sub(r"[/|\\_~]", " ", working_text)

        # Remove remaining punctuation except alphanumeric & protected tokens
        working_text = re.sub(r"[^\w\s]", " ", working_text)

        # Normalize spaces
        tokens = working_text.split()

        # Stopword filtering & optional lemmatization
        processed_tokens = []
        for tok in tokens:
            # Revert protected token back to clean normalized format
            if tok in TECH_REVERT_MAP:
                clean_tok = TECH_REVERT_MAP[tok].lower()
            else:
                clean_tok = tok

            if self.remove_stopwords and clean_tok in self.stopwords:
                continue

            if self.lemmatize and self._lemmatizer and len(clean_tok) > 2:
                clean_tok = self._lemmatizer.lemmatize(clean_tok)

            if clean_tok and len(clean_tok) > 1:
                processed_tokens.append(clean_tok)

        return " ".join(processed_tokens)

    def extract_sections(self, text: str) -> Dict[str, str]:
        """Segments resume text into structural sections: summary, experience, education, skills, projects, certifications."""
        sections = {
            "summary": "",
            "experience": "",
            "education": "",
            "skills": "",
            "projects": "",
            "certifications": "",
            "other": "",
        }

        # Patterns for common resume section headings
        patterns = {
            "summary": r"(?:summary|objective|profile|about\s+me|professional\s+summary)",
            "experience": r"(?:experience|employment|work\s+history|work\s+experience|professional\s+experience)",
            "education": r"(?:education|academic\s+background|qualifications|degrees)",
            "skills": r"(?:skills|technical\s+skills|core\s+competencies|technologies|tools)",
            "projects": r"(?:projects|academic\s+projects|personal\s+projects|portfolio)",
            "certifications": r"(?:certifications|certificates|licenses|courses|awards)",
        }

        # Combine into master regex for section splitting
        master_pat = r"^(?:[#*=-]*\s*)(" + "|".join(patterns.values()) + r")(?:[:\s*-]*)$"
        regex = re.compile(master_pat, re.IGNORECASE | re.MULTILINE)

        splits = list(regex.finditer(text))
        if not splits:
            # If no explicit headings found, treat all as other/summary
            sections["other"] = text.strip()
            return sections

        # Determine matched ranges
        current_section = "other"
        last_idx = 0

        for match in splits:
            matched_title = match.group(1).lower()
            start_pos = match.start()

            # Record previous segment
            segment_text = text[last_idx:start_pos].strip()
            if segment_text:
                sections[current_section] = (sections[current_section] + "\n" + segment_text).strip()

            # Match section title to canonical key
            for key, pat in patterns.items():
                if re.search(pat, matched_title, re.IGNORECASE):
                    current_section = key
                    break

            last_idx = match.end()

        # Final segment
        segment_text = text[last_idx:].strip()
        if segment_text:
            sections[current_section] = (sections[current_section] + "\n" + segment_text).strip()

        return sections
