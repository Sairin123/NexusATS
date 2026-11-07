"""Skills extraction module using curated technical and soft skills taxonomy."""
import re
from typing import Dict, List, Set, Tuple

# Comprehensive taxonomy of skills mapped to categories
SKILLS_TAXONOMY: Dict[str, List[str]] = {
    "machine_learning_ai": [
        "python", "pytorch", "tensorflow", "keras", "scikit-learn", "sklearn",
        "pandas", "numpy", "scipy", "hugging face", "transformers", "bert",
        "sentence transformers", "sbert", "llm", "large language models",
        "natural language processing", "nlp", "computer vision", "deep learning",
        "machine learning", "neural networks", "xgboost", "lightgbm", "spacy",
        "nltk", "opencv", "rag", "langchain", "llamaindex", "prompt engineering",
        "vector databases", "chromadb", "pinecone", "milvus", "qdrant", "weaviate",
        "fine-tuning", "genai", "generative ai", "reinforcement learning", "supervised learning",
        "unsupervised learning", "mlflow", "wandb", "dvc", "kubeflow"
    ],
    "programming_languages": [
        "python", "java", "c++", "c#", "c", "javascript", "typescript", "golang", "go",
        "rust", "ruby", "php", "swift", "kotlin", "scala", "r", "julia", "matlab",
        "sql", "bash", "shell", "powershell", "dart", "html", "css"
    ],
    "web_frameworks": [
        "react", "react.js", "next.js", "vue", "vue.js", "angular", "node.js",
        "express", "express.js", "django", "flask", "fastapi", "spring boot", "spring",
        "asp.net", ".net core", ".net", "ruby on rails", "laravel", "graphql",
        "rest api", "restful", "tailwind css", "bootstrap", "redux", "svelte"
    ],
    "cloud_devops": [
        "aws", "amazon web services", "azure", "gcp", "google cloud", "docker",
        "kubernetes", "k8s", "terraform", "ansible", "ci/cd", "continuous integration",
        "github actions", "gitlab ci", "jenkins", "helm", "linux", "cloudformation",
        "serverless", "lambda", "cloudwatch", "prometheus", "grafana", "nginx", "argo"
    ],
    "databases_data_eng": [
        "postgresql", "postgres", "mysql", "mongodb", "redis", "cassandra",
        "dynamodb", "elasticsearch", "sqlite", "snowflake", "bigquery", "redshift",
        "neo4j", "apache spark", "spark", "apache kafka", "kafka", "apache airflow",
        "airflow", "hadoop", "flink", "dbt", "etl", "data warehousing", "databricks"
    ],
    "business_analysis": [
        "business analysis", "requirements gathering", "user stories", "gap analysis",
        "stakeholder management", "process mapping", "bpm", "bpmn", "swot analysis",
        "feasibility study", "jira", "confluence", "visio", "power bi", "tableau",
        "excel", "vlookup", "pivot tables", "kpi tracking", "market research", "data visualization"
    ],
    "corporate_finance_hr": [
        "financial modeling", "budgeting", "forecasting", "p&l", "variance analysis",
        "gaap", "dcf", "valuation", "cost analysis", "auditing", "risk management",
        "talent acquisition", "recruiting", "hris", "onboarding", "employee relations",
        "performance management", "workday", "payroll", "change management", "compliance",
        "vendor management", "strategic planning", "contract negotiation"
    ],
    "cybersecurity_it": [
        "soc", "siem", "penetration testing", "nist", "iso 27001", "vulnerability assessment",
        "incident response", "firewall", "encryption", "iam", "zero trust", "network security"
    ],
    "soft_skills_practices": [
        "leadership", "agile", "scrum", "kanban", "communication", "problem solving",
        "critical thinking", "project management", "mentoring", "cross-functional",
        "system design", "code review", "unit testing", "tdd", "software architecture"
    ]
}

# Inverted mapping: skill -> category
SKILL_TO_CATEGORY: Dict[str, str] = {}
for category, skills in SKILLS_TAXONOMY.items():
    for skill in skills:
        SKILL_TO_CATEGORY[skill.lower()] = category


class SkillsExtractor:
    """Extracts technical and soft skills from resume and job description texts."""

    def __init__(self, custom_taxonomy: Dict[str, List[str]] = None):
        self.taxonomy = custom_taxonomy or SKILLS_TAXONOMY
        self._build_regex_patterns()

    def _build_regex_patterns(self):
        """Builds compiled regular expressions for exact skill matching with word boundaries."""
        self.patterns = []
        # Sort skills by length descending to match longer multi-word phrases first (e.g. 'sentence transformers' before 'transformers')
        all_skills = set()
        for cat_skills in self.taxonomy.values():
            all_skills.update(s.lower() for s in cat_skills)

        sorted_skills = sorted(all_skills, key=lambda s: len(s), reverse=True)

        for skill in sorted_skills:
            # Escape regex special characters while maintaining word boundary check
            escaped = re.escape(skill)
            # Special case for C++, C#, .NET
            if skill in ["c++", "c#", ".net"]:
                pattern = re.compile(rf"(?<!\w){escaped}(?!\w)", re.IGNORECASE)
            else:
                pattern = re.compile(rf"\b{escaped}\b", re.IGNORECASE)
            self.patterns.append((skill, pattern))

    def extract_skills(self, text: str) -> Dict[str, List[str]]:
        """Extracts skills categorized by domain from the given text."""
        if not text:
            return {cat: [] for cat in self.taxonomy.keys()}

        found_skills: Set[str] = set()
        for skill, pattern in self.patterns:
            if pattern.search(text):
                found_skills.add(skill)

        # Categorize extracted skills
        categorized: Dict[str, List[str]] = {cat: [] for cat in self.taxonomy.keys()}
        for skill in found_skills:
            cat = SKILL_TO_CATEGORY.get(skill, "other")
            if cat in categorized:
                categorized[cat].append(skill)
            else:
                categorized.setdefault(cat, []).append(skill)

        # Sort each list alphabetically
        for cat in categorized:
            categorized[cat].sort()

        return categorized

    def get_all_skills_flat(self, text: str) -> List[str]:
        """Returns a flat sorted list of unique extracted skills."""
        categorized = self.extract_skills(text)
        flat = []
        for skills in categorized.values():
            flat.extend(skills)
        return sorted(list(set(flat)))

    def calculate_skill_overlap(self, resume_text: str, job_text: str) -> Dict[str, any]:
        """Calculates matched skills, missing skills, and Jaccard / Overlap similarity."""
        resume_skills = set(self.get_all_skills_flat(resume_text))
        job_skills = set(self.get_all_skills_flat(job_text))

        matched = resume_skills.intersection(job_skills)
        missing = job_skills.difference(resume_skills)
        additional = resume_skills.difference(job_skills)

        # Coverage ratio: matched / required
        coverage = len(matched) / len(job_skills) if job_skills else 1.0

        # Jaccard index
        union = resume_skills.union(job_skills)
        jaccard = len(matched) / len(union) if union else 1.0

        return {
            "matched_skills": sorted(list(matched)),
            "missing_skills": sorted(list(missing)),
            "additional_skills": sorted(list(additional)),
            "required_count": len(job_skills),
            "matched_count": len(matched),
            "missing_count": len(missing),
            "coverage_score": round(coverage, 4),
            "jaccard_score": round(jaccard, 4),
        }

