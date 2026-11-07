"""Command-Line Interface (CLI) for AI-Powered Resume Screening and Job Matching."""
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import JOBS_DIR, RESUMES_DIR
from src.pipeline import ResumeScreeningPipeline


def print_banner():
    banner = """
========================================================================
   AI-POWERED RESUME SCREENING & JOB MATCHING SYSTEM
   [NLP | TF-IDF | Sentence Transformers (SBERT) | Cosine Similarity]
========================================================================
"""
    print(banner)


def run_single(pipeline: ResumeScreeningPipeline, resume_path: str, job_path: str, output: str = None):
    print(f"[*] Screening resume: {resume_path}")
    print(f"[*] Against job:      {job_path}")
    print("-" * 72)

    result = pipeline.screen_single(resume_path, job_path)

    print(f"\n>> CANDIDATE:     {result['candidate']['name']}")
    if result['candidate']['email']:
        print(f">> CONTACT:       {result['candidate']['email']} | {result['candidate']['phone']}")
    print(f">> TARGET ROLE:   {result['job_title']}")
    print(f">> OVERALL MATCH: {result['overall_score']}%  [{result['match_tier'].upper()}]")
    print(f">> LATENCY:       {result['latency_ms']} ms\n")

    print("--- SCORE BREAKDOWN (Cosine Similarity & Metrics) ---")
    comps = result["components"]
    print(f"  * Semantic Cosine Similarity (SBERT): {comps['semantic_similarity']}%")
    print(f"  * Lexical Cosine Similarity (TF-IDF): {comps['tfidf_similarity']}%")
    print(f"  * Hard Skills Coverage:               {comps['skill_coverage']}%")
    print(f"  * Experience Relevance:               {comps['experience_relevance']}%\n")

    skills = result["skill_analysis"]
    print(f"--- SKILLS OVERVIEW (Matched: {skills['matched_count']}/{skills['required_count']}) ---")
    print(f"  [+] Matched: {', '.join(skills['matched_skills']) if skills['matched_skills'] else 'None'}")
    print(f"  [-] Missing: {', '.join(skills['missing_skills']) if skills['missing_skills'] else 'None'}\n")

    if result["top_matching_keywords"]:
        print("--- TOP TF-IDF CONTRIBUTING KEYWORDS ---")
        kw_list = [f"{k['keyword']} ({k['importance_score']})" for k in result["top_matching_keywords"][:6]]
        print("  " + ", ".join(kw_list) + "\n")

    print("--- EXECUTIVE SUMMARY ---")
    print(f"  {result['executive_summary'].replace('**', '')}\n")

    print("--- ACTIONABLE RECOMMENDATIONS ---")
    for rec in result["recommendations"]:
        print(f"  - {rec.replace('**', '').replace('`', '')}")
    print("=" * 72)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"\n[+] Full report saved to: {output}")


def run_batch(pipeline: ResumeScreeningPipeline, resumes_dir: str, job_path: str, output: str = None):
    res_dir = Path(resumes_dir)
    resume_files = sorted(list(res_dir.glob("*.txt")) + list(res_dir.glob("*.pdf")) + list(res_dir.glob("*.docx")))

    if not resume_files:
        print(f"[!] No resume files found in {resumes_dir}")
        return

    print(f"[*] Screening {len(resume_files)} resumes against: {job_path}\n")

    ranked = pipeline.screen_batch(resume_files, job_path)

    print(f"{'RANK':<5} {'CANDIDATE':<20} {'SCORE':<10} {'TIER':<16} {'SEMANTIC':<10} {'TF-IDF':<10} {'SKILLS':<8}")
    print("-" * 82)
    for r in ranked:
        comps = r["components"]
        print(
            f"{r['rank']:<5} {r['candidate']['name']:<20} {r['overall_score']:<9}% "
            f"{r['match_tier']:<16} {comps['semantic_similarity']:<9}% {comps['tfidf_similarity']:<9}% "
            f"{r['skill_analysis']['matched_count']}/{r['skill_analysis']['required_count']}"
        )
    print("=" * 82)

    if output:
        with open(output, "w", encoding="utf-8") as f:
            json.dump(ranked, f, indent=2)
        print(f"\n[+] Batch leaderboard saved to: {output}")


def run_recommend(pipeline: ResumeScreeningPipeline, resume_path: str, jobs_dir: str):
    j_dir = Path(jobs_dir)
    job_files = sorted(list(j_dir.glob("*.json")) + list(j_dir.glob("*.txt")))

    print(f"[*] Finding best job matches for resume: {resume_path}\n")
    matches = pipeline.match_jobs_for_candidate(resume_path, job_files)

    print(f"{'RANK':<5} {'JOB TITLE':<35} {'SCORE':<10} {'TIER':<16}")
    print("-" * 68)
    for m in matches:
        print(f"{m['rank']:<5} {m['job_title']:<35} {m['overall_score']:<9}% {m['match_tier']:<16}")
    print("=" * 68)


def run_benchmark(pipeline: ResumeScreeningPipeline, resume_path: str, job_path: str):
    print(f"[*] Benchmarking matching models on: {Path(resume_path).name} vs {Path(job_path).name}\n")
    bm = pipeline.benchmark_methods(resume_path, job_path)

    print(f"{'MODEL / METHOD':<28} {'SCORE':<12} {'LATENCY':<12} {'PARADIGM'}")
    print("-" * 72)
    for name, data in bm.items():
        print(f"{name.upper():<28} {data['score_percent']:<11}% {data['latency_ms']:<11}ms {data['type']}")
        print(f"  -> Advantage: {data['advantage']}")
    print("=" * 72)


def main():
    print_banner()

    parser = argparse.ArgumentParser(description="AI Resume Screening and Job Matching CLI")
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "recommend", "benchmark"],
        default="single",
        help="Operation mode (single, batch, recommend, benchmark)",
    )
    parser.add_argument("--resume", default=str(RESUMES_DIR / "resume_ml_engineer_alice.txt"), help="Path to resume file")
    parser.add_argument("--job", default=str(JOBS_DIR / "job_senior_ml_engineer.json"), help="Path to job file")
    parser.add_argument("--resumes-dir", default=str(RESUMES_DIR), help="Directory of resumes for batch screening")
    parser.add_argument("--jobs-dir", default=str(JOBS_DIR), help="Directory of jobs for recommendation")
    parser.add_argument("--output", default=None, help="Output JSON file for results")

    args = parser.parse_args()

    print("[*] Initializing NLP & SBERT Pipeline (SentenceTransformer all-MiniLM-L6-v2)...")
    pipeline = ResumeScreeningPipeline()
    print("[+] Pipeline initialized successfully!\n")

    if args.mode == "single":
        run_single(pipeline, args.resume, args.job, args.output)
    elif args.mode == "batch":
        run_batch(pipeline, args.resumes_dir, args.job, args.output)
    elif args.mode == "recommend":
        run_recommend(pipeline, args.resume, args.jobs_dir)
    elif args.mode == "benchmark":
        run_benchmark(pipeline, args.resume, args.job)


if __name__ == "__main__":
    main()

