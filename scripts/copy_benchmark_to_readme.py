#!/usr/bin/env python3
"""scripts/copy_benchmark_to_readme.py: Sync benchmark table from evals/results.md into README.md."""
import re
import sys
from pathlib import Path

# Ensure UTF-8 output
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def extract_benchmark_table(results_path: Path) -> str:
    """Extract the markdown comparison table from Section 1 of evals/results.md."""
    content = results_path.read_text(encoding="utf-8")
    section_match = re.search(
        r"## 1\. Summary Comparison.*?\n(\| Method \|.*?)(?=\n## 2\.|\Z)",
        content,
        re.DOTALL,
    )
    if not section_match:
        raise ValueError("Could not find Section 1 comparison table in evals/results.md")

    table_text = section_match.group(1).strip()
    return table_text


def update_readme(readme_path: Path, table_text: str) -> None:
    """Inject the extracted table between benchmark comment tags in README.md."""
    content = readme_path.read_text(encoding="utf-8")
    start_tag = "<!-- BENCHMARK_TABLE_START -->"
    end_tag = "<!-- BENCHMARK_TABLE_END -->"

    pattern = re.compile(f"{re.escape(start_tag)}.*?{re.escape(end_tag)}", re.DOTALL)
    replacement = f"{start_tag}\n\n{table_text}\n\n{end_tag}"

    if not pattern.search(content):
        raise ValueError(f"Could not find markers {start_tag} and {end_tag} in README.md")

    updated = pattern.sub(replacement, content)
    readme_path.write_text(updated, encoding="utf-8")
    print(f"Successfully copied benchmark table to {readme_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    results_path = repo_root / "evals" / "results.md"
    readme_path = repo_root / "README.md"

    if not results_path.exists():
        print(f"Error: {results_path} not found", file=sys.stderr)
        sys.exit(1)
    if not readme_path.exists():
        print(f"Error: {readme_path} not found", file=sys.stderr)
        sys.exit(1)

    table = extract_benchmark_table(results_path)
    update_readme(readme_path, table)


if __name__ == "__main__":
    main()
