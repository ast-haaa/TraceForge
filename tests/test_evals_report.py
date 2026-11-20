"""Tests for validating evals/results.md structure and content."""
from pathlib import Path


def test_evals_results_file_exists_and_non_empty():
    results_path = Path("evals/results.md")
    assert results_path.exists(), "evals/results.md does not exist"
    content = results_path.read_text(encoding="utf-8")
    assert len(content.strip()) > 0, "evals/results.md is empty"


def test_evals_results_contains_four_required_sections():
    """Asserts that evals/results.md parses and contains all four mandatory sections."""
    results_path = Path("evals/results.md")
    content = results_path.read_text(encoding="utf-8")

    required_sections = [
        "## 1. Summary Comparison",
        "## 2. Per-Fault Breakdown",
        "## 3. Held-Out Evaluation",
        "## 4. Where It Fails",
    ]

    for section in required_sections:
        assert section in content, f"Missing required section in evals/results.md: {section}"


def test_evals_results_summary_table_structure():
    """Verify that Section 1 contains the comparative evaluation table with required columns."""
    results_path = Path("evals/results.md")
    content = results_path.read_text(encoding="utf-8")

    # Check key columns
    assert "Top-1 Accuracy" in content
    assert "Top-3 Accuracy" in content
    assert "Mean Tokens" in content
    assert "Mean Latency" in content

    # Check that baselines and investigator are listed
    assert "Random Step" in content
    assert "Always Decision Step" in content
    assert "Deterministic Diff" in content
    assert "LLM Investigator" in content


def test_readme_benchmark_table_matches_evals_results():
    """Verify that the benchmark table in README.md matches evals/results.md via the sync script."""
    from scripts.copy_benchmark_to_readme import extract_benchmark_table

    results_path = Path("evals/results.md")
    readme_path = Path("README.md")
    table = extract_benchmark_table(results_path)
    readme_content = readme_path.read_text(encoding="utf-8")
    assert table in readme_content, "README.md benchmark table does not match evals/results.md"

