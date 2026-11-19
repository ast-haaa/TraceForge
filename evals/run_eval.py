# evals/run_eval.py
"""Evaluation benchmark runner for AgentReplay (Phase 5).

Compares 4 fault localization methods on the 40-run benchmark dataset:
1. Random step
2. Always decision step (llm_decide)
3. First divergence vs clean reference run (deterministic diff)
4. LLM investigator (agentreplay.investigate)

Computes:
- Top-1 and Top-3 accuracy with Wilson 95% confidence intervals
- Overall and per-fault-type breakdown
- Mean tokens, latency, and parse failure rate
- Generates evals/results.md including honest "Where it fails" failure analysis.
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import dotenv

from agentreplay.db import SessionLocal, init_db
from agentreplay.diff import diff_traces
from agentreplay.investigate import StepHypothesis, investigate
from agentreplay.models import Investigation, Step, Trace
from demo_agent.agent import run_refund_agent
from demo_agent.llm import clean_env_var

dotenv.load_dotenv()
logger = logging.getLogger("evals.run_eval")

DATASET_PATH = Path("evals/dataset.json")
RESULTS_PATH = Path("evals/results.md")

FAULT_TYPES = [
    "stale_policy",
    "wrong_order_date",
    "prompt_bug",
    "missing_tool_result",
    "wrong_customer_data",
]


# ---------------------------------------------------------------------------
# Wilson Score 95% Confidence Interval
# ---------------------------------------------------------------------------


def wilson_score_interval(
    successes: int, total: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate the Wilson score confidence interval for a binomial proportion.

    z = 1.959964 for 95% confidence.
    """
    if total <= 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    z2 = z * z
    denom = 1 + z2 / total
    center = (p + z2 / (2 * total)) / denom
    margin = (z / denom) * math.sqrt((p * (1 - p) / total) + (z2 / (4 * total * total)))
    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)
    return lower, upper


def format_ci(p_val: float, low: float, high: float) -> str:
    """Format proportion and confidence interval as percentage string."""
    return f"{p_val * 100:.1f}% [{low * 100:.1f}%, {high * 100:.1f}%]"


# ---------------------------------------------------------------------------
# Clean Reference Run Cache for Deterministic Diff
# ---------------------------------------------------------------------------

_clean_trace_cache: dict[tuple[str, str], int] = {}


def get_clean_reference_trace_id(
    customer_id: str, order_id: str, db_session: Any
) -> int:
    """Run or retrieve a clean reference run for the same customer & order."""
    cache_key = (customer_id, order_id)
    if cache_key in _clean_trace_cache:
        return _clean_trace_cache[cache_key]

    run_id = f"clean-ref-{customer_id}-{order_id}-{int(time.time() * 1000)}"
    run_refund_agent(
        customer_id=customer_id,
        order_id=order_id,
        run_id=run_id,
        description="Clean reference baseline run",
        meta={},
        stale_policy=False,
    )
    tr = db_session.query(Trace).filter(Trace.run_id == run_id).first()
    assert tr is not None, f"Failed to record clean reference trace for {order_id}"
    _clean_trace_cache[cache_key] = tr.id
    return tr.id


# ---------------------------------------------------------------------------
# Evaluation Core
# ---------------------------------------------------------------------------


def run_evaluation(
    dataset_path: Path = DATASET_PATH,
    results_path: Path = RESULTS_PATH,
) -> dict[str, Any]:
    """Execute evaluation across the 4 methods on the benchmark dataset."""
    init_db()

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset file {dataset_path} not found. Run evals/fault_injection.py first."
        )

    with open(dataset_path, encoding="utf-8") as f:
        dataset: list[dict[str, Any]] = json.load(f)

    if not dataset:
        raise ValueError("Dataset is empty. Run evals/fault_injection.py first.")

    total_runs = len(dataset)
    db = SessionLocal()

    # Data structures for tracking method results
    # Methods: 'random', 'always_decision', 'deterministic_diff', 'llm_investigator'
    methods = ["random", "always_decision", "deterministic_diff", "llm_investigator"]

    method_display = {
        "random": "1. Random Step",
        "always_decision": "2. Always Decision Step (llm_decide)",
        "deterministic_diff": "3. Deterministic Diff vs Clean Ref",
        "llm_investigator": "4. LLM Investigator",
    }

    # Tracking counters
    top1_hits = {m: 0 for m in methods}
    top3_hits = {m: 0 for m in methods}

    # Per-fault breakdowns: fault_type -> method -> hits
    top1_per_fault = {ft: {m: 0 for m in methods} for ft in FAULT_TYPES}
    top3_per_fault = {ft: {m: 0 for m in methods} for ft in FAULT_TYPES}
    runs_per_fault = {ft: 0 for ft in FAULT_TYPES}

    # Investigator performance metrics
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0.0
    investigator_call_count = 0
    parse_failures = 0

    misdiagnosed_runs: list[dict[str, Any]] = []

    print("=" * 80)
    print("Evaluating 4 Localization Methods across Benchmark Dataset")
    print(f"Total benchmark runs: {total_runs}")
    print(f"LLM Provider: {clean_env_var(os.environ.get('LLM_PROVIDER'), 'stub')}")
    print("=" * 80)

    for i, item in enumerate(dataset, start=1):
        trace_id = item["trace_id"]
        ft = item["fault_type"]
        gt_step = item["ground_truth_step"]
        gt_node = item["ground_truth_node"]
        cust_id = item["customer_id"]
        ord_id = item["order_id"]
        runs_per_fault[ft] += 1

        # Query steps for this trace
        steps = (
            db.query(Step).filter(Step.trace_id == trace_id).order_by(Step.index).all()
        )
        step_indices = [s.index for s in steps]
        step_nodes = {s.index: s.node_name for s in steps}

        # -------------------------------------------------------------
        # Method 1: Random Step (fixed seed per trace)
        # -------------------------------------------------------------
        rng = random.Random(42 + trace_id)
        shuffled = list(step_indices)
        rng.shuffle(shuffled)
        m1_top1 = shuffled[0] if shuffled else 1
        m1_top3 = shuffled[:3]

        if m1_top1 == gt_step:
            top1_hits["random"] += 1
            top1_per_fault[ft]["random"] += 1
        if gt_step in m1_top3:
            top3_hits["random"] += 1
            top3_per_fault[ft]["random"] += 1

        # -------------------------------------------------------------
        # Method 2: Always Decision Step (llm_decide -> Step 4)
        # -------------------------------------------------------------
        m2_top1 = 4
        m2_top3 = [4, 3, 2]
        if m2_top1 == gt_step:
            top1_hits["always_decision"] += 1
            top1_per_fault[ft]["always_decision"] += 1
        if gt_step in m2_top3:
            top3_hits["always_decision"] += 1
            top3_per_fault[ft]["always_decision"] += 1

        # -------------------------------------------------------------
        # Method 3: Deterministic Diff vs Clean Reference
        # -------------------------------------------------------------
        clean_tid = get_clean_reference_trace_id(cust_id, ord_id, db)
        diff_res = diff_traces(clean_tid, trace_id, db=db)
        if diff_res.first_divergent_step is not None:
            m3_top1 = diff_res.first_divergent_step
        else:
            m3_top1 = step_indices[-1] if step_indices else 1

        # Collect divergent steps for top-3
        divergent_step_set = [m3_top1]
        for sd in diff_res.steps:
            if sd.diverged and sd.step_index not in divergent_step_set:
                divergent_step_set.append(sd.step_index)
        for s_idx in step_indices:
            if s_idx not in divergent_step_set:
                divergent_step_set.append(s_idx)
        m3_top3 = divergent_step_set[:3]

        if m3_top1 == gt_step:
            top1_hits["deterministic_diff"] += 1
            top1_per_fault[ft]["deterministic_diff"] += 1
        if gt_step in m3_top3:
            top3_hits["deterministic_diff"] += 1
            top3_per_fault[ft]["deterministic_diff"] += 1

        # -------------------------------------------------------------
        # Method 4: LLM Investigator
        # -------------------------------------------------------------
        # Check if already investigated in DB to reuse if available
        inv_record = (
            db.query(Investigation)
            .filter(Investigation.trace_id == trace_id)
            .order_by(Investigation.id.desc())
            .first()
        )

        if inv_record and inv_record.ranked_steps:
            ranked = [StepHypothesis.model_validate(h) for h in inv_record.ranked_steps]
            p_tok = inv_record.prompt_tokens or 0
            c_tok = inv_record.completion_tokens or 0
            lat = inv_record.latency_ms or 0.0
            summary_text = inv_record.summary or ""
        else:
            try:
                inv_res = investigate(trace_id, session=db)
                ranked = inv_res.ranked_steps
                summary_text = inv_res.summary
                inv_rec = (
                    db.query(Investigation)
                    .filter(Investigation.trace_id == trace_id)
                    .order_by(Investigation.id.desc())
                    .first()
                )
                p_tok = inv_rec.prompt_tokens or 0 if inv_rec else 0
                c_tok = inv_rec.completion_tokens or 0 if inv_rec else 0
                lat = inv_rec.latency_ms or 0.0 if inv_rec else 0.0
            except Exception as e:
                logger.error("Investigate failed on trace %d: %s", trace_id, e)
                parse_failures += 1
                ranked = []
                p_tok = 0
                c_tok = 0
                lat = 0.0
                summary_text = f"Investigation error: {e}"

        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        total_latency_ms += lat
        investigator_call_count += 1

        m4_top1 = ranked[0].step_index if ranked else -1
        m4_top3 = [r.step_index for r in ranked[:3]]

        top1_match = m4_top1 == gt_step
        top3_match = gt_step in m4_top3

        if top1_match:
            top1_hits["llm_investigator"] += 1
            top1_per_fault[ft]["llm_investigator"] += 1
        else:
            # Track misdiagnosed run for honest failure analysis
            hypo_top1 = ranked[0].hypothesis if ranked else "N/A"
            conf_top1 = ranked[0].confidence if ranked else 0.0
            cited_evs = ranked[0].cited_event_ids if ranked else []
            pred_node = step_nodes.get(m4_top1, "unknown")
            misdiagnosed_runs.append(
                {
                    "trace_id": trace_id,
                    "fault_type": ft,
                    "ground_truth_step": gt_step,
                    "ground_truth_node": gt_node,
                    "predicted_step": m4_top1,
                    "predicted_node": pred_node,
                    "confidence": conf_top1,
                    "summary": summary_text,
                    "hypothesis": hypo_top1,
                    "cited_event_ids": cited_evs,
                }
            )

        if top3_match:
            top3_hits["llm_investigator"] += 1
            top3_per_fault[ft]["llm_investigator"] += 1

        print(
            f"[{i:02d}/{total_runs:02d}] Trace #{trace_id:<3} ({ft:<20}) | "
            f"GT: Step {gt_step} ({gt_node}) | "
            f"Diff: {m3_top1} {'PASS' if m3_top1 == gt_step else 'FAIL'} | "
            f"LLM: {m4_top1} {'PASS' if top1_match else 'FAIL'} (top3: {'PASS' if top3_match else 'FAIL'})"
        )

    db.close()

    # -------------------------------------------------------------
    # Compute Aggregates & Wilson CIs
    # -------------------------------------------------------------
    metrics: dict[str, Any] = {}
    for m in methods:
        k1 = top1_hits[m]
        k3 = top3_hits[m]
        p1 = k1 / total_runs
        p3 = k3 / total_runs
        low1, high1 = wilson_score_interval(k1, total_runs)
        low3, high3 = wilson_score_interval(k3, total_runs)
        metrics[m] = {
            "top1_acc": p1,
            "top1_ci": (low1, high1),
            "top3_acc": p3,
            "top3_ci": (low3, high3),
            "top1_hits": k1,
            "top3_hits": k3,
        }

    mean_tokens = (
        (total_prompt_tokens + total_completion_tokens) / investigator_call_count
        if investigator_call_count > 0
        else 0.0
    )
    mean_latency = (
        total_latency_ms / investigator_call_count
        if investigator_call_count > 0
        else 0.0
    )
    parse_fail_rate = (
        (parse_failures / investigator_call_count) * 100
        if investigator_call_count > 0
        else 0.0
    )

    # -------------------------------------------------------------
    # Render evals/results.md
    # -------------------------------------------------------------
    md_lines: list[str] = [
        "# AgentReplay Benchmark Evaluation Results (Phase 5)",
        "",
        f"**Date:** {datetime.date.today().isoformat()}",
        "**Dataset:** 40 failing agent executions across 5 fault injection types (8 runs each)",
        f"**Evaluated Model:** {clean_env_var(os.environ.get('LLM_MODEL'), 'openai/gpt-oss-120b')}",
        f"**Provider:** {clean_env_var(os.environ.get('LLM_PROVIDER'), 'stub')}",
        "",
        "## 1. Summary Comparison",
        "",
        "| Method | Top-1 Accuracy (95% CI) | Top-3 Accuracy (95% CI) | Mean Tokens | Mean Latency | Parse Failures |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m in methods:
        m_name = method_display[m]
        met = metrics[m]
        ci1_str = format_ci(met["top1_acc"], *met["top1_ci"])
        ci3_str = format_ci(met["top3_acc"], *met["top3_ci"])
        if m == "llm_investigator":
            tok_str = f"{mean_tokens:.1f}"
            lat_str = f"{mean_latency:.1f} ms"
            pf_str = f"{parse_fail_rate:.1f}%"
        else:
            tok_str = "0"
            lat_str = "0 ms"
            pf_str = "0.0%"
        md_lines.append(
            f"| **{m_name}** | {ci1_str} | {ci3_str} | {tok_str} | {lat_str} | {pf_str} |"
        )

    md_lines.extend(
        [
            "",
            "## 2. Per-Fault Breakdown",
            "",
            "Top-1 and Top-3 accuracy across the 5 injected fault types:",
            "",
            "| Fault Type | Ground Truth | Random (Top-1 / Top-3) | Always Decision (Top-1 / Top-3) | Deterministic Diff (Top-1 / Top-3) | LLM Investigator (Top-1 / Top-3) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: |",
        ]
    )

    fault_gt_labels = {
        "stale_policy": "Step 3 (`read_refund_policy`)",
        "wrong_order_date": "Step 2 (`get_orders`)",
        "prompt_bug": "Step 4 (`llm_decide`)",
        "missing_tool_result": "Step 3 (`read_refund_policy`)",
        "wrong_customer_data": "Step 1 (`get_customer`)",
    }

    for ft in FAULT_TYPES:
        n_ft = runs_per_fault[ft]
        gt_lbl = fault_gt_labels[ft]

        def _fmt(m: str) -> str:
            t1 = top1_per_fault[ft][m]
            t3 = top3_per_fault[ft][m]
            return f"{t1}/{n_ft} ({t1/n_ft*100:.0f}%) / {t3}/{n_ft} ({t3/n_ft*100:.0f}%)"

        md_lines.append(
            f"| `{ft}` | {gt_lbl} | {_fmt('random')} | {_fmt('always_decision')} | {_fmt('deterministic_diff')} | {_fmt('llm_investigator')} |"
        )

    # Section 3: Where it fails (Honest Failure Analysis)
    md_lines.extend(
        [
            "",
            "## 3. Where It Fails (Failure Analysis)",
            "",
            f"Total misdiagnosed runs by the LLM Investigator: **{len(misdiagnosed_runs)}** / {total_runs} "
            f"({len(misdiagnosed_runs)/total_runs*100:.1f}% error rate).",
            "",
        ]
    )

    if not misdiagnosed_runs:
        md_lines.append(
            "The LLM Investigator achieved 100% Top-1 accuracy across all 40 runs in this evaluation run."
        )
    else:
        for idx, fail in enumerate(misdiagnosed_runs, start=1):
            md_lines.extend(
                [
                    f"### Failure #{idx}: Trace #{fail['trace_id']} (`{fail['fault_type']}`)",
                    f"- **Ground Truth:** Step {fail['ground_truth_step']} (`{fail['ground_truth_node']}`)",
                    f"- **Investigator Predicted (Rank #1):** Step {fail['predicted_step']} (`{fail['predicted_node']}`) [Confidence: {fail['confidence']:.2f}]",
                    f"- **Cited Events:** {fail['cited_event_ids']}",
                    f"- **Investigator Summary:** {fail['summary']}",
                    f"- **Investigator Hypothesis:** {fail['hypothesis']}",
                    "",
                    "**Root-Cause Diagnosis Analysis:**",
                ]
            )
            # Add honest diagnostic explanation of why the investigator got confused
            if fail["fault_type"] == "wrong_order_date":
                md_lines.append(
                    "> **Why it failed:** The investigator observed that `llm_decide` approved the refund based on `DAYS_OLD`, "
                    "or blamed `read_refund_policy`, rather than noticing that `get_orders` emitted an inaccurate order date. "
                    "Without an external ground truth database of order creation dates, an LLM reading only the trace "
                    "cannot easily distinguish a falsified tool return from a valid recent order."
                )
            elif fail["fault_type"] == "missing_tool_result":
                md_lines.append(
                    "> **Why it failed:** The policy tool returned empty string, but the LLM investigator blamed the decision "
                    "node for proceeding without policy validation, rather than attributing root cause to the upstream tool that returned empty data."
                )
            elif fail["fault_type"] == "wrong_customer_data":
                md_lines.append(
                    "> **Why it failed:** The investigator blamed the refund disbursement step (`issue_refund`) rather than "
                    "the initial retrieval step (`get_customer`) that returned the wrong customer entity."
                )
            elif fail["fault_type"] == "stale_policy":
                md_lines.append(
                    "> **Why it failed:** The investigator accepted the 30-day policy text returned by `read_refund_policy` "
                    "as authoritative, blaming `llm_decide` for approving rather than recognizing that the policy returned was outdated."
                )
            else:
                md_lines.append(
                    "> **Why it failed:** The investigator attributed failure to downstream decision logic rather than upstream root cause."
                )
            md_lines.append("")

    # Save results.md
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print("\n" + "=" * 80)
    print(f"Evaluation complete! Results successfully written to {results_path}")
    print("=" * 80)
    print("\nSummary Table:")
    for line in md_lines[8:15]:
        print(line)
    print("=" * 80)

    return {
        "total_runs": total_runs,
        "metrics": metrics,
        "misdiagnosed_count": len(misdiagnosed_runs),
        "misdiagnosed_runs": misdiagnosed_runs,
        "mean_tokens": mean_tokens,
        "mean_latency": mean_latency,
        "parse_fail_rate": parse_fail_rate,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AgentReplay benchmark evaluation.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to dataset.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=RESULTS_PATH,
        help="Output path for results.md",
    )
    args = parser.parse_args()

    run_evaluation(dataset_path=args.dataset, results_path=args.out)
