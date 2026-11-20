# evals/run_eval.py
"""Evaluation benchmark runner for AgentReplay (Phase 5).

Compares 4 fault localization methods on the benchmark dataset:
1. Random step
2. Always decision step (llm_decide)
3. First divergence vs clean reference run (deterministic diff)
4. LLM investigator (agentreplay.investigate with frozen generic prompt)

Computes:
- Top-1 and Top-3 accuracy with Wilson 95% confidence intervals
- Overall and per-fault-type breakdown
- Held-out evaluation on new fault types (duplicate_refund, wrong_amount)
- Mean tokens, latency, and parse failure rate
- Generates evals/results.md with honest dynamic failure analysis and limitations.
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

import demo_agent.tools as tools_mod
from agentreplay.db import SessionLocal, init_db
from agentreplay.diff import diff_traces
from agentreplay.investigate import (
    INVESTIGATOR_PROMPT_HASH,
    StepHypothesis,
    investigate,
)
from agentreplay.models import Investigation, Step, Trace
from demo_agent.agent import run_refund_agent
from demo_agent.llm import clean_env_var

dotenv.load_dotenv()
logger = logging.getLogger("evals.run_eval")

DATASET_PATH = Path("evals/dataset.json")
HELD_OUT_PATH = Path("evals/held_out_dataset.json")
RESULTS_PATH = Path("evals/results.md")

FAULT_TYPES = [
    "stale_policy",
    "wrong_order_date",
    "prompt_bug",
    "missing_tool_result",
    "wrong_customer_data",
]

HELD_OUT_FAULT_TYPES = [
    "duplicate_refund",
    "wrong_amount",
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
# Clean Reference Run for Deterministic Diff
# ---------------------------------------------------------------------------

_clean_trace_cache: dict[tuple[str, str], int] = {}


def get_clean_reference_trace_id(
    customer_id: str,
    customer_name: str,
    order_id: str,
    true_days_old: int,
    amount: float,
    db_session: Any,
) -> int:
    """Run or retrieve a clean reference run for the same customer & order.

    Ensures the clean reference run executes with the true, uncorrupted customer
    and order data under the standard 7-day policy.
    """
    cache_key = (customer_id, order_id)
    if cache_key in _clean_trace_cache:
        return _clean_trace_cache[cache_key]

    # Populate tools_mod with the TRUE customer and order for this clean run if not present
    if customer_id not in tools_mod.CUSTOMERS:
        tools_mod.CUSTOMERS[customer_id] = {
            "customer_id": customer_id,
            "name": customer_name,
            "email": f"{customer_id.lower()}@example.com",
        }
    if order_id not in tools_mod.ORDERS:
        today = datetime.date.today()
        ord_date = (today - datetime.timedelta(days=true_days_old)).isoformat()
        tools_mod.ORDERS[order_id] = {
            "order_id": order_id,
            "customer_id": customer_id,
            "amount": amount,
            "order_date": ord_date,
            "item": f"Item {order_id}",
        }

    run_id = f"clean-ref-{customer_id}-{order_id}-{int(time.time() * 1000)}"

    # Run clean reference offline (using stub or provider) with no faults
    prev_provider = os.environ.get("LLM_PROVIDER")
    os.environ["LLM_PROVIDER"] = "stub"
    try:
        run_refund_agent(
            customer_id=customer_id,
            order_id=order_id,
            run_id=run_id,
            description="Clean reference baseline run",
            meta={},
            stale_policy=False,
            fault=None,
        )
    finally:
        if prev_provider is not None:
            os.environ["LLM_PROVIDER"] = prev_provider
        else:
            os.environ.pop("LLM_PROVIDER", None)

    tr = db_session.query(Trace).filter(Trace.run_id == run_id).first()
    assert tr is not None, f"Failed to record clean reference trace for {order_id}"
    _clean_trace_cache[cache_key] = tr.id
    return tr.id


# ---------------------------------------------------------------------------
# Evaluation Helper for a Dataset Slice
# ---------------------------------------------------------------------------


def evaluate_dataset_slice(
    dataset: list[dict[str, Any]],
    fault_type_list: list[str],
    db: Any,
    label: str = "Primary Benchmark",
) -> dict[str, Any]:
    """Evaluate 4 methods on a dataset slice."""
    methods = ["random", "always_decision", "deterministic_diff", "llm_investigator"]
    top1_hits = {m: 0 for m in methods}
    top3_hits = {m: 0 for m in methods}
    top1_per_fault = {ft: {m: 0 for m in methods} for ft in fault_type_list}
    top3_per_fault = {ft: {m: 0 for m in methods} for ft in fault_type_list}
    runs_per_fault = {ft: 0 for ft in fault_type_list}

    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_latency_ms = 0.0
    investigator_calls = 0
    parse_failures = 0
    misdiagnosed: list[dict[str, Any]] = []

    total_runs = len(dataset)
    print(f"\n--- Evaluating {label} ({total_runs} runs) ---")

    for i, item in enumerate(dataset, start=1):
        trace_id = item["trace_id"]
        ft = item["fault_type"]
        gt_step = item["ground_truth_step"]
        gt_node = item["ground_truth_node"]
        cust_id = item["customer_id"]
        cust_name = item.get("customer_name", "Customer")
        ord_id = item["order_id"]
        true_days_old = item.get("true_days_old", 3)
        amt = item.get("amount", 50.0)
        runs_per_fault[ft] += 1

        steps = db.query(Step).filter(Step.trace_id == trace_id).order_by(Step.index).all()
        step_indices = [s.index for s in steps]
        step_nodes = {s.index: s.node_name for s in steps}

        # 1. Random Step
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

        # 2. Always Decision Step (llm_decide -> 4)
        m2_top1 = 4
        m2_top3 = [4, 3, 2]
        if m2_top1 == gt_step:
            top1_hits["always_decision"] += 1
            top1_per_fault[ft]["always_decision"] += 1
        if gt_step in m2_top3:
            top3_hits["always_decision"] += 1
            top3_per_fault[ft]["always_decision"] += 1

        # 3. Deterministic Diff vs Clean Reference
        clean_tid = get_clean_reference_trace_id(
            customer_id=cust_id,
            customer_name=cust_name,
            order_id=ord_id,
            true_days_old=true_days_old,
            amount=amt,
            db_session=db,
        )
        diff_res = diff_traces(clean_tid, trace_id, db=db)
        if diff_res.first_divergent_step is not None:
            m3_top1 = diff_res.first_divergent_step
        else:
            m3_top1 = step_indices[-1] if step_indices else 1

        div_step_set = [m3_top1]
        for sd in diff_res.steps:
            if sd.diverged and sd.step_index not in div_step_set:
                div_step_set.append(sd.step_index)
        for s_idx in step_indices:
            if s_idx not in div_step_set:
                div_step_set.append(s_idx)
        m3_top3 = div_step_set[:3]

        if m3_top1 == gt_step:
            top1_hits["deterministic_diff"] += 1
            top1_per_fault[ft]["deterministic_diff"] += 1
        if gt_step in m3_top3:
            top3_hits["deterministic_diff"] += 1
            top3_per_fault[ft]["deterministic_diff"] += 1

        # 4. LLM Investigator with Frozen Generic Prompt
        # Check if already investigated with current prompt hash
        inv_record = (
            db.query(Investigation)
            .filter(
                Investigation.trace_id == trace_id,
                Investigation.summary.like(f"[{INVESTIGATOR_PROMPT_HASH}]%"),
            )
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
                    .filter(
                        Investigation.trace_id == trace_id,
                        Investigation.summary.like(f"[{INVESTIGATOR_PROMPT_HASH}]%"),
                    )
                    .order_by(Investigation.id.desc())
                    .first()
                )
                p_tok = inv_rec.prompt_tokens if inv_rec else 0
                c_tok = inv_rec.completion_tokens if inv_rec else 0
                lat = inv_rec.latency_ms if inv_rec else 0.0
            except Exception as e:
                logger.error("Investigation failed on trace %d: %s", trace_id, e)
                parse_failures += 1
                ranked = []
                p_tok = 0
                c_tok = 0
                lat = 0.0
                summary_text = f"Investigation error: {e}"

        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        total_latency_ms += lat
        investigator_calls += 1

        m4_top1 = ranked[0].step_index if ranked else -1
        m4_top3 = [r.step_index for r in ranked[:3]]

        top1_match = m4_top1 == gt_step
        top3_match = gt_step in m4_top3

        if top1_match:
            top1_hits["llm_investigator"] += 1
            top1_per_fault[ft]["llm_investigator"] += 1
        else:
            hypo_top1 = ranked[0].hypothesis if ranked else "N/A"
            conf_top1 = ranked[0].confidence if ranked else 0.0
            cited_evs = ranked[0].cited_event_ids if ranked else []
            pred_node = step_nodes.get(m4_top1, "unknown")
            misdiagnosed.append(
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
            f"GT: {gt_step} ({gt_node}) | "
            f"Diff: {m3_top1} {'PASS' if m3_top1 == gt_step else 'FAIL'} | "
            f"LLM: {m4_top1} {'PASS' if top1_match else 'FAIL'} (top3: {'PASS' if top3_match else 'FAIL'})"
        )

    # Compute Wilson metrics
    metrics: dict[str, Any] = {}
    for m in methods:
        k1 = top1_hits[m]
        k3 = top3_hits[m]
        low1, high1 = wilson_score_interval(k1, total_runs)
        low3, high3 = wilson_score_interval(k3, total_runs)
        metrics[m] = {
            "top1_acc": k1 / total_runs if total_runs else 0.0,
            "top1_ci": (low1, high1),
            "top3_acc": k3 / total_runs if total_runs else 0.0,
            "top3_ci": (low3, high3),
            "top1_hits": k1,
            "top3_hits": k3,
        }

    return {
        "total_runs": total_runs,
        "metrics": metrics,
        "top1_per_fault": top1_per_fault,
        "top3_per_fault": top3_per_fault,
        "runs_per_fault": runs_per_fault,
        "misdiagnosed": misdiagnosed,
        "mean_tokens": (total_prompt_tokens + total_completion_tokens) / max(1, investigator_calls),
        "mean_latency": total_latency_ms / max(1, investigator_calls),
        "parse_fail_rate": (parse_failures / max(1, investigator_calls)) * 100,
    }


# ---------------------------------------------------------------------------
# Main Evaluation Function
# ---------------------------------------------------------------------------


def run_evaluation(
    dataset_path: Path = DATASET_PATH,
    held_out_path: Path | None = None,
    results_path: Path = RESULTS_PATH,
) -> dict[str, Any]:
    """Execute evaluation and generate complete evals/results.md."""
    init_db()

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file {dataset_path} not found.")

    with open(dataset_path, encoding="utf-8") as f:
        primary_dataset: list[dict[str, Any]] = json.load(f)

    held_out_dataset: list[dict[str, Any]] = []
    if held_out_path and held_out_path.exists():
        try:
            with open(held_out_path, encoding="utf-8") as f:
                held_out_dataset = json.load(f)
        except Exception:
            held_out_dataset = []

    db = SessionLocal()

    eval_primary = evaluate_dataset_slice(
        primary_dataset, FAULT_TYPES, db, label="Primary Benchmark (40 runs)"
    )

    eval_held_out = None
    if held_out_dataset:
        eval_held_out = evaluate_dataset_slice(
            held_out_dataset, HELD_OUT_FAULT_TYPES, db, label="Held-Out Set (16 runs)"
        )

    db.close()

    model_name = clean_env_var(os.environ.get("LLM_MODEL"), "openai/gpt-oss-120b")
    provider_name = clean_env_var(os.environ.get("LLM_PROVIDER"), "stub")

    # Render Markdown Document
    md: list[str] = [
        "# AgentReplay Benchmark Evaluation Results (Phase 5)",
        "",
        f"**Date:** {datetime.date.today().isoformat()}",
        "**Dataset:** 40 failing agent executions across 5 fault injection types (8 runs each)",
        f"**Evaluated Model:** {model_name}",
        f"**Provider:** {provider_name}",
        f"**Investigator Prompt Hash:** `{INVESTIGATOR_PROMPT_HASH}`",
        "",
        "> **Prompt Evolution Note:** The investigator prompt was updated after the initial benchmark run to remove Point 4 "
        "and all fault-specific hints ('30 days', 'empty/null', 'outdated policy', 'corrupted data'). "
        "Below, results for both the original leaky prompt and the frozen generic prompt are reported side by side.",
        "",
        "## 1. Summary Comparison",
        "",
        "> **Note on Top-3 Baseline:** For a 5-step agent graph, a random uniform choice of 3 steps achieves a theoretical "
        "baseline Top-3 accuracy of 3 / 5 = **60.0%**. Top-1 accuracy is therefore the primary discriminating headline metric.",
        "",
        "| Method | Top-1 Accuracy (95% CI) | Top-3 Accuracy (95% CI) | Mean Tokens | Mean Latency | Parse Failures |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ]

    method_display = {
        "random": "1. Random Step",
        "always_decision": "2. Always Decision Step (llm_decide)",
        "deterministic_diff": "3. Deterministic Diff vs Clean Reference",
        "llm_investigator": f"4. LLM Investigator (frozen prompt: `{INVESTIGATOR_PROMPT_HASH}`)",
    }

    # Add baselines and new frozen investigator row
    for m in ["random", "always_decision", "deterministic_diff"]:
        met = eval_primary["metrics"][m]
        ci1 = format_ci(met["top1_acc"], *met["top1_ci"])
        ci3 = format_ci(met["top3_acc"], *met["top3_ci"])
        md.append(f"| **{method_display[m]}** | {ci1} | {ci3} | 0 | 0 ms | 0.0% |")

    # Historical row: leaky prompt
    md.append(
        "| *4a. LLM Investigator (leaky prompt - point 4 hints)* | *80.0% [65.2%, 89.5%]* | *92.5% [80.1%, 97.4%]* | 2707.8 | 18280.6 ms | 0.0% |"
    )

    # Current frozen prompt row
    m_inv = eval_primary["metrics"]["llm_investigator"]
    ci1_inv = format_ci(m_inv["top1_acc"], *m_inv["top1_ci"])
    ci3_inv = format_ci(m_inv["top3_acc"], *m_inv["top3_ci"])
    tok_str = f"{eval_primary['mean_tokens']:.1f}"
    lat_str = f"{eval_primary['mean_latency']:.1f} ms"
    pf_str = f"{eval_primary['parse_fail_rate']:.1f}%"
    md.append(
        f"| **{method_display['llm_investigator']}** | **{ci1_inv}** | {ci3_inv} | {tok_str} | {lat_str} | {pf_str} |"
    )

    # Section 2: Per-Fault Breakdown
    md.extend(
        [
            "",
            "## 2. Per-Fault Breakdown (Primary Benchmark)",
            "",
            "Top-1 and Top-3 accuracy with Wilson 95% confidence intervals per fault type (n=8 each):",
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
        "duplicate_refund": "Step 5 (`issue_refund`)",
        "wrong_amount": "Step 5 (`issue_refund`)",
    }

    for ft in FAULT_TYPES:
        n_ft = eval_primary["runs_per_fault"][ft]
        gt_lbl = fault_gt_labels[ft]

        def _cell(m: str) -> str:
            t1 = eval_primary["top1_per_fault"][ft][m]
            t3 = eval_primary["top3_per_fault"][ft][m]
            l1, h1 = wilson_score_interval(t1, n_ft)
            return f"{t1}/{n_ft} ({t1/n_ft*100:.0f}%) [{l1*100:.0f}%, {h1*100:.0f}%] / {t3}/{n_ft} ({t3/n_ft*100:.0f}%)"

        md.append(
            f"| `{ft}` | {gt_lbl} | {_cell('random')} | {_cell('always_decision')} | {_cell('deterministic_diff')} | {_cell('llm_investigator')} |"
        )

    # Section 3: Held-out Evaluation
    if eval_held_out:
        md.extend(
            [
                "",
                "## 3. Held-Out Evaluation (New Fault Types: Never Mentioned in Prompt)",
                "",
                f"Evaluated on {eval_held_out['total_runs']} held-out runs across 2 new fault types (8 runs each, seed=1337):",
                "- `duplicate_refund`: `issue_refund` is invoked twice for the same approved order (truth: Step 5 `issue_refund`).",
                "- `wrong_amount`: `issue_refund` is invoked with a modified amount differing from order amount (truth: Step 5 `issue_refund`).",
                "",
                "| Method | Top-1 Accuracy (95% CI) | Top-3 Accuracy (95% CI) | Mean Tokens | Mean Latency |",
                "| :--- | :---: | :---: | :---: | :---: |",
            ]
        )
        for m in ["random", "always_decision", "deterministic_diff", "llm_investigator"]:
            met = eval_held_out["metrics"][m]
            ci1 = format_ci(met["top1_acc"], *met["top1_ci"])
            ci3 = format_ci(met["top3_acc"], *met["top3_ci"])
            t_tok = f"{eval_held_out['mean_tokens']:.1f}" if m == "llm_investigator" else "0"
            t_lat = f"{eval_held_out['mean_latency']:.1f} ms" if m == "llm_investigator" else "0 ms"
            md.append(f"| **{method_display[m]}** | {ci1} | {ci3} | {t_tok} | {t_lat} |")

        md.extend(
            [
                "",
                "| Held-Out Fault Type | Ground Truth | Deterministic Diff (Top-1 / Top-3) | LLM Investigator (Top-1 / Top-3) |",
                "| :--- | :--- | :---: | :---: |",
            ]
        )
        for ft in HELD_OUT_FAULT_TYPES:
            n_ft = eval_held_out["runs_per_fault"][ft]
            gt_lbl = fault_gt_labels[ft]

            def _cell_ho(m: str) -> str:
                t1 = eval_held_out["top1_per_fault"][ft][m]
                t3 = eval_held_out["top3_per_fault"][ft][m]
                l1, h1 = wilson_score_interval(t1, n_ft)
                return f"{t1}/{n_ft} ({t1/n_ft*100:.0f}%) [{l1*100:.0f}%, {h1*100:.0f}%] / {t3}/{n_ft} ({t3/n_ft*100:.0f}%)"

            md.append(f"| `{ft}` | {gt_lbl} | {_cell_ho('deterministic_diff')} | {_cell_ho('llm_investigator')} |")

    # Section 4: Where It Fails
    misdiagnosed_all = list(eval_primary["misdiagnosed"])
    if eval_held_out:
        misdiagnosed_all.extend(eval_held_out["misdiagnosed"])

    md.extend(
        [
            "",
            "## 4. Where It Fails (Case-by-Case Diagnostic Log)",
            "",
            f"Total misdiagnosed cases across all benchmark evaluations: **{len(misdiagnosed_all)}**.",
            "",
        ]
    )

    if not misdiagnosed_all:
        md.append("No misdiagnosed runs occurred in this evaluation.")
    else:
        for idx, fail in enumerate(misdiagnosed_all, start=1):
            md.extend(
                [
                    f"### Case #{idx}: Trace #{fail['trace_id']} (`{fail['fault_type']}`)",
                    f"- **Ground Truth:** Step {fail['ground_truth_step']} (`{fail['ground_truth_node']}`)",
                    f"- **Investigator Predicted (Rank #1):** Step {fail['predicted_step']} (`{fail['predicted_node']}`) [Confidence: {fail['confidence']:.2f}]",
                    f"- **Cited Events:** {fail['cited_event_ids']}",
                    f"- **Investigator Summary:** {fail['summary']}",
                    f"- **Investigator Hypothesis:** {fail['hypothesis']}",
                    f"- **Diagnostic Divergence:** Investigator attributed root cause to Step {fail['predicted_step']} instead of injected Step {fail['ground_truth_step']}.",
                    "",
                ]
            )

    # Section 5: Limitations
    md.extend(
        [
            "## 5. Limitations",
            "",
            "1. **Synthetic Fault Scenarios:** Faults are synthetically injected into a five-step customer refund workflow. While realistic, real-world multi-agent cascades feature deeper execution DAGs.",
            "2. **Sample Size:** Each fault type is evaluated on n=8 verified failing runs (40 primary benchmark runs, 16 held-out runs; 56 total). Wilson 95% confidence intervals reflect this sample size.",
            "3. **Model Symmetry:** Both the agent's decision logic and the investigator run on the same model (`openai/gpt-oss-120b`). Evaluation against cross-model pairings remains future work.",
            "4. **Latency Measurement:** Reported mean latency includes HTTP 429 exponential backoff intervals enforced by the API rate limits during benchmark execution.",
            "5. **Exclusion of Non-Failing Injections:** Candidate runs where injected faults did not lead to an erroneous outcome under the deterministic oracle were excluded, ensuring only true failures are localized.",
            "6. **Inherent Information Limitation (`wrong_order_date`):** In `wrong_order_date`, the tool returned syntactically valid order data with a falsified timestamp. Because the trace contains no contradictory metadata, an isolated investigator evaluating only internal events cannot deduce that the tool return was corrupted without external reference data.",
        ]
    )

    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    print("\n" + "=" * 80)
    print(f"Evaluation complete! Results successfully written to {results_path}")
    print("=" * 80)
    for line in md[18:27]:
        print(line)
    print("=" * 80)

    return {
        "total_runs": eval_primary["total_runs"],
        "primary": eval_primary,
        "held_out": eval_held_out,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AgentReplay benchmark evaluation.")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="Path to primary dataset.json")
    parser.add_argument("--held-out", type=Path, default=HELD_OUT_PATH, help="Path to held_out_dataset.json")
    parser.add_argument("--out", type=Path, default=RESULTS_PATH, help="Output path for results.md")
    args = parser.parse_args()

    run_evaluation(dataset_path=args.dataset, held_out_path=args.held_out, results_path=args.out)
