# AgentReplay Benchmark Evaluation Results (Phase 5)

**Date:** 2026-11-19
**Dataset:** 40 failing agent executions across 5 fault injection types (8 runs each)
**Evaluated Model:** openai/gpt-oss-120b
**Provider:** openai_compatible
**Investigator Prompt Hash:** `ff2cb21dea6d27d7`

> **Prompt Evolution Note:** The investigator prompt was updated after the initial benchmark run to remove Point 4 and all fault-specific hints ('30 days', 'empty/null', 'outdated policy', 'corrupted data'). Below, results for both the original leaky prompt and the frozen generic prompt are reported side by side.

## 1. Summary Comparison

> **Note on Top-3 Baseline:** For a 5-step agent graph, a random uniform choice of 3 steps achieves a theoretical baseline Top-3 accuracy of 3 / 5 = **60.0%**. Top-1 accuracy is therefore the primary discriminating headline metric.

| Method | Top-1 Accuracy (95% CI) | Top-3 Accuracy (95% CI) | Mean Tokens | Mean Latency | Parse Failures |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Random Step** | 17.5% [8.7%, 31.9%] | 60.0% [44.6%, 73.7%] | 0 | 0 ms | 0.0% |
| **2. Always Decision Step (llm_decide)** | 20.0% [10.5%, 34.8%] | 80.0% [65.2%, 89.5%] | 0 | 0 ms | 0.0% |
| **3. Deterministic Diff vs Clean Reference** | 67.5% [52.0%, 79.9%] | 100.0% [91.2%, 100.0%] | 0 | 0 ms | 0.0% |
| *4a. LLM Investigator (leaky prompt - point 4 hints)* | *80.0% [65.2%, 89.5%]* | *92.5% [80.1%, 97.4%]* | 2707.8 | 18280.6 ms | 0.0% |
| **4. LLM Investigator (frozen prompt: `ff2cb21dea6d27d7`)** | **82.5% [68.1%, 91.3%]** | 100.0% [91.2%, 100.0%] | 2677.7 | 181233.9 ms | 0.0% |

## 2. Per-Fault Breakdown (Primary Benchmark)

Top-1 and Top-3 accuracy with Wilson 95% confidence intervals per fault type (n=8 each):

| Fault Type | Ground Truth | Random (Top-1 / Top-3) | Always Decision (Top-1 / Top-3) | Deterministic Diff (Top-1 / Top-3) | LLM Investigator (Top-1 / Top-3) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `stale_policy` | Step 3 (`read_refund_policy`) | 2/8 (25%) [7%, 59%] / 6/8 (75%) | 0/8 (0%) [0%, 32%] / 8/8 (100%) | 8/8 (100%) [68%, 100%] / 8/8 (100%) | 8/8 (100%) [68%, 100%] / 8/8 (100%) |
| `wrong_order_date` | Step 2 (`get_orders`) | 2/8 (25%) [7%, 59%] / 6/8 (75%) | 0/8 (0%) [0%, 32%] / 8/8 (100%) | 8/8 (100%) [68%, 100%] / 8/8 (100%) | 1/8 (12%) [2%, 47%] / 8/8 (100%) |
| `prompt_bug` | Step 4 (`llm_decide`) | 1/8 (12%) [2%, 47%] / 5/8 (62%) | 8/8 (100%) [68%, 100%] / 8/8 (100%) | 0/8 (0%) [0%, 32%] / 8/8 (100%) | 8/8 (100%) [68%, 100%] / 8/8 (100%) |
| `missing_tool_result` | Step 3 (`read_refund_policy`) | 1/8 (12%) [2%, 47%] / 5/8 (62%) | 0/8 (0%) [0%, 32%] / 8/8 (100%) | 3/8 (38%) [14%, 69%] / 8/8 (100%) | 8/8 (100%) [68%, 100%] / 8/8 (100%) |
| `wrong_customer_data` | Step 1 (`get_customer`) | 1/8 (12%) [2%, 47%] / 2/8 (25%) | 0/8 (0%) [0%, 32%] / 0/8 (0%) | 8/8 (100%) [68%, 100%] / 8/8 (100%) | 8/8 (100%) [68%, 100%] / 8/8 (100%) |

## 3. Held-Out Evaluation (New Fault Types: Never Mentioned in Prompt)

Evaluated on 16 held-out runs across 2 new fault types (8 runs each, seed=1337):
- `duplicate_refund`: `issue_refund` is invoked twice for the same approved order (truth: Step 5 `issue_refund`).
- `wrong_amount`: `issue_refund` is invoked with a modified amount differing from order amount (truth: Step 5 `issue_refund`).

| Method | Top-1 Accuracy (95% CI) | Top-3 Accuracy (95% CI) | Mean Tokens | Mean Latency |
| :--- | :---: | :---: | :---: | :---: |
| **1. Random Step** | 37.5% [18.5%, 61.4%] | 75.0% [50.5%, 89.8%] | 0 | 0 ms |
| **2. Always Decision Step (llm_decide)** | 0.0% [0.0%, 19.4%] | 0.0% [0.0%, 19.4%] | 0 | 0 ms |
| **3. Deterministic Diff vs Clean Reference** | 0.0% [0.0%, 19.4%] | 0.0% [0.0%, 19.4%] | 0 | 0 ms |
| **4. LLM Investigator (frozen prompt: `ff2cb21dea6d27d7`)** | 100.0% [80.6%, 100.0%] | 100.0% [80.6%, 100.0%] | 2700.9 | 972635.9 ms |

| Held-Out Fault Type | Ground Truth | Deterministic Diff (Top-1 / Top-3) | LLM Investigator (Top-1 / Top-3) |
| :--- | :--- | :---: | :---: |
| `duplicate_refund` | Step 5 (`issue_refund`) | 0/8 (0%) [0%, 32%] / 0/8 (0%) | 8/8 (100%) [68%, 100%] / 8/8 (100%) |
| `wrong_amount` | Step 5 (`issue_refund`) | 0/8 (0%) [0%, 32%] / 0/8 (0%) | 8/8 (100%) [68%, 100%] / 8/8 (100%) |

## 4. Where It Fails (Case-by-Case Diagnostic Log)

Total misdiagnosed cases across all benchmark evaluations: **7**.

### Case #1: Trace #160 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 5 (`issue_refund`) [Confidence: 0.92]
- **Cited Events:** [784]
- **Investigator Summary:** [ff2cb21dea6d27d7] The agent’s final action (Step 5 – issue_refund) failed to satisfy the policy requirement that the refund be explicitly credited to the requesting customer. The tool call omitted any customer identifier or confirmation of crediting, meaning the refund may not be applied to the correct account despite the decision being correct. This omission is the first point where the policy‑compliant outcome breaks, making Step 5 the root‑cause.
- **Investigator Hypothesis:** The issue_refund call lacks a customer reference, violating invariant #3 that refunds must be credited to the requesting customer. This is the first step where the policy is not fully enforced.
- **Diagnostic Divergence:** Investigator attributed root cause to Step 5 instead of injected Step 2.

### Case #2: Trace #161 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 4 (`llm_decide`) [Confidence: 0.92]
- **Cited Events:** [788]
- **Investigator Summary:** [ff2cb21dea6d27d7] The agent approved and issued a refund based on a hard‑coded 'days_old' value supplied in the state rather than computing it from the order_date and the current date. Because the days‑old metric was not derived dynamically, the decision logic could be based on incorrect or stale data, making this the primary root‑cause of the policy‑compliance error.
- **Investigator Hypothesis:** Decision relied on a pre‑populated 'days_old' field (value 3) that was never computed from the order_date, so if the date calculation were wrong the policy check would be invalid.
- **Diagnostic Divergence:** Investigator attributed root cause to Step 4 instead of injected Step 2.

### Case #3: Trace #162 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 4 (`llm_decide`) [Confidence: 0.92]
- **Cited Events:** [793]
- **Investigator Summary:** [ff2cb21dea6d27d7] The agent approved a refund based on an incorrect or undocumented calculation of DAYS_OLD. The trace never shows a step that computes the order age, yet the LLM decision in step 4 trusts a pre‑populated days_old value of 4, leading to a potentially wrong approval. This missing/incorrect age computation is the root cause of the unexpected outcome.
- **Investigator Hypothesis:** The LLM made the approve/deny decision using a DAYS_OLD value that was never derived in the trace, so the decision rests on faulty or missing age data.
- **Diagnostic Divergence:** Investigator attributed root cause to Step 4 instead of injected Step 2.

### Case #4: Trace #163 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 4 (`llm_decide`) [Confidence: 0.78]
- **Cited Events:** [798]
- **Investigator Summary:** [ff2cb21dea6d27d7] The agent approved the refund based on a pre‑supplied 'days_old' value without explicitly calculating it from the order date. This reliance on assumed data introduces a potential policy violation if the days_old is incorrect, making the decision step the primary root‑cause of the unexpected outcome.
- **Investigator Hypothesis:** Decision was made using a hard‑coded 'days_old' (2) rather than computing it from the order_date, so an incorrect age could lead to policy breach.
- **Diagnostic Divergence:** Investigator attributed root cause to Step 4 instead of injected Step 2.

### Case #5: Trace #164 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 4 (`llm_decide`) [Confidence: 0.92]
- **Cited Events:** [803]
- **Investigator Summary:** [ff2cb21dea6d27d7] The agent approved the refund based on a hard‑coded days_old value (3) that was never computed from the order_date and current date. Because the step that should calculate DAYS_OLD is missing, the decision relies on potentially inaccurate data, making this the root cause of the policy‑compliance error.
- **Investigator Hypothesis:** Decision uses a pre‑populated days_old (3) that was never derived from the order_date, so the approval may be based on incorrect age information.
- **Diagnostic Divergence:** Investigator attributed root cause to Step 4 instead of injected Step 2.

### Case #6: Trace #165 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 4 (`llm_decide`) [Confidence: 0.92]
- **Cited Events:** [808]
- **Investigator Summary:** [ff2cb21dea6d27d7] The refund was approved despite the order being older than the 7‑day limit because the agent relied on an incorrect 'days_old' value (4). No step computed or validated the order age against the current date, so the LLM decision used bad data and approved the refund. The root cause is the missing/incorrect age calculation prior to the decision step.
- **Investigator Hypothesis:** The LLM decision used a fabricated 'days_old' of 4, leading to an approve decision; the error originates from the incorrect age data supplied to this step.
- **Diagnostic Divergence:** Investigator attributed root cause to Step 4 instead of injected Step 2.

### Case #7: Trace #167 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 4 (`llm_decide`) [Confidence: 0.86]
- **Cited Events:** [818]
- **Investigator Summary:** [ff2cb21dea6d27d7] The first logical flaw was introduced in Step 4 (llm_decide). The LLM made a refund approval decision based on a pre‑populated 'days_old' value without independently verifying that this value correctly reflects the order_date relative to the current date. This reliance on unchecked input creates a potential point of failure if the days_old data were inaccurate, making Step 4 the root‑cause of the unexpected outcome.
- **Investigator Hypothesis:** Decision relies on an assumed 'days_old' field that was never computed or validated, so any error in that field directly leads to an incorrect approve/deny outcome.
- **Diagnostic Divergence:** Investigator attributed root cause to Step 4 instead of injected Step 2.

## 5. Limitations

1. **Synthetic Fault Scenarios:** Faults are synthetically injected into a five-step customer refund workflow. While realistic, real-world multi-agent cascades feature deeper execution DAGs.
2. **Sample Size:** Each fault type is evaluated on n=8 verified failing runs (40 primary benchmark runs, 16 held-out runs; 56 total). Wilson 95% confidence intervals reflect this sample size.
3. **Model Symmetry:** Both the agent's decision logic and the investigator run on the same model (`openai/gpt-oss-120b`). Evaluation against cross-model pairings remains future work.
4. **Latency Measurement:** Reported mean latency includes HTTP 429 exponential backoff intervals enforced by the API rate limits during benchmark execution.
5. **Exclusion of Non-Failing Injections:** Candidate runs where injected faults did not lead to an erroneous outcome under the deterministic oracle were excluded, ensuring only true failures are localized.
6. **Inherent Information Limitation (`wrong_order_date`):** In `wrong_order_date`, the tool returned syntactically valid order data with a falsified timestamp. Because the trace contains no contradictory metadata, an isolated investigator evaluating only internal events cannot deduce that the tool return was corrupted without external reference data.
