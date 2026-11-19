# AgentReplay Benchmark Evaluation Results (Phase 5)

**Date:** 2026-11-19
**Dataset:** 40 failing agent executions across 5 fault injection types (8 runs each)
**Evaluated Model:** openai/gpt-oss-120b
**Provider:** openai_compatible

## 1. Summary Comparison

| Method | Top-1 Accuracy (95% CI) | Top-3 Accuracy (95% CI) | Mean Tokens | Mean Latency | Parse Failures |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **1. Random Step** | 17.5% [8.7%, 31.9%] | 60.0% [44.6%, 73.7%] | 0 | 0 ms | 0.0% |
| **2. Always Decision Step (llm_decide)** | 20.0% [10.5%, 34.8%] | 80.0% [65.2%, 89.5%] | 0 | 0 ms | 0.0% |
| **3. Deterministic Diff vs Clean Ref** | 20.0% [10.5%, 34.8%] | 80.0% [65.2%, 89.5%] | 0 | 0 ms | 0.0% |
| **4. LLM Investigator** | 80.0% [65.2%, 89.5%] | 92.5% [80.1%, 97.4%] | 2707.8 | 18280.6 ms | 0.0% |

## 2. Per-Fault Breakdown

Top-1 and Top-3 accuracy across the 5 injected fault types:

| Fault Type | Ground Truth | Random (Top-1 / Top-3) | Always Decision (Top-1 / Top-3) | Deterministic Diff (Top-1 / Top-3) | LLM Investigator (Top-1 / Top-3) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| `stale_policy` | Step 3 (`read_refund_policy`) | 2/8 (25%) / 6/8 (75%) | 0/8 (0%) / 8/8 (100%) | 0/8 (0%) / 8/8 (100%) | 8/8 (100%) / 8/8 (100%) |
| `wrong_order_date` | Step 2 (`get_orders`) | 2/8 (25%) / 6/8 (75%) | 0/8 (0%) / 8/8 (100%) | 0/8 (0%) / 8/8 (100%) | 0/8 (0%) / 5/8 (62%) |
| `prompt_bug` | Step 4 (`llm_decide`) | 1/8 (12%) / 5/8 (62%) | 8/8 (100%) / 8/8 (100%) | 0/8 (0%) / 0/8 (0%) | 8/8 (100%) / 8/8 (100%) |
| `missing_tool_result` | Step 3 (`read_refund_policy`) | 1/8 (12%) / 5/8 (62%) | 0/8 (0%) / 8/8 (100%) | 0/8 (0%) / 8/8 (100%) | 8/8 (100%) / 8/8 (100%) |
| `wrong_customer_data` | Step 1 (`get_customer`) | 1/8 (12%) / 2/8 (25%) | 0/8 (0%) / 0/8 (0%) | 8/8 (100%) / 8/8 (100%) | 8/8 (100%) / 8/8 (100%) |

## 3. Where It Fails (Failure Analysis)

Total misdiagnosed runs by the LLM Investigator: **8** / 40 (20.0% error rate).

### Failure #1: Trace #160 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 1 (`get_customer`) [Confidence: 0.00]
- **Cited Events:** [780]
- **Investigator Summary:** After reviewing the trace, the agent’s behavior aligns with the policy: the order is 2 days old (≤ 7 days), the policy correctly states a 7‑day window, the LLM decision to approve is appropriate, and the refund was issued for the correct order and customer. No tool failure, missing data, or policy mismatch is observed. Consequently, no root‑cause error was introduced in any step.
- **Investigator Hypothesis:** Tool returned correct customer data; no error introduced.

**Root-Cause Diagnosis Analysis:**
> **Why it failed:** The investigator observed that `llm_decide` approved the refund based on `DAYS_OLD`, or blamed `read_refund_policy`, rather than noticing that `get_orders` emitted an inaccurate order date. Without an external ground truth database of order creation dates, an LLM reading only the trace cannot easily distinguish a falsified tool return from a valid recent order.

### Failure #2: Trace #161 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 1 (`get_customer`) [Confidence: 0.02]
- **Cited Events:** [785]
- **Investigator Summary:** After thorough examination, all steps in the trace adhere to the policy invariants: the order is 3 days old (within the 7‑day window), the correct customer is used, the policy was read accurately, and the refund was issued for the original order. No erroneous data, missing information, or flawed instruction was introduced, so there is no identifiable root‑cause of failure in this execution.
- **Investigator Hypothesis:** Customer retrieval succeeded and returned correct data; unlikely to be the cause.

**Root-Cause Diagnosis Analysis:**
> **Why it failed:** The investigator observed that `llm_decide` approved the refund based on `DAYS_OLD`, or blamed `read_refund_policy`, rather than noticing that `get_orders` emitted an inaccurate order date. Without an external ground truth database of order creation dates, an LLM reading only the trace cannot easily distinguish a falsified tool return from a valid recent order.

### Failure #3: Trace #162 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 4 (`llm_decide`) [Confidence: 0.15]
- **Cited Events:** [793]
- **Investigator Summary:** After reviewing the trace, the agent correctly retrieved the customer, fetched the order, read the up‑to‑date 7‑day refund policy, calculated DAYS_OLD = 4, approved the refund per policy, and issued the refund for the correct order and customer. No step introduced bad, missing, or outdated data, and the decision aligns with the policy invariant. Therefore the unexpected outcome is not attributable to any fault in the executed steps.
- **Investigator Hypothesis:** The LLM could have mis‑interpreted the policy or output malformed JSON, but it correctly returned an 'approve' decision and a valid reason, so it is unlikely the root cause.

**Root-Cause Diagnosis Analysis:**
> **Why it failed:** The investigator observed that `llm_decide` approved the refund based on `DAYS_OLD`, or blamed `read_refund_policy`, rather than noticing that `get_orders` emitted an inaccurate order date. Without an external ground truth database of order creation dates, an LLM reading only the trace cannot easily distinguish a falsified tool return from a valid recent order.

### Failure #4: Trace #163 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 4 (`llm_decide`) [Confidence: 0.70]
- **Cited Events:** [798]
- **Investigator Summary:** The likely root‑cause is that the LLM decision (Step 4) relied on a possibly incorrect or fabricated `days_old` value (set to 2) rather than a verified calculation from the order date. This premature decision caused the refund to be issued without proper validation of the actual age of the order, violating the policy invariant that refunds must be denied for orders older than 7 days. Subsequent steps simply acted on that decision.
- **Investigator Hypothesis:** The LLM made the approve/deny decision using a `days_old` field that was not derived from the order date in this trace; if that value is wrong, the decision violates the 7‑day rule.

**Root-Cause Diagnosis Analysis:**
> **Why it failed:** The investigator observed that `llm_decide` approved the refund based on `DAYS_OLD`, or blamed `read_refund_policy`, rather than noticing that `get_orders` emitted an inaccurate order date. Without an external ground truth database of order creation dates, an LLM reading only the trace cannot easily distinguish a falsified tool return from a valid recent order.

### Failure #5: Trace #164 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 4 (`llm_decide`) [Confidence: 0.92]
- **Cited Events:** [803]
- **Investigator Summary:** The agent approved a refund based on a days_old value that was supplied without verification. No earlier step introduced incorrect policy or order data, so the first point where faulty or unchecked information could cause a wrong decision is the LLM decision step that trusted the unvalidated days_old field.
- **Investigator Hypothesis:** The LLM decision used a pre‑populated 'days_old' value (3) without recomputing or validating it against the order_date, so if that value were wrong the decision would be incorrect.

**Root-Cause Diagnosis Analysis:**
> **Why it failed:** The investigator observed that `llm_decide` approved the refund based on `DAYS_OLD`, or blamed `read_refund_policy`, rather than noticing that `get_orders` emitted an inaccurate order date. Without an external ground truth database of order creation dates, an LLM reading only the trace cannot easily distinguish a falsified tool return from a valid recent order.

### Failure #6: Trace #165 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 4 (`llm_decide`) [Confidence: 0.15]
- **Cited Events:** [808]
- **Investigator Summary:** The trace shows a correct execution: the order is 4 days old, within the 7‑day refund window, the policy was read accurately, the LLM approved the refund, and the refund was issued for the correct order and customer. No step introduced bad or missing data, nor did any tool fail. Therefore there is no root‑cause error in this trace.
- **Investigator Hypothesis:** If the outcome were unexpected, the decision logic could be at fault, but the decision to approve matches the policy and days_old, so unlikely.

**Root-Cause Diagnosis Analysis:**
> **Why it failed:** The investigator observed that `llm_decide` approved the refund based on `DAYS_OLD`, or blamed `read_refund_policy`, rather than noticing that `get_orders` emitted an inaccurate order date. Without an external ground truth database of order creation dates, an LLM reading only the trace cannot easily distinguish a falsified tool return from a valid recent order.

### Failure #7: Trace #166 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step -1 (`unknown`) [Confidence: 0.00]
- **Cited Events:** []
- **Investigator Summary:** The execution trace complies with all policy invariants: the order is 2 days old (≤7), the policy correctly states a 7‑day window, the LLM decision to approve is appropriate, and the refund is issued for the correct order/customer. No erroneous, missing, or outdated data was introduced, so there is no root‑cause step responsible for a failure.
- **Investigator Hypothesis:** N/A

**Root-Cause Diagnosis Analysis:**
> **Why it failed:** The investigator observed that `llm_decide` approved the refund based on `DAYS_OLD`, or blamed `read_refund_policy`, rather than noticing that `get_orders` emitted an inaccurate order date. Without an external ground truth database of order creation dates, an LLM reading only the trace cannot easily distinguish a falsified tool return from a valid recent order.

### Failure #8: Trace #167 (`wrong_order_date`)
- **Ground Truth:** Step 2 (`get_orders`)
- **Investigator Predicted (Rank #1):** Step 4 (`llm_decide`) [Confidence: 0.60]
- **Cited Events:** [818]
- **Investigator Summary:** The trace shows the agent approved a refund for an order that is 3 days old, which complies with the 7‑day policy, and the refund was issued correctly. No explicit violation of the policy or tool failure is evident. If the outcome was considered unexpected, the most plausible hidden issue is an incorrect calculation of DAYS_OLD that was introduced before the LLM decision step, but the trace does not contain the step that performed that calculation.
- **Investigator Hypothesis:** The LLM decision relies on the DAYS_OLD value (3). If that value was miscomputed earlier (a step not shown), the approval could be based on faulty data, making this the likely point where the error manifested.

**Root-Cause Diagnosis Analysis:**
> **Why it failed:** The investigator observed that `llm_decide` approved the refund based on `DAYS_OLD`, or blamed `read_refund_policy`, rather than noticing that `get_orders` emitted an inaccurate order date. Without an external ground truth database of order creation dates, an LLM reading only the trace cannot easily distinguish a falsified tool return from a valid recent order.

