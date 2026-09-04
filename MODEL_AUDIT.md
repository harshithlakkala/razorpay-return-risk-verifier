# Model & Agent Audit

This document records what was tested, what was found, and what is *not*
claimed. It exists because the headline numbers in this project look
implausibly good, and the reason they look that way matters more than the
numbers themselves.

Everything below is reproducible from scripts in this repository.

---

## 1. The perfect scores are the dataset, not the model

The Random Forest scores 1.000 precision and 1.000 recall on 12,000 held-out
orders. So does Logistic Regression. So does Gradient Boosting. When three
model families with completely different inductive biases all achieve a
flawless score, that is not three good models — it is a dataset with
deterministic labels.

To confirm it, we fitted deliberately crippled decision trees against the
raw numeric features:

| Model | Precision | Recall |
|---|---|---|
| Depth-1 tree (a single `if`) | 1.0000 | 0.9992 |
| **Depth-2 tree (two `if`s)** | **1.0000** | **1.0000** |
| Random Forest, 200 estimators | 1.0000 | 1.0000 |

The depth-2 tree, printed in full:

```
if return_rate_pct <= 15.80:
    if customer_support_contacts <= 1.50 -> LEGITIMATE
    else                                 -> ABUSE
else                                     -> ABUSE
```

That is the dataset's label-generation rule, recovered exactly. A 200-tree
ensemble is doing nothing that two comparisons do not already do.

Feature importances agree: `return_rate_pct` alone carries 28.3% of the
model's importance, and the top three features carry 63.1%. The remaining 26
features are largely decorative — they cannot matter, because the labels do
not depend on them.

**What this means:** the model has not learned to detect fraud. It has learned
to invert a synthetic generator. The cost sweep's unusually wide zero-cost
"safe zone" (thresholds 0.17–0.68 all cost $0) is the same artefact viewed
from another angle — real fraud data has no such margin.

Reproduce: feature importances and the tree experiment are described above;
`model_comparison.py` produces the three-model table.

---

## 2. The decision policy was moved out of the prompt and into code

Originally the decision rules lived in the agent's system prompt, i.e. the
model was *asked* to apply them. A 7B local model does not reliably obey a
paragraph of instructions, and we observed exactly that failure:

- **Schema violations.** The model invented a tool argument
  (`abuse_probability` on `log_decision`), crashing a 100-order evaluation at
  order 21. Fixed by filtering tool arguments against each function's real
  signature before calling it.
- **Non-determinism.** Order `ORD2013623` was approved in several runs and
  flagged in another, on byte-identical input, because sampling ran at the
  default temperature (~0.8).

Both are unacceptable in a system that moves money. The rules now live in
`policy_decision()` in `risk_agent_local.py`, evaluated in Python:

1. `cross_merchant_pattern_detected` → `flag_for_review`, regardless of ML score
2. Damage/defect claimed **and** no photo evidence → `request_more_evidence`
3. `abuse_probability > 0.43` → `flag_for_review`
4. Otherwise → `approve`

The threshold of 0.43 is the centre of the range that minimised real
false-positive/false-negative dollar cost (`cost_threshold.py`), not a default.

The LLM still investigates, calls all five tools, and writes the
human-readable justification. It no longer holds the rule. Where the model's
proposed action disagrees with the policy, the policy wins and the override is
written into the audit log, so the record shows both what the model concluded
and what was actually done.

**This is the central engineering claim of the project** — not that an agent
was trained, but that the decision logic was made deterministic, testable, and
impossible for the model to silently violate.

---

## 3. Determinism was fixed and verified

All inference calls now run with `temperature=0, top_p=1, top_k=1, seed=42`,
pinned both per-call and baked into the custom model (`Modelfile.risk-agent`).

`verify_determinism.py` is the regression test. On the previously unstable
order:

```
Running ORD2013623 3 times with temperature=0...
  run 1: approve
  run 2: approve
  run 3: approve
DETERMINISTIC: all 3 runs returned 'approve'
```

`approve` is also the correct call — the order is legitimate, ML score 0.0016.

**Known limitation:** the *action* is reproducible; the free-text reasoning is
not byte-identical across runs (floating-point non-associativity in the
inference kernels). The decision and the figures it cites are stable; the
prose wording may vary slightly.

---

## 4. Full-set evaluation

Because the policy is enforced in code, the agent's final action is guaranteed
to equal the policy's action. That makes the policy evaluable across the
entire test set in seconds, rather than the ~10 days of local CPU inference a
genuine 12,000-order agent run would require (`evaluate_policy.py`).

Across all 12,000 held-out orders:

| Action | Count | Abuse | Legitimate |
|---|---|---|---|
| `approve` | 7,631 | 0 | 7,631 |
| `flag_for_review` | 3,517 | 3,517 | 0 |
| `request_more_evidence` | 852 | 71 | 781 |

On the 11,148 binary decisions: **precision 1.0000, recall 1.0000**, zero
false positives, zero missed abuse, **$0** total error cost.

Escalations are reported separately and never scored as binary
correct/incorrect — holding an order for evidence is not the same as deciding
it wrongly.

For context, an earlier agent-in-the-loop run over a 72-order sample (before
policy enforcement) scored 0.958 precision / 1.000 recall, with one legitimate
order wrongly flagged.

---

## 5. What is *not* claimed

**The 1.000 does not demonstrate fraud-detection ability.** It demonstrates
that the policy correctly reproduces a synthetic generator (see §1). On real
data, where abusive and legitimate behaviour overlap heavily, these numbers
would not survive.

**Rule 1 cannot fail on this data, by construction.**
`simulate_cross_merchant_data.py` plants shared refund destinations *only* on
orders already labelled abusive. The cross-merchant signal is therefore
perfectly correlated with abuse in the simulation, and cannot produce a false
positive. In production it certainly would: families share accounts, one
person pays for several people, resellers legitimately reuse a destination.
The rule's real-world precision is untested and should not be assumed high.

**Rule 2 is currently a blunt instrument.** Of the 852 orders escalated for
evidence, only 71 (8.3%) are actually abusive — meaning **781 legitimate
customers would be asked to supply a photo**. That is real friction traded for
caution. Narrowing the rule (for example, only escalating above a probability
floor) would reduce it substantially.

**No model was trained or fine-tuned.** The custom `risk-agent` model is stock
`qwen2.5` with a system prompt, five in-context worked examples, and pinned
sampling parameters. Zero gradient steps were taken; zero of the ~7.6 billion
parameters were updated. This is verifiable: the base `qwen2.5` and the
specialised `risk-agent` reference the **identical** 4,466 MB weight file
(`sha256:2bada8a745067700…`) in the local Ollama store — Ollama does not even
copy it, both manifests point at the same blob. The only difference between
them is a few kilobytes of system prompt, worked examples, and parameters.

**Evaluation is single-dataset.** No cross-dataset validation, adversarial
testing, or drift analysis was performed.

---

## 6. Reproducing this audit

```bash
python model_comparison.py       # three-model comparison incl. calibration
python cost_threshold.py         # threshold sweep by dollar cost
python evaluate_policy.py        # full 12,000-order policy evaluation
python verify_determinism.py     # determinism regression test
python merge_all_data.py         # join order/payment/network data
python generate_full_audit_report.py   # per-decision audit trail
```
