# Payment-Layer Refund Fraud Verifier — Track 2: AI Risk Manager

## Why a payment gateway, not the merchant, owns this
Product-condition review ("is this shirt actually damaged?") belongs to the
merchant — Razorpay isn't the seller and shouldn't be looking at product
photos to adjudicate returns. But **refund fraud has a payment-layer
signature that no single merchant can see, and only the gateway sitting
across all of them can:**

- The same **bank account or UPI ID receiving refunds** ("refund destination")
  can be reused across *multiple, unrelated merchants* to farm illegitimate
  refunds — merchant A sees one suspicious refund; only Razorpay sees the same
  destination account collecting refunds from merchants A, B, and C in the
  same week.
- The same **payment instrument opening multiple customer accounts**
  (`multiple_accounts_flag`) is invisible to a merchant who only sees their
  own signup data, but visible instantly to whoever processes the underlying
  payment credential across the whole network.
- This is the same logic behind why Stripe built Radar as a payment-layer
  fraud product sold to merchants, rather than leaving every merchant to
  build their own — the gateway has cross-merchant visibility no single
  seller has.

**This is the core pitch: a refund-fraud verifier built on payment-layer
signals Razorpay uniquely has visibility into, not a product-condition judge.**

## The problem
Returns quietly eat into merchant margin — not because returns are bad, but
because a meaningful share of them are abuse: wardrobing, serial policy abuse,
and outright fraudulent refund claims that exploit payment-layer weaknesses
(refund-to-different-account, multi-account payment reuse) rather than product
condition.

## What this is
An **agentic verifier**, not just a classifier. For every return/refund
request, the agent investigates using tools (order details, customer history,
an ML-based risk score built primarily on payment and account signals),
reasons about what it finds, and decides one of two actions — approve or flag
for human review — logging its reasoning to an auditable trail every time.

## Architecture
1. **Risk model** (`train_model.py`) — a Random Forest trained on 48,000
   historical returns, predicting probability of abuse. The highest-signal
   features are payment/account-layer, not product-layer: return rate,
   dispute history, account age, refund-to-different-account and
   multiple-accounts flags — exactly the signals a gateway, not a merchant,
   is positioned to observe at scale.
2. **Agent layer** (`risk_agent_local.py` / `risk_agent.py`) — Claude (or a free
   local model via Ollama) is given 4 tools:
   - get_order_details
   - get_customer_history
   - get_risk_score (calls the trained model)
   - log_decision (writes the final call + reasoning to an audit log)
   The agent decides *how* to investigate each order, not just accept a single
   number — e.g. it can override a high-risk score when context clearly
   explains it (a long-tenured, low-return customer with one defective item),
   which a raw threshold rule cannot do.
3. **Guardrail:** the agent is instructed to lean toward flag_for_review above
   0.5 abuse probability unless context clearly justifies otherwise — a
   deliberate human-in-the-loop safety net, not full autonomy.

## Cross-merchant fraud pattern detection (the payment-layer signal, made real)
This is the feature that makes the "only Razorpay can see this" claim
concrete in code, not just prose. A new tool, `check_cross_merchant_pattern`,
checks whether an order's refund destination account also receives refunds
under *other, unrelated merchants* — a pattern that is structurally invisible
to any single merchant (they only see their own transactions) but directly
visible to a payment gateway sitting across all of them.

**Data note:** the real Kaggle dataset is single-merchant and has no merchant
ID or actual refund-account field (only a `refund_to_different_account`
true/false flag). To demonstrate this capability, we built
`simulate_cross_merchant_data.py`, which assigns the real 12,000 test orders
across 20 simulated merchants and deliberately plants "refund farming rings"
— shared destination accounts reused across 2-5 merchants for a subset of
abuse orders — exactly the pattern this tool is built to catch. This is
clearly a simulation of the architecture, not a claim that real fraud rings
exist in this specific Kaggle dataset.

Result: **717 of 12,000 orders (6.0%) were tied to a refund account shared
across multiple merchants** in the simulation, across 30 distinct ring
accounts spanning 2-5 merchants each.

The agent's guardrail was updated accordingly: if
`cross_merchant_pattern_detected=true`, the agent flags for review
*regardless of the ML probability score* — because this signal source is
categorically different (network visibility) from the order/customer-history
features the ML model was trained on, and should not be diluted by them.

## Cost-sensitive threshold tuning (not just accuracy)
A fixed 0.5 decision threshold treats a false positive (wrongly flagging a
legitimate return) and a false negative (missing real abuse) as equally bad.
They aren't — a false positive costs roughly the order value in customer
friction/lost trust, while a false negative costs the refund amount lost to
abuse. We swept every threshold from 0.01 to 0.99 and computed actual dollar
cost at each, rather than optimizing for accuracy or F1.

Findings on the held-out test set (12,000 orders):
- A careless low threshold (0.01) costs **$209,045** in false-positive
  friction alone — a concrete illustration of why threshold choice isn't
  a neutral hyperparameter.
- The **zero-cost "safe zone" spans 0.17 to 0.68** — a wide, 51-point margin,
  which reflects strong class separation in this model rather than a
  knife-edge decision boundary. (Consistent with the earlier honest caveat
  that this synthetic benchmark has cleanly separated classes by construction.)
- The cost curve is **asymmetric in shape, not just in per-error cost**: the
  false-positive side rises steeply (many more legitimate than abusive orders
  in the data, so a bad low threshold catches large volumes fast), while the
  false-negative side rises more gradually as the threshold climbs too high.
- Full sweep and code: `cost_threshold.py` / `threshold_sweep_results.csv`

## Guardrails enforced in code, not in the prompt
The decision rules originally lived in the agent's system prompt — the model
was *asked* to apply them. A 7B local model does not reliably obey a paragraph
of instructions, and it didn't: it invented a tool argument that crashed a
100-order evaluation partway through, and it returned **opposite verdicts on
byte-identical input** (order ORD2013623 was approved in some runs and flagged
in another) because sampling ran at the default temperature.

Neither is acceptable in a system that moves money. So the rules were moved
into `policy_decision()` and are now evaluated in Python. The agent still
investigates, calls all five tools, and writes the human-readable
justification — it no longer holds the rule. Where the model's proposed action
disagrees with the policy, the policy wins and the override is recorded in the
audit log, so the record shows both what the model concluded and what was
actually done.

All inference also runs at `temperature=0` with a fixed seed, verified by a
regression test (`verify_determinism.py`) on the order that was previously
unstable.

**This is the core engineering claim: not that an agent was trained, but that
the decision logic was made deterministic, testable, and impossible for the
model to silently violate.**

## Results (held-out test set, never seen during training)
Because the policy is enforced in code, the agent's final action provably
equals the policy's action — so it can be evaluated across the *entire* test
set in seconds, rather than the ~10 days of local CPU inference a genuine
12,000-order agent run would need (`evaluate_policy.py`).

Across all 12,000 held-out orders:

| Action | Count | Abuse | Legitimate |
|---|---|---|---|
| `approve` | 7,631 | 0 | 7,631 |
| `flag_for_review` | 3,517 | 3,517 | 0 |
| `request_more_evidence` | 852 | 71 | 781 |

- **Precision 1.000, recall 1.000** on the 11,148 binary decisions
- **Zero** false positives, **zero** missed abuse, **$0** total error cost
- Escalations reported separately, never scored as binary correct/incorrect —
  holding an order for evidence is not the same as deciding it wrongly

For reference, an earlier agent-in-the-loop run over a 72-order sample, before
policy enforcement, scored 0.958 precision / 1.000 recall.

**These numbers should not be read as fraud-detection skill.** A two-line
decision tree reproduces the ML model perfectly on this data, which means the
benchmark's labels are generated by a simple rule and the policy is inverting
that rule rather than detecting fraud. The full analysis — including which
results are artefacts of the simulation and which trade-offs are real — is in
[`MODEL_AUDIT.md`](MODEL_AUDIT.md). It is deliberately unflattering, and it is
the document worth reading first.

## Honest caveats
Full detail in [`MODEL_AUDIT.md`](MODEL_AUDIT.md); the short version:

- **The benchmark is solvable by a two-line rule.** A depth-2 decision tree
  reproduces the Random Forest perfectly on all 12,000 held-out orders:
  `return_rate_pct > 15.8` or `customer_support_contacts > 1.5` → abuse. The
  labels are synthetically generated, so a perfect score means the policy
  inverted the generator, not that it detects fraud. Real deployment would
  need continuous retraining as genuine abuse overlaps far more with
  legitimate behaviour.
- **The cross-merchant rule cannot fail here, by construction.**
  `simulate_cross_merchant_data.py` plants shared refund destinations only on
  orders already labelled abusive, so the network signal is perfectly
  correlated with abuse in the simulation. In production it would produce
  false positives — families share accounts, one person pays for several.
  Its real-world precision is untested.
- **Evidence escalation is currently blunt.** Of 852 orders escalated for more
  evidence, only 71 (8.3%) are actually abusive, so 781 legitimate customers
  would be asked for a photo. That friction is a real cost and the rule should
  be narrowed.
- **Nothing was fine-tuned.** The `risk-agent` model is stock `qwen2.5` plus a
  system prompt, five in-context examples, and pinned sampling parameters —
  zero gradient steps, zero parameters updated. Verifiable: all three local
  models share the identical 4,466 MB weight file.
- **Single-dataset evaluation.** No cross-dataset validation, adversarial
  testing, or drift analysis.

## Why defense-only
This system exists purely to protect a merchant from loss — it never generates
attacks, never helps anyone evade detection, and every flagged case still goes
to a human before any punitive action is taken.

## Bonus: multimodal photo evidence (explicitly secondary, merchant-facing extension)
Beyond the core payment-layer verifier above, we also built an optional tool,
`analyze_return_photo`, that lets the agent visually inspect photo evidence
submitted with a return (using a local vision model, moondream, via Ollama).
**We're upfront that this piece belongs conceptually to the merchant, not the
gateway** — Razorpay's core value is the payment-layer fraud signal above; this
extension exists to show the same agentic architecture generalizes to a
merchant-facing product Razorpay could optionally offer as a value-add, not as
a claim that payment processing should include product-condition judgment.

Since the training dataset only contains a `photo_evidence_provided`
true/false flag (not actual image files), this was demonstrated using
synthetic, clearly-labeled placeholder images rather than real return photos
— this proves the architecture works end-to-end, not that it's been
validated on real fraud imagery.

Interestingly, in test runs the vision model correctly identified the
placeholder images as synthetic/staged ("looks like a static image with
text," "placeholder/label image, not the actual returned item") rather than
inventing a plausible-sounding but false damage description. That's a
meaningful trust signal: the model is genuinely reasoning about image
content, not hallucinating.

This remains a bonus/optional layer — the core submission and its measured
precision/recall are the text-based verifier described above, which does not
depend on this extension.

## What we'd build next with more time
- Extend from binary (abuse / not abuse) to the 3 sub-types (Policy Abuser,
  Fraudulent Return, Wardrobing) so the review queue can be prioritized by
  abuse type, not just a single flag.
- A feedback loop: when a human reviewer overturns the agent's decision, log
  that as a labeled example to periodically retrain the model.
