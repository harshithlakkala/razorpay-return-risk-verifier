"""
decision_policy.py
THE decision policy. One definition, imported by every entry point.

Why this module exists
----------------------
The rules originally lived in the agent's system prompt, i.e. the model was
*asked* to follow them. A 7B local model does not reliably obey a paragraph of
instructions: it invented a tool argument that crashed a 100-order evaluation,
and it returned opposite verdicts on byte-identical input. Neither is
acceptable in a system that moves money.

So the rules live here instead, as a pure function - no I/O, no model, no
randomness. Every caller (the CLI agent, both web apps, the interactive
tester, and the full-set evaluator) imports this same function, which is what
makes "the policy is enforced in code" a claim about the whole system rather
than about one script.

`decide()` takes values already fetched by the caller's own tools, so each
entry point keeps its own data access (in-memory dict, CSV, or Supabase) while
sharing identical decision logic.
"""

# Cost-optimal threshold: the centre of the range that minimised real
# false-positive/false-negative dollar cost on held-out data, not a default.
# See cost_threshold.py.
DECISION_THRESHOLD = 0.43

# The only actions the policy may produce. Anything else from a model is a
# malformed tool call, not a decision.
VALID_ACTIONS = {"approve", "flag_for_review", "request_more_evidence"}

# Words in a return reason that assert a physical defect - a claim that cannot
# be assessed at all without a photo.
_DAMAGE_WORDS = ("damag", "broken", "defect", "faulty")


def claims_damage(return_reason) -> bool:
    """True if the stated return reason asserts physical damage or a defect."""
    return any(w in str(return_reason).lower() for w in _DAMAGE_WORDS)


def decide(abuse_probability, cross_merchant_detected, return_reason,
           has_photo_evidence, distinct_merchants=None,
           threshold: float = DECISION_THRESHOLD) -> dict:
    """Apply the decision policy. Pure function: same inputs, same output.

    Rules, in strict priority order - the first match wins:

      1. Cross-merchant refund destination -> flag_for_review, whatever the ML
         score says. The model was never trained on network data, so a low
         probability is not evidence against a shared destination and must not
         be allowed to dilute it.
      2. Damage/defect claimed with no photo -> request_more_evidence. The
         claim can be neither confirmed nor denied, so no verdict is forced.
      3. abuse_probability above the cost-optimal threshold -> flag_for_review.
      4. Otherwise -> approve.

    Returns the action plus reasoning citing the actual figures, so the audit
    log records why, not just what.
    """
    proba = abuse_probability
    cross_merchant = bool(cross_merchant_detected)

    if cross_merchant:
        n = distinct_merchants if distinct_merchants is not None else "multiple"
        return {
            "action": "flag_for_review",
            "reasoning": (
                f"Rule 1: refund destination is shared across {n} merchants - a "
                f"payment-network pattern no single merchant can observe. Flagged "
                f"regardless of the ML score ({proba})."
            ),
        }

    if claims_damage(return_reason) and not has_photo_evidence:
        return {
            "action": "request_more_evidence",
            "reasoning": (
                f"Rule 2: return reason '{return_reason}' claims physical damage but "
                f"no photo evidence was submitted, so the claim cannot be verified "
                f"either way. ML score {proba}, no cross-merchant pattern."
            ),
        }

    if proba is not None and proba > threshold:
        return {
            "action": "flag_for_review",
            "reasoning": (
                f"Rule 3: ML abuse probability {proba} exceeds the cost-optimal "
                f"threshold {threshold}; no cross-merchant pattern detected."
            ),
        }

    return {
        "action": "approve",
        "reasoning": (
            f"Rule 4: ML abuse probability {proba} is below the cost-optimal "
            f"threshold {threshold}, no cross-merchant pattern, and no evidence gap."
        ),
    }


def reconcile(agent_decision, policy, order_id: str):
    """Compare the model's proposed action against the policy.

    Returns (final_action, final_reasoning). Where the two disagree the policy
    wins, and the override is written into the reasoning so the audit log shows
    both what the model concluded and what was actually done.
    """
    if agent_decision is None:
        return policy["action"], (
            f"[policy decision - agent returned no verdict] {policy['reasoning']}"
        )

    proposed = agent_decision.get("action")
    if proposed not in VALID_ACTIONS or proposed != policy["action"]:
        return policy["action"], (
            f"[policy override - agent proposed '{proposed}'] {policy['reasoning']} "
            f"Agent's reasoning was: {agent_decision.get('reasoning', '')}"
        )

    return proposed, agent_decision.get("reasoning", policy["reasoning"])
