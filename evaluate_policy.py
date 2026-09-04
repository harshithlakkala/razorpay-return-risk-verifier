"""
evaluate_policy.py
Evaluates the DECISION POLICY across the entire held-out test set.

Why this is possible instantly: since the policy is enforced in code
(policy_decision() in risk_agent_local.py, with run_agent_on_order overriding
the model whenever the two disagree), the agent's final action is guaranteed
to equal the policy's action. So the policy's accuracy over all 12,000 orders
is the agent's accuracy - measurable in seconds instead of the ~10 days of
local CPU inference a real 12,000-order agent run would take.

That is exactly why the guardrails were moved out of the prompt: it makes the
system's behaviour provable rather than merely observed on a small sample.

request_more_evidence is reported separately and never scored as a binary
correct/incorrect, since escalating is not the same as getting it wrong.

Usage:
  python evaluate_policy.py
"""
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix

import risk_agent_local as agent
from decision_policy import decide, DECISION_THRESHOLD

test = pd.read_csv("test.csv")
network = pd.read_csv("cross_merchant_network.csv")

print(f"Evaluating policy over {len(test)} held-out orders...\n")

# Score every order in one batched model call rather than 12,000 single ones
feat = test[agent.FEATURE_COLUMNS]
proba = agent.MODEL.predict_proba(feat)[:, 1]

# Which refund destinations appear under more than one merchant
merchants_per_account = network.groupby("refund_destination_account")["merchant_id"].nunique()
shared_accounts = set(merchants_per_account[merchants_per_account > 1].index)
account_by_order = network.set_index("order_id")["refund_destination_account"].to_dict()

reason = test["return_reason"].astype(str).str.lower()
damage_claim = reason.str.contains("damag|broken|defect|faulty", regex=True)
has_photo = test["photo_evidence_provided"].astype(bool)

# Call the SAME decide() the agent uses in production, once per order, rather
# than reimplementing the rules here. If the two ever diverged, this evaluation
# would stop being evidence about the deployed system.
actions = []
for i, order_id in enumerate(test["order_id"]):
    result = decide(
        abuse_probability=proba[i],
        cross_merchant_detected=account_by_order.get(order_id) in shared_accounts,
        return_reason=test["return_reason"].iloc[i],
        has_photo_evidence=bool(has_photo.iloc[i]),
    )
    actions.append(result["action"])

test["agent_action"] = actions
test["actual_label"] = test["is_abuse"].map({1: "ABUSE", 0: "LEGITIMATE"})

print("ACTION DISTRIBUTION")
print(test["agent_action"].value_counts().to_string())
print()
print("ACTION vs GROUND TRUTH")
print(pd.crosstab(test["agent_action"], test["actual_label"]).to_string())
print()

escalated = test[test["agent_action"] == "request_more_evidence"]
binary = test[test["agent_action"] != "request_more_evidence"]
y_true = binary["is_abuse"]
y_pred = (binary["agent_action"] == "flag_for_review").astype(int)

p = precision_score(y_true, y_pred)
r = recall_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print("BINARY DECISIONS (escalations excluded, reported separately)")
print(f"  Scored:    {len(binary)} of {len(test)} orders")
print(f"  Precision: {p:.4f}")
print(f"  Recall:    {r:.4f}")
print(f"  Correct:   {(y_true == y_pred).sum()} / {len(binary)}")
print()
print("                 Predicted Legit   Predicted Abuse")
print(f"  Actual Legit   {cm[0][0]:>13}   {cm[0][1]:>15}   <- false positives")
print(f"  Actual Abuse   {cm[1][0]:>13}   {cm[1][1]:>15}   <- MISSED ABUSE")
print()

if len(escalated):
    abuse_in_esc = int(escalated["is_abuse"].sum())
    print(f"ESCALATED (request_more_evidence): {len(escalated)} orders "
          f"({len(escalated) / len(test):.1%})")
    print(f"  Of those, actually abuse: {abuse_in_esc} ({abuse_in_esc / len(escalated):.1%})")
    print("  These are held for evidence, not wrongly approved or wrongly flagged.")
    print()

# Dollar cost of the errors that remain
fp = binary[(y_true == 0) & (y_pred == 1)]
fn = binary[(y_true == 1) & (y_pred == 0)]
fp_cost = fp["avg_order_value_usd"].sum()
fn_cost = fn["refund_amount_requested_usd"].sum()
print("DOLLAR COST OF REMAINING ERRORS")
print(f"  False positives: {len(fp)} orders, ${fp_cost:,.2f} customer friction")
print(f"  False negatives: {len(fn)} orders, ${fn_cost:,.2f} abuse not caught")
print(f"  Total:           ${fp_cost + fn_cost:,.2f}")
