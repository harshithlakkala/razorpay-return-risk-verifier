"""
compute_partial_metrics.py
Reconstructs precision/recall from audit_log.txt when an --evaluate run
got interrupted partway through (e.g. killed before printing final metrics).

Handles the fact that audit_log.txt has entries mixed in from multiple runs
(the --demo run, and any earlier --evaluate attempts) by keeping only the
MOST RECENT decision per order_id, then scoring those against test.csv's
ground truth.

Usage:
  python compute_partial_metrics.py
"""
import json
import pandas as pd
from sklearn.metrics import precision_score, recall_score, confusion_matrix

# Step 1: read every line of the audit log, keep only the latest decision
# per order_id (later lines overwrite earlier ones for the same order)
decisions = {}
with open("audit_log.txt", "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        entry = json.loads(line)
        decisions[entry["order_id"]] = entry  # later entries overwrite earlier

print(f"Found {len(decisions)} unique order decisions in audit_log.txt")

# Step 2: match those order_ids against the held-out test set's ground truth
test_df = pd.read_csv("test.csv")
test_df = test_df.set_index("order_id")

y_true, y_pred, matched_ids, escalated_count = [], [], [], 0
for order_id, entry in decisions.items():
    if order_id not in test_df.index:
        continue  # skip if somehow not in the held-out set
    matched_ids.append(order_id)
    if entry["action"] == "request_more_evidence":
        escalated_count += 1
        continue  # handled separately below, not folded into binary precision/recall
    y_true.append(int(test_df.loc[order_id, "is_abuse"]))
    y_pred.append(1 if entry["action"] == "flag_for_review" else 0)

print(f"Matched {len(matched_ids)} of those to ground truth in test.csv")
if escalated_count:
    print(f"({escalated_count} were request_more_evidence - excluded from binary precision/recall, reported separately)")

if len(matched_ids) == 0:
    print("No matches found - check that audit_log.txt and test.csv are in this folder.")
else:
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    print("\n" + "=" * 50)
    print(f"PARTIAL RESULTS ({len(matched_ids)} orders)")
    print("=" * 50)
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print("\nConfusion matrix:")
    print("                  Predicted Legit   Predicted Abuse")
    print(f"Actual Legit       {cm[0][0]:>6}            {cm[0][1]:>6}")
    print(f"Actual Abuse       {cm[1][0]:>6}            {cm[1][1]:>6}")
