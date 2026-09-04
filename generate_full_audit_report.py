"""
generate_full_audit_report.py
The most complete version of the audit report: EVERY entry ever logged to
audit_log.txt (no deduping - if an order was reviewed 3 times across
different test runs, all 3 appear), each one enriched with the FULL context
behind that decision - order details, customer history, ML risk score,
cross-merchant network data, and the real ground-truth label.

This is the single most complete artifact of "what did the agent see, and
what did it decide" - useful for your GitHub repo as proof the system is
fully auditable end-to-end, not just a black box producing a verdict.

Requires: audit_log.txt, merged_data.csv (from merge_all_data.py), and
risk_model.joblib all in this folder.

Usage:
  python generate_full_audit_report.py
"""
import json
import pandas as pd
import joblib

FEATURE_COLUMNS = [
    "age", "account_age_days", "avg_order_value_usd",
    "refund_amount_requested_usd", "is_high_value_item", "discount_used",
    "days_to_return", "total_orders_lifetime", "total_returns_lifetime",
    "return_rate_pct", "item_returned_opened", "return_packaging_intact",
    "photo_evidence_provided", "tracking_number_valid",
    "address_change_before_delivery", "refund_to_different_account",
    "multiple_accounts_flag", "customer_support_contacts",
    "previous_dispute_count", "wishlist_to_cart_time_hrs",
    "review_left_after_return", "customer_segment", "country", "platform",
    "device_type", "payment_method", "product_category", "return_reason",
    "shipping_carrier",
]

# --- Load every audit log entry, in order, no deduping ---
log_entries = []
with open("audit_log.txt", "r") as f:
    for line in f:
        line = line.strip()
        if line:
            log_entries.append(json.loads(line))
print(f"Total logged decisions (every run, including repeats): {len(log_entries)}")

# --- Load full order context and the model, to enrich each entry ---
data = pd.read_csv("merged_data.csv").set_index("order_id")
model = joblib.load("risk_model.joblib")

rows = []
for entry in log_entries:
    order_id = entry["order_id"]
    row = {
        "timestamp": entry["timestamp"],
        "order_id": order_id,
        "agent_action": entry["action"],
        "agent_reasoning": entry["reasoning"],
    }
    if order_id in data.index:
        o = data.loc[order_id]
        # Ground truth and identifiers
        row["actual_label"] = o.get("actual_label")
        row["decision_correct"] = (
            (entry["action"] == "flag_for_review" and o.get("actual_label") == "ABUSE") or
            (entry["action"] == "approve" and o.get("actual_label") == "LEGITIMATE")
        )
        row["razorpay_payment_id"] = o.get("razorpay_payment_id")
        row["razorpay_order_id"] = o.get("razorpay_order_id")
        row["razorpay_refund_id"] = o.get("razorpay_refund_id")
        row["refund_destination_account"] = o.get("refund_destination_account")
        row["merchant_id"] = o.get("merchant_id")
        # Full order/customer context
        for col in FEATURE_COLUMNS:
            row[col] = o.get(col)
        # Recompute the ML score for full traceability
        try:
            feat_row = pd.DataFrame([{c: o[c] for c in FEATURE_COLUMNS}])
            row["ml_abuse_probability"] = round(float(model.predict_proba(feat_row)[0][1]), 4)
        except Exception:
            row["ml_abuse_probability"] = None
    else:
        row["actual_label"] = "UNKNOWN (order not found in merged_data.csv)"
    rows.append(row)

report_df = pd.DataFrame(rows)
report_df.to_csv("full_audit_report.csv", index=False)
print(f"Saved full_audit_report.csv - {len(report_df)} rows, {len(report_df.columns)} columns")

with open("full_audit_report.json", "w") as f:
    json.dump(rows, f, indent=2, default=str)
print(f"Saved full_audit_report.json")

# Quick summary printed to screen
correct = report_df["decision_correct"].sum() if "decision_correct" in report_df else 0
total_with_label = report_df["actual_label"].notna().sum()
print(f"\nOf entries with a known ground truth label: {correct}/{total_with_label} decisions matched (note: request_more_evidence entries are not scored correct/incorrect here, only approve/flag)")
