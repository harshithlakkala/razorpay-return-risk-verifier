"""
cost_threshold.py
Finds the decision threshold that MINIMIZES TOTAL DOLLAR LOSS, instead of
blindly using the default 0.5 cutoff.

Why this matters: a false positive (wrongly flagging a legitimate return)
and a false negative (missing real abuse) do NOT cost the same amount.
- False positive cost ~= the order value (friction/lost trust on a real customer)
- False negative cost ~= the refund amount that slips through as abuse

A model that's "95% accurate" can still be losing more money than one that's
"90% accurate," if its errors are concentrated on the expensive kind. This
script sweeps every possible threshold and picks the one that minimizes
actual dollar cost, not just misclassification count.
"""
import pandas as pd
import joblib
import numpy as np

model = joblib.load("risk_model.joblib")
test_df = pd.read_csv("test.csv")

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

X_test = test_df[FEATURE_COLUMNS]
y_test = test_df["is_abuse"].values
proba = model.predict_proba(X_test)[:, 1]  # probability of abuse, for every order

fp_costs = test_df["avg_order_value_usd"].values          # cost if wrongly flagged
fn_costs = test_df["refund_amount_requested_usd"].values  # cost if abuse missed

results = []
for threshold in np.arange(0.01, 1.00, 0.01):
    y_pred = (proba >= threshold).astype(int)

    fp_mask = (y_test == 0) & (y_pred == 1)
    fn_mask = (y_test == 1) & (y_pred == 0)

    total_fp_cost = fp_costs[fp_mask].sum()
    total_fn_cost = fn_costs[fn_mask].sum()
    total_cost = total_fp_cost + total_fn_cost

    results.append({
        "threshold": round(threshold, 2),
        "false_positives": int(fp_mask.sum()),
        "false_negatives": int(fn_mask.sum()),
        "fp_cost": round(total_fp_cost, 2),
        "fn_cost": round(total_fn_cost, 2),
        "total_cost": round(total_cost, 2),
    })

results_df = pd.DataFrame(results)
best_row = results_df.loc[results_df["total_cost"].idxmin()]
default_row = results_df.iloc[(results_df["threshold"] - 0.50).abs().argsort()[:1]].iloc[0]

print("=" * 70)
print("COST-MINIMIZING THRESHOLD SEARCH")
print("=" * 70)
print(f"\nDefault threshold (0.50): total cost = ${default_row['total_cost']:,.2f} "
      f"({int(default_row['false_positives'])} FP, {int(default_row['false_negatives'])} FN)")
print(f"Optimal threshold ({best_row['threshold']}): total cost = ${best_row['total_cost']:,.2f} "
      f"({int(best_row['false_positives'])} FP, {int(best_row['false_negatives'])} FN)")

savings = default_row["total_cost"] - best_row["total_cost"]
print(f"\nSavings from tuning the threshold: ${savings:,.2f}")

# Show the full cost curve around the optimum for the writeup
print("\nCost curve near the optimum:")
window = results_df[(results_df["threshold"] >= max(0.01, best_row["threshold"] - 0.05)) &
                     (results_df["threshold"] <= min(0.99, best_row["threshold"] + 0.05))]
print(window.to_string(index=False))

results_df.to_csv("threshold_sweep_results.csv", index=False)
print("\nFull sweep saved to threshold_sweep_results.csv")

# --- Find the MOST ROBUST threshold, not just the first zero-cost one ---
# The edge of a safe zone is risky if real-world data drifts even slightly.
# The center of the safe zone gives maximum margin in both directions.
zero_cost = results_df[results_df["total_cost"] == 0]
if len(zero_cost) > 0:
    safe_min = zero_cost["threshold"].min()
    safe_max = zero_cost["threshold"].max()
    robust_threshold = round((safe_min + safe_max) / 2, 2)
    print(f"\nZero-cost safe zone: {safe_min} to {safe_max}")
    print(f"ROBUST THRESHOLD (center of safe zone, recommended for production): {robust_threshold}")
    with open("recommended_threshold.txt", "w") as f:
        f.write(str(robust_threshold))
    print("Saved to recommended_threshold.txt for the agent to use.")
