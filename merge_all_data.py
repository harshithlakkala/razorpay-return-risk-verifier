"""
merge_all_data.py
Joins the three data sources into one wide table that downstream reporting
scripts read from:

  test.csv                    -> order/customer features + is_abuse ground truth
  razorpay_ids.csv            -> razorpay order/payment/refund IDs
  cross_merchant_network.csv  -> merchant_id + refund destination account

The ground-truth is_abuse flag is also exposed as a human-readable
actual_label ("ABUSE" / "LEGITIMATE"), which is what the audit report prints
next to each decision.

Usage:
  python merge_all_data.py
"""
import pandas as pd

orders = pd.read_csv("test.csv")
ids = pd.read_csv("razorpay_ids.csv")
network = pd.read_csv("cross_merchant_network.csv")

merged = orders.merge(ids, on="order_id", how="left")
merged = merged.merge(network, on="order_id", how="left")

merged["actual_label"] = merged["is_abuse"].map({1: "ABUSE", 0: "LEGITIMATE"})

merged.to_csv("merged_data.csv", index=False)
print(f"Merged {len(merged)} orders, {len(merged.columns)} columns -> merged_data.csv")
print(f"  with razorpay IDs:      {merged['razorpay_payment_id'].notna().sum()}")
print(f"  with network records:   {merged['refund_destination_account'].notna().sum()}")
print(f"  label breakdown:        {merged['actual_label'].value_counts().to_dict()}")
