"""
generate_razorpay_ids.py
Your friend is right: real Razorpay data doesn't look like "ORD2013623" -
it looks like pay_29QQoUBi66xm2f (payments), rfnd_FP8QHiV938haTz (refunds),
and order_XXXXXXXXXXXXXX (Razorpay's own payment-order concept, which is
NOT the same as a merchant's product order - it's the payment transaction
itself). Verified against Razorpay's actual API docs.

This script generates realistic-format IDs for every order in your dataset,
deterministically (same order_id always maps to the same Razorpay IDs, so
re-running this doesn't shuffle everything), and maps payment method to a
realistic refund destination format:
  - UPI payments  -> a VPA-style destination (name@bank)
  - Card payments -> a masked card fingerprint
  - Other methods -> a masked bank account number

Usage:
  python generate_razorpay_ids.py
"""
import pandas as pd
import hashlib
import random

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def deterministic_id(seed: str, prefix: str, length: int = 14) -> str:
    """Same seed always produces the same ID - realistic-looking, reproducible."""
    h = hashlib.sha256(seed.encode()).digest()
    chars = "".join(CHARSET[b % len(CHARSET)] for b in h[:length])
    return f"{prefix}_{chars}"


def refund_destination_for_method(order_id: str, payment_method: str) -> str:
    h = hashlib.sha256(order_id.encode()).digest()
    method = str(payment_method).lower()
    if "upi" in method:
        bank = ["okhdfcbank", "oksbi", "okicici", "okaxis", "ybl", "paytm"][h[0] % 6]
        handle = deterministic_id(order_id + "upi", "", 8).lower().lstrip("_")
        return f"{handle}@{bank}"
    elif "card" in method:
        digits = "".join(str(b % 10) for b in h[1:9])  # 8 digits, 10^8 space
        return f"card_fp_{digits[:4]}****{digits[4:]}"
    else:
        digits = "".join(str(b % 10) for b in h[2:10])  # 8 digits, 10^8 space
        return f"bank_acct_{digits[:4]}****{digits[4:]}"


df = pd.read_csv("test.csv")

records = []
for _, row in df.iterrows():
    order_id = row["order_id"]
    method = row.get("payment_method", "")
    records.append({
        "order_id": order_id,  # kept as an internal join key only
        "razorpay_order_id": deterministic_id(order_id + "order", "order"),
        "razorpay_payment_id": deterministic_id(order_id + "pay", "pay"),
        "razorpay_refund_id": deterministic_id(order_id + "rfnd", "rfnd"),
        "refund_destination": refund_destination_for_method(order_id, method),
    })

ids_df = pd.DataFrame(records)
ids_df.to_csv("razorpay_ids.csv", index=False)
print(f"Generated Razorpay-format IDs for {len(ids_df)} orders.")
print(ids_df.head(3).to_string(index=False))
print("\nSaved to razorpay_ids.csv")
