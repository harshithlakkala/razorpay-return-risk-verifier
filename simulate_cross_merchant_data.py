"""
simulate_cross_merchant_data.py
Your real dataset is single-merchant (it has no merchant_id or actual refund
destination account - only a refund_to_different_account TRUE/FALSE flag).
But the whole point of this feature is a signal that only exists ACROSS
merchants, which no single-merchant dataset can show.

So this script builds a realistic multi-merchant simulation on top of your
real orders:
  - Assigns each order to one of 20 simulated merchants
  - Assigns each order a refund destination account fingerprint
  - Deliberately plants "refund farming rings": a subset of ABUSE orders
    share the same destination account across MULTIPLE merchants - exactly
    the pattern a single merchant could never see, but a payment gateway
    sitting across all of them could.

This is clearly a simulation for demonstrating the architecture, not a claim
about real fraud rings in the Kaggle dataset - stated plainly in the writeup.

Usage:
  python simulate_cross_merchant_data.py
"""
import pandas as pd
import random
import hashlib

random.seed(42)

CHARSET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"


def realistic_destination(seed: str) -> str:
    """Generate a Razorpay-realistic refund destination (UPI VPA or masked
    bank account), matching real Razorpay API conventions rather than a
    generic placeholder like ACCT-00123. Uses enough entropy in BOTH branches
    to avoid accidental collisions across thousands of orders (a masked
    account showing only 4 digits has just 10,000 possible values, which
    WILL collide at this scale - fixed by using 8 digits split first4****last4)."""
    h = hashlib.sha256(seed.encode()).digest()
    if h[0] % 2 == 0:
        bank = ["okhdfcbank", "oksbi", "okicici", "okaxis", "ybl"][h[1] % 5]
        handle = "".join(CHARSET[b % len(CHARSET)] for b in h[2:9]).lower()
        return f"{handle}@{bank}"
    else:
        digits = "".join(str(b % 10) for b in h[2:10])  # 8 digits, 10^8 space
        return f"bank_acct_{digits[:4]}****{digits[4:]}"


df = pd.read_csv("test.csv")

MERCHANTS = [f"MERCH{i:03d}" for i in range(1, 21)]  # 20 simulated merchants
RING_ACCOUNTS = [realistic_destination(f"ring{i}") for i in range(1, 31)]  # 30 "ring" destinations, Razorpay-format

records = []
abuse_indices = df[df["is_abuse"] == 1].index.tolist()
random.shuffle(abuse_indices)

# Plant rings: 20% of abuse orders get assigned to a shared ring account,
# and we make sure each ring account is used across 2-5 DIFFERENT merchants
ring_order_count = int(len(abuse_indices) * 0.20)
ring_orders = set(abuse_indices[:ring_order_count])

# Pre-assign each ring account a set of merchants it'll appear under
ring_account_merchants = {acct: random.sample(MERCHANTS, k=random.randint(2, 5))
                           for acct in RING_ACCOUNTS}

for idx, row in df.iterrows():
    order_id = row["order_id"]
    if idx in ring_orders:
        account = random.choice(RING_ACCOUNTS)
        merchant = random.choice(ring_account_merchants[account])
    else:
        merchant = random.choice(MERCHANTS)
        account = realistic_destination(order_id)  # effectively unique per order
    records.append({"order_id": order_id, "merchant_id": merchant, "refund_destination_account": account})

network_df = pd.DataFrame(records)
network_df.to_csv("cross_merchant_network.csv", index=False)

# Report how many genuine cross-merchant patterns got created, for sanity
acct_merchant_counts = network_df.groupby("refund_destination_account")["merchant_id"].nunique()
multi_merchant_accounts = acct_merchant_counts[acct_merchant_counts > 1]
print(f"Simulated {len(MERCHANTS)} merchants, {len(df)} orders.")
print(f"Accounts used across MULTIPLE merchants: {len(multi_merchant_accounts)}")
print(f"Orders tied to a cross-merchant account: "
      f"{network_df['refund_destination_account'].isin(multi_merchant_accounts.index).sum()}")
print("Saved to cross_merchant_network.csv")
