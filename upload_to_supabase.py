"""
upload_to_supabase.py
Uploads your local test.csv, razorpay_ids.csv, and cross_merchant_network.csv
into the 3 Supabase tables created by schema.sql.

SETUP (one-time):
  1. Go to supabase.com, sign up free, create a new project (takes ~2 min to spin up)
  2. In the project, go to SQL Editor -> New Query, paste in schema.sql, click Run
  3. Go to Project Settings -> API. Copy your "Project URL" and "anon public" key
  4. pip install supabase pandas

USAGE:
  Set these two environment variables first (PowerShell):
    $env:SUPABASE_URL="https://your-project.supabase.co"
    $env:SUPABASE_KEY="your-anon-public-key"
  Then run:
    python upload_to_supabase.py
"""
import os
import pandas as pd
import numpy as np
from supabase import create_client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise SystemExit(
        "Missing SUPABASE_URL or SUPABASE_KEY environment variables.\n"
        "Set them first, e.g. in PowerShell:\n"
        '  $env:SUPABASE_URL="https://your-project.supabase.co"\n'
        '  $env:SUPABASE_KEY="your-anon-public-key"'
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)


# test.csv carries columns the orders table deliberately does NOT store:
# customer_id/order_date/return_date are identifiers, and abuse_type/abuse_label
# are the answer leaking in (same exclusions train_model.py makes). Upload only
# the columns schema.sql actually defines.
ORDERS_COLUMNS = [
    "order_id", "age", "account_age_days", "avg_order_value_usd",
    "refund_amount_requested_usd", "is_high_value_item", "discount_used",
    "days_to_return", "total_orders_lifetime", "total_returns_lifetime",
    "return_rate_pct", "item_returned_opened", "return_packaging_intact",
    "photo_evidence_provided", "tracking_number_valid",
    "address_change_before_delivery", "refund_to_different_account",
    "multiple_accounts_flag", "customer_support_contacts",
    "previous_dispute_count", "wishlist_to_cart_time_hrs",
    "review_left_after_return", "customer_segment", "country", "platform",
    "device_type", "payment_method", "product_category", "return_reason",
    "shipping_carrier", "is_abuse",
]


def upload_dataframe(df: pd.DataFrame, table_name: str, batch_size: int = 500,
                     columns: list = None):
    """Uploads in batches - Supabase/Postgres rejects giant single inserts,
    and batching also means a mid-upload failure doesn't lose everything."""
    if columns is not None:
        df = df[columns]
    df = df.replace({np.nan: None})
    records = df.to_dict(orient="records")
    total = len(records)
    for i in range(0, total, batch_size):
        batch = records[i:i + batch_size]
        supabase.table(table_name).upsert(batch).execute()
        print(f"  {table_name}: uploaded {min(i + batch_size, total)}/{total}")


print("Uploading orders...")
orders_df = pd.read_csv("test.csv")
upload_dataframe(orders_df, "orders", columns=ORDERS_COLUMNS)

print("Uploading razorpay_ids...")
ids_df = pd.read_csv("razorpay_ids.csv")
upload_dataframe(ids_df, "razorpay_ids")

print("Uploading cross_merchant_network...")
network_df = pd.read_csv("cross_merchant_network.csv")
upload_dataframe(network_df, "cross_merchant_network")

print("\nDone. All 3 tables populated in Supabase.")
