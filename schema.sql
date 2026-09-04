-- schema.sql
-- Run this in the Supabase SQL Editor (Project → SQL Editor → New Query)
-- before uploading any data. Creates the 3 tables that replace your CSVs.

create table if not exists orders (
    order_id text primary key,
    age integer,
    account_age_days integer,
    avg_order_value_usd numeric,
    refund_amount_requested_usd numeric,
    is_high_value_item integer,
    discount_used integer,
    days_to_return integer,
    total_orders_lifetime integer,
    total_returns_lifetime integer,
    return_rate_pct numeric,
    item_returned_opened integer,
    return_packaging_intact integer,
    photo_evidence_provided integer,
    tracking_number_valid integer,
    address_change_before_delivery integer,
    refund_to_different_account integer,
    multiple_accounts_flag integer,
    customer_support_contacts integer,
    previous_dispute_count integer,
    wishlist_to_cart_time_hrs numeric,
    review_left_after_return integer,
    customer_segment text,
    country text,
    platform text,
    device_type text,
    payment_method text,
    product_category text,
    return_reason text,
    shipping_carrier text,
    is_abuse integer
);

create table if not exists razorpay_ids (
    order_id text primary key references orders(order_id),
    razorpay_order_id text,
    razorpay_payment_id text unique,
    razorpay_refund_id text,
    refund_destination text
);

create table if not exists cross_merchant_network (
    order_id text primary key references orders(order_id),
    merchant_id text,
    refund_destination_account text
);

-- Index for the actual cross-merchant lookup pattern (find other orders
-- sharing the same refund destination) - this is the query that matters most
create index if not exists idx_refund_destination
    on cross_merchant_network (refund_destination_account);

create index if not exists idx_payment_id
    on razorpay_ids (razorpay_payment_id);
