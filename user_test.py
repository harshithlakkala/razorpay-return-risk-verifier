"""
user_test.py
INTERACTIVE test - YOU pick the order and the photo, and watch the agent
reason through it live. Good for demos and for building confidence in how
the agent behaves, beyond the automated batch evaluation.

Usage:
  python user_test.py
"""
import inspect
import json
import joblib
import pandas as pd
import ollama

from decision_policy import decide, reconcile

# Specialised model built from Modelfile.risk-agent. Rebuild with:
#   ollama create risk-agent -f Modelfile.risk-agent
TEXT_MODEL = "risk-agent"
VISION_MODEL = "moondream"

# Same order + same evidence must always yield the same verdict (see
# risk_agent_local.py for why). Greedy decoding, no sampling, fixed seed.
DETERMINISTIC_OPTIONS = {"temperature": 0, "top_p": 1, "top_k": 1, "seed": 42}
MODEL = joblib.load("risk_model.joblib")

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

PLACEHOLDER_PHOTOS = {
    "1": ("sample_photos/genuine_damage.png", "Genuine damage"),
    "2": ("sample_photos/suspicious_unused.png", "Suspicious / unused, tags attached"),
    "3": ("sample_photos/mismatched_item.png", "Mismatched item"),
    "4": (None, "No photo submitted"),
}


def ask(prompt, cast=float, default=None):
    raw = input(f"{prompt} [{default}]: ").strip()
    if raw == "":
        return default
    return cast(raw)


def lookup_real_order(preset_id=None):
    """Look up a real order from test.csv by its order_id."""
    df = pd.read_csv("test.csv")
    order_id = preset_id or input("Enter an order_id (e.g. ORD2013623): ").strip()
    match = df[df["order_id"] == order_id]
    if match.empty:
        print(f"No order found with id '{order_id}'. Here are 5 real IDs you can try:")
        print(df["order_id"].sample(5, random_state=1).tolist())
        return None, None

    row = match.iloc[0]
    order = row.to_dict()
    print("\n--- Found order ---")
    for col in FEATURE_COLUMNS:
        print(f"  {col}: {order[col]}")
    print(f"  (actual label in dataset: {'ABUSE' if order['is_abuse'] == 1 else 'LEGITIMATE'})")

    print("\nPhoto evidence options (dataset says photo_evidence_provided ="
          f" {order['photo_evidence_provided']}):")
    for k, (_, label) in PLACEHOLDER_PHOTOS.items():
        print(f"  {k}. {label}")
    choice = input("Pick a photo to attach for this test [4]: ").strip() or "4"
    photo_path, _ = PLACEHOLDER_PHOTOS.get(choice, (None, None))
    order["_photo_path"] = photo_path
    return order, order_id


def build_custom_order():
    print("\n--- Enter order details (press Enter to accept the default shown) ---")
    order = {
        "age": ask("Customer age", int, 30),
        "account_age_days": ask("Account age (days)", int, 200),
        "avg_order_value_usd": ask("Average order value (USD)", float, 60.0),
        "refund_amount_requested_usd": ask("Refund amount requested (USD)", float, 60.0),
        "is_high_value_item": ask("Is high value item? (1/0)", int, 0),
        "discount_used": ask("Discount used? (1/0)", int, 0),
        "days_to_return": ask("Days to return", int, 10),
        "total_orders_lifetime": ask("Total lifetime orders", int, 20),
        "total_returns_lifetime": ask("Total lifetime returns", int, 2),
        "return_rate_pct": ask("Return rate %", float, 10.0),
        "item_returned_opened": ask("Item was opened? (1/0)", int, 1),
        "return_packaging_intact": ask("Packaging intact? (1/0)", int, 1),
        "tracking_number_valid": ask("Tracking number valid? (1/0)", int, 1),
        "address_change_before_delivery": ask("Address changed before delivery? (1/0)", int, 0),
        "refund_to_different_account": ask("Refund to different account? (1/0)", int, 0),
        "multiple_accounts_flag": ask("Multiple accounts flag? (1/0)", int, 0),
        "customer_support_contacts": ask("Support contacts", int, 1),
        "previous_dispute_count": ask("Previous disputes", int, 0),
        "wishlist_to_cart_time_hrs": ask("Wishlist-to-cart time (hrs)", float, 24.0),
        "review_left_after_return": ask("Review left after return? (1/0)", int, 0),
        "customer_segment": input("Customer segment [Regular]: ").strip() or "Regular",
        "country": input("Country [India]: ").strip() or "India",
        "platform": input("Platform [Web]: ").strip() or "Web",
        "device_type": input("Device type [Mobile]: ").strip() or "Mobile",
        "payment_method": input("Payment method [UPI]: ").strip() or "UPI",
        "product_category": input("Product category [Apparel]: ").strip() or "Apparel",
        "return_reason": input("Return reason [Wrong size]: ").strip() or "Wrong size",
        "shipping_carrier": input("Shipping carrier [Delhivery]: ").strip() or "Delhivery",
    }

    print("\nPhoto evidence options:")
    for k, (_, label) in PLACEHOLDER_PHOTOS.items():
        print(f"  {k}. {label}")
    choice = input("Pick a photo option [4]: ").strip() or "4"
    photo_path, _ = PLACEHOLDER_PHOTOS.get(choice, (None, None))
    order["photo_evidence_provided"] = 1 if photo_path else 0
    order["_photo_path"] = photo_path
    return order


# --- Tool implementations, reading from the single order dict we just built ---
_CURRENT_ORDER = {}


def get_order_details(order_id: str) -> dict:
    o = _CURRENT_ORDER
    return {
        "order_id": order_id,
        "customer_segment": o["customer_segment"],
        "product_category": o["product_category"],
        "avg_order_value_usd": o["avg_order_value_usd"],
        "refund_amount_requested_usd": o["refund_amount_requested_usd"],
        "return_reason": o["return_reason"],
        "days_to_return": o["days_to_return"],
        "payment_method": o["payment_method"],
        "has_photo_evidence": bool(o["photo_evidence_provided"]),
    }


def get_customer_history(order_id: str) -> dict:
    o = _CURRENT_ORDER
    return {
        "account_age_days": o["account_age_days"],
        "total_orders_lifetime": o["total_orders_lifetime"],
        "total_returns_lifetime": o["total_returns_lifetime"],
        "return_rate_pct": o["return_rate_pct"],
        "previous_dispute_count": o["previous_dispute_count"],
        "customer_support_contacts": o["customer_support_contacts"],
        "multiple_accounts_flag": bool(o["multiple_accounts_flag"]),
    }


def get_risk_score(order_id: str) -> dict:
    o = _CURRENT_ORDER
    row = pd.DataFrame([{col: o[col] for col in FEATURE_COLUMNS}])
    proba = MODEL.predict_proba(row)[0][1]
    return {"order_id": order_id, "abuse_probability": round(float(proba), 4)}


_NETWORK_DF = None


def check_cross_merchant_pattern(order_id: str) -> dict:
    """Checks the payment network for this order's refund account being
    reused across multiple, unrelated merchants - only possible for real
    orders looked up from test.csv (custom hand-built orders have no
    network record, since they don't correspond to a real simulated order)."""
    global _NETWORK_DF
    if _NETWORK_DF is None:
        try:
            _NETWORK_DF = pd.read_csv("cross_merchant_network.csv")
        except FileNotFoundError:
            return {"error": "cross_merchant_network.csv not found - run simulate_cross_merchant_data.py first"}

    match = _NETWORK_DF[_NETWORK_DF["order_id"] == order_id]
    if match.empty:
        return {"result": "No network record for this order (likely a hand-built custom order, not a real simulated one)."}

    account = match.iloc[0]["refund_destination_account"]
    this_merchant = match.iloc[0]["merchant_id"]
    same_account_rows = _NETWORK_DF[_NETWORK_DF["refund_destination_account"] == account]
    distinct_merchants = sorted(same_account_rows["merchant_id"].unique().tolist())

    return {
        "order_id": order_id,
        "refund_destination_account": account,
        "this_order_merchant": this_merchant,
        "distinct_merchants_using_this_account": len(distinct_merchants),
        "merchants": distinct_merchants,
        "cross_merchant_pattern_detected": len(distinct_merchants) > 1,
    }


def analyze_return_photo(order_id: str) -> dict:
    photo_path = _CURRENT_ORDER.get("_photo_path")
    if not photo_path:
        return {"result": "No photo was submitted with this return."}
    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "This is a photo submitted as evidence for a product return. "
                "Describe what it shows and note anything inconsistent with "
                "a genuine defective-item return."
            ),
            "images": [photo_path],
        }],
        options=DETERMINISTIC_OPTIONS,
    )
    return {"order_id": order_id, "photo_analysis": response["message"]["content"]}


DECISION_LOG = []


def log_decision(order_id: str, action: str, reasoning: str) -> dict:
    entry = {"order_id": order_id, "action": action, "reasoning": reasoning}
    DECISION_LOG.append(entry)
    return {"status": "logged", "entry": entry}


TOOL_FUNCTIONS = {
    "get_order_details": get_order_details,
    "get_customer_history": get_customer_history,
    "get_risk_score": get_risk_score,
    "check_cross_merchant_pattern": check_cross_merchant_pattern,
    "analyze_return_photo": analyze_return_photo,
    "log_decision": log_decision,
}

TOOLS = [
    {"type": "function", "function": {"name": "get_order_details", "description": "Get order-level details.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "get_customer_history", "description": "Get customer account/return history.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "get_risk_score", "description": "Get ML abuse probability (0-1).",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "check_cross_merchant_pattern", "description": "Check if this order's refund destination account is shared across multiple merchants - a payment-network-level signal no single merchant can see.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "analyze_return_photo", "description": "Analyze the submitted return photo, if any.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "log_decision", "description": "Record final decision + reasoning. Call once, last.",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string"},
         "action": {"type": "string", "enum": ["approve", "flag_for_review", "request_more_evidence"]},
         "reasoning": {"type": "string"}}, "required": ["order_id", "action", "reasoning"]}}},
]

SYSTEM_PROMPT = """You are a Return Risk Verifier agent for an e-commerce platform.
Investigate using the tools available, then decide one of THREE actions:
- "approve": let the return proceed
- "flag_for_review": likely abuse, hold for human review
- "request_more_evidence": signals are genuinely mixed or evidence is
  incomplete - don't force a binary call

Real uncertainty comes from evidence gaps or signal conflicts, not the raw ML
score alone (which is usually confidently near 0 or 1 on this data):
- Return reason implies damage/defect but has_photo_evidence is false ->
  request_more_evidence
- ML abuse_probability is low BUT cross_merchant_pattern_detected=true
  (signals disagree) -> request_more_evidence
- Otherwise, when signals agree and evidence is sufficient: abuse_probability
  above 0.43 (see cost_threshold.py) -> flag_for_review;
  cross_merchant_pattern_detected=true -> flag_for_review regardless of ML
  score (payment-network signal no single merchant could see); otherwise ->
  approve. If photo analysis suggests the item doesn't match a genuine
  defective-item claim, treat that as a strong signal toward flagging too.

Always call check_cross_merchant_pattern. Always call log_decision exactly
once as your final action, with reasoning a human could audit, noting which
signals agreed, conflicted, or were missing.
"""


def policy_decision(order_id: str) -> dict:
    """Fetch this order's evidence with the local tools, then apply the shared
    policy from decision_policy.py (the same function every entry point uses)."""
    risk = get_risk_score(order_id)
    network = check_cross_merchant_pattern(order_id)
    details = get_order_details(order_id)
    return decide(
        abuse_probability=risk.get("abuse_probability"),
        cross_merchant_detected=network.get("cross_merchant_pattern_detected"),
        return_reason=details.get("return_reason"),
        has_photo_evidence=details.get("has_photo_evidence"),
        distinct_merchants=network.get("distinct_merchants_using_this_account"),
    )


def run_agent(order_id="TEST-ORDER-1"):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Review order {order_id} and decide whether to approve or flag it for review."},
    ]
    for _ in range(8):
        response = ollama.chat(model=TEXT_MODEL, messages=messages, tools=TOOLS,
                               options=DETERMINISTIC_OPTIONS)
        message = response["message"]
        messages.append(message)
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            break
        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            if isinstance(args, str):
                args = json.loads(args)
            fn = TOOL_FUNCTIONS[name]
            valid_params = set(inspect.signature(fn).parameters)
            args = {k: v for k, v in args.items() if k in valid_params}
            result = fn(**args)
            messages.append({"role": "tool", "content": json.dumps(result), "name": name})
        if DECISION_LOG:
            break

    # --- Policy enforcement (same shared policy as every other entry point) --
    agent_decision = DECISION_LOG[-1] if DECISION_LOG else None
    policy = policy_decision(order_id)
    action, reasoning = reconcile(agent_decision, policy, order_id)

    if agent_decision is not None and action == agent_decision["action"]:
        return agent_decision

    return log_decision(order_id, action, reasoning)["entry"]


if __name__ == "__main__":
    print("=== Manual Agent Test ===")
    print("1. Look up a real order by ID (from test.csv)")
    print("2. Build a custom order by hand")
    mode = input("Choose [1]: ").strip() or "1"

    real_order_id = None
    if mode == "2":
        _CURRENT_ORDER = build_custom_order()  # noqa
        order_id_to_use = "TEST-ORDER-1"
    else:
        # Forgiving: if they typed an order ID directly instead of "1", use it
        preset_id = mode if mode not in ("1", "") else None
        order, real_order_id = lookup_real_order(preset_id)
        while order is None:
            order, real_order_id = lookup_real_order()
        _CURRENT_ORDER = order  # noqa
        order_id_to_use = real_order_id
    print("\nAgent is reviewing your order now (this may take a minute)...\n")
    decision = run_agent(order_id_to_use)
    print("\n=== DECISION ===")
    print(json.dumps(decision, indent=2))
