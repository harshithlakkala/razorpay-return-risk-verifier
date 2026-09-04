"""
risk_agent_with_photos.py
BONUS extension of risk_agent_local.py: adds a 5th tool that lets the agent
look at a photo of the returned item and reason about it visually.

IMPORTANT: This is additive, not a replacement. If this file or the vision
model gives you trouble, risk_agent_local.py (your working, measured
submission) is completely unaffected and still works on its own.

SETUP (in addition to what you already have):
  ollama pull moondream
  (a small, fast vision-capable model - much lighter than llava, better for
   a laptop CPU. moondream is ~1.7GB.)

Since the real dataset only has a photo_evidence_provided TRUE/FALSE flag
(no actual photo files), this demo maps that flag to one of the placeholder
images in sample_photos/ so you can show the mechanism working end-to-end.
Be upfront about this in your demo: it's illustrative of the architecture,
not a claim that these specific images are "real" evidence.

Usage:
  python risk_agent_with_photos.py --demo
"""
import inspect
import json
import joblib
import pandas as pd
from datetime import datetime
import ollama
import random

from decision_policy import decide, reconcile

# Specialised model built from Modelfile.risk-agent. Rebuild with:
#   ollama create risk-agent -f Modelfile.risk-agent
TEXT_MODEL = "risk-agent"
VISION_MODEL = "moondream"

# Same order + same evidence must always yield the same verdict (see
# risk_agent_local.py for why). Applied to the vision call too, so a photo
# gets described the same way every time it's examined.
DETERMINISTIC_OPTIONS = {"temperature": 0, "top_p": 1, "top_k": 1, "seed": 42}
MODEL = joblib.load("risk_model.joblib")
AUDIT_LOG_PATH = "audit_log.txt"

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

_ORDERS_DB = {}

# Demo-only: since we don't have real photos, randomly assign a placeholder
# to orders that have photo_evidence_provided = True, so the tool has
# something real to look at.
PLACEHOLDER_PHOTOS = [
    "sample_photos/genuine_damage.png",
    "sample_photos/suspicious_unused.png",
    "sample_photos/mismatched_item.png",
]


def load_orders_from_df(df: pd.DataFrame):
    for _, row in df.iterrows():
        order = row.to_dict()
        if order.get("photo_evidence_provided"):
            order["_demo_photo_path"] = random.choice(PLACEHOLDER_PHOTOS)
        _ORDERS_DB[row["order_id"]] = order


def get_order_details(order_id: str) -> dict:
    order = _ORDERS_DB.get(order_id)
    if order is None:
        return {"error": f"No order found with id {order_id}"}
    return {
        "order_id": order_id,
        "customer_segment": order["customer_segment"],
        "product_category": order["product_category"],
        "avg_order_value_usd": order["avg_order_value_usd"],
        "refund_amount_requested_usd": order["refund_amount_requested_usd"],
        "return_reason": order["return_reason"],
        "days_to_return": order["days_to_return"],
        "payment_method": order["payment_method"],
        "has_photo_evidence": bool(order.get("photo_evidence_provided")),
    }


def get_customer_history(order_id: str) -> dict:
    order = _ORDERS_DB.get(order_id)
    if order is None:
        return {"error": f"No order found with id {order_id}"}
    return {
        "account_age_days": order["account_age_days"],
        "total_orders_lifetime": order["total_orders_lifetime"],
        "total_returns_lifetime": order["total_returns_lifetime"],
        "return_rate_pct": order["return_rate_pct"],
        "previous_dispute_count": order["previous_dispute_count"],
        "customer_support_contacts": order["customer_support_contacts"],
        "multiple_accounts_flag": bool(order["multiple_accounts_flag"]),
    }


def get_risk_score(order_id: str) -> dict:
    order = _ORDERS_DB.get(order_id)
    if order is None:
        return {"error": f"No order found with id {order_id}"}
    row = pd.DataFrame([{col: order[col] for col in FEATURE_COLUMNS}])
    proba = MODEL.predict_proba(row)[0][1]
    return {"order_id": order_id, "abuse_probability": round(float(proba), 4)}


def analyze_return_photo(order_id: str) -> dict:
    """BONUS tool: looks at the photo submitted with the return, if any."""
    order = _ORDERS_DB.get(order_id)
    if order is None:
        return {"error": f"No order found with id {order_id}"}
    photo_path = order.get("_demo_photo_path")
    if not photo_path:
        return {"result": "No photo was submitted with this return."}

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[{
            "role": "user",
            "content": (
                "This is a photo submitted as evidence for a product return. "
                "Describe what it shows, and note anything that looks "
                "inconsistent with a genuine defective-item return (e.g. "
                "tags still attached, no visible damage, mismatched item)."
            ),
            "images": [photo_path],
        }],
        options=DETERMINISTIC_OPTIONS,
    )
    return {"order_id": order_id, "photo_analysis": response["message"]["content"]}


def log_decision(order_id: str, action: str, reasoning: str) -> dict:
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "order_id": order_id,
        "action": action,
        "reasoning": reasoning,
    }
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return {"status": "logged", "entry": entry}


TOOL_FUNCTIONS = {
    "get_order_details": get_order_details,
    "get_customer_history": get_customer_history,
    "get_risk_score": get_risk_score,
    "analyze_return_photo": analyze_return_photo,
    "log_decision": log_decision,
}

TOOLS = [
    {"type": "function", "function": {
        "name": "get_order_details",
        "description": "Get the order-level details for a return request.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_customer_history",
        "description": "Get the requesting customer's account and return history.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    }},
    {"type": "function", "function": {
        "name": "get_risk_score",
        "description": "Run the trained ML risk model to get a probability (0-1) that this return is abusive.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    }},
    {"type": "function", "function": {
        "name": "analyze_return_photo",
        "description": "If the customer submitted a photo with their return, analyze it for signs of genuine damage vs staged/unused/mismatched items.",
        "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]},
    }},
    {"type": "function", "function": {
        "name": "log_decision",
        "description": "Record the final decision and reasoning. Call exactly once, as the LAST action.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "action": {"type": "string", "enum": ["approve", "flag_for_review"]},
                "reasoning": {"type": "string"},
            },
            "required": ["order_id", "action", "reasoning"],
        },
    }},
]

SYSTEM_PROMPT = """You are a Return Risk Verifier agent for an e-commerce platform.

For each order, investigate using the tools available: order details, customer
history, the ML risk score, and (if photo evidence was submitted) the photo
analysis tool. Then decide: "approve" or "flag_for_review".

Guardrail: if abuse_probability is above 0.43 (cost-optimal threshold, see
cost_threshold.py), lean toward flag_for_review
unless context clearly explains it. If a photo was submitted and the analysis
suggests it doesn't match a genuine defective-item claim (e.g. tags still
attached, no damage visible, wrong item), that is a strong signal to flag
regardless of the ML score. Always call log_decision exactly once as your
final action, with reasoning a human could audit.
"""


def policy_decision(order_id: str) -> dict:
    """Fetch this order's evidence with the local tools, then apply the shared
    policy from decision_policy.py (the same function every entry point uses)."""
    risk = get_risk_score(order_id)
    details = get_order_details(order_id)
    return decide(
        abuse_probability=risk.get("abuse_probability"),
        cross_merchant_detected=False,  # this variant has no network tool
        return_reason=details.get("return_reason"),
        has_photo_evidence=details.get("has_photo_evidence"),
    )


def run_agent_on_order(order_id: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Review order {order_id} and decide whether to approve or flag it for review."},
    ]
    final_decision = None

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
            if name == "log_decision":
                final_decision = result["entry"]
            messages.append({"role": "tool", "content": json.dumps(result), "name": name})

        if final_decision is not None:
            break

    # --- Policy enforcement (same shared policy as every other entry point) --
    policy = policy_decision(order_id)
    action, reasoning = reconcile(final_decision, policy, order_id)

    if final_decision is not None and action == final_decision["action"]:
        return final_decision  # already logged by the model's own tool call

    return log_decision(order_id, action, reasoning)["entry"]


def run_demo():
    df = pd.read_csv("test.csv")
    # Prefer picking a few orders that actually have photo evidence, so the
    # bonus tool gets exercised in the demo
    with_photo = df[df["photo_evidence_provided"] == 1].sample(3, random_state=2)
    without_photo = df[df["photo_evidence_provided"] == 0].sample(2, random_state=2)
    sample = pd.concat([with_photo, without_photo])

    load_orders_from_df(sample)
    for order_id in sample["order_id"]:
        print(f"\n--- Reviewing {order_id} ---")
        decision = run_agent_on_order(order_id)
        print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    run_demo()
