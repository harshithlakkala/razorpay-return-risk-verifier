"""
risk_agent_local.py
FREE version of the return-abuse risk VERIFIER agent.
Runs entirely on your own machine using Ollama (open-source local model) -
no API key, no billing, no internet required once the model is downloaded.

SETUP (one-time):
  1. Install Ollama: https://ollama.com/download  (free, Windows/Mac/Linux)
  2. In PowerShell, run: ollama pull qwen2.5
     (downloads a ~4.7GB model that supports tool-calling - one time only)
  3. pip install ollama pandas scikit-learn joblib

USAGE:
  python risk_agent_local.py --demo
  python risk_agent_local.py --evaluate 100
  (start with a SMALL number for --evaluate since local models are slower
   than the cloud API - each order can take 10-30 seconds on a laptop CPU)
"""
import argparse
import inspect
import json
import joblib
import pandas as pd
from datetime import datetime
import ollama

from decision_policy import decide, reconcile, DECISION_THRESHOLD, VALID_ACTIONS

# The specialised model built from Modelfile.risk-agent: base qwen2.5 with the
# decision policy, worked examples, and deterministic parameters baked in.
# Rebuild it with:  ollama create risk-agent -f Modelfile.risk-agent
# Set to "qwen2.5" to compare against the unspecialised base model.
OLLAMA_MODEL = "risk-agent"

# A fraud decision must be reproducible: the same order, with the same evidence,
# must always produce the same verdict, or the audit log can't be defended.
# Default sampling (temperature ~0.8) made the agent give opposite verdicts on
# identical input (ORD2013623 was approved in some runs, flagged in another).
# temperature=0 forces greedy decoding; top_p/top_k pinned so no sampling path
# remains, and a fixed seed makes any residual tie-breaking reproducible.
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


def load_orders_from_df(df: pd.DataFrame):
    for _, row in df.iterrows():
        _ORDERS_DB[row["order_id"]] = row.to_dict()


# ---------------- TOOL IMPLEMENTATIONS (identical logic to the API version) ----------------

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


_NETWORK_DF = None


def check_cross_merchant_pattern(order_id: str) -> dict:
    """
    THIS IS THE CORE RAZORPAY-SPECIFIC SIGNAL: checks whether this order's
    refund destination account also appears under OTHER, unrelated merchants.
    A single merchant can never see this - they only see their own orders.
    A payment gateway sitting across many merchants can.
    """
    global _NETWORK_DF
    if _NETWORK_DF is None:
        _NETWORK_DF = pd.read_csv("cross_merchant_network.csv")

    match = _NETWORK_DF[_NETWORK_DF["order_id"] == order_id]
    if match.empty:
        return {"error": f"No network record for order {order_id}"}

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
    "check_cross_merchant_pattern": check_cross_merchant_pattern,
    "log_decision": log_decision,
}

# Ollama uses OpenAI-style tool definitions
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Get the order-level details for a return request.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_customer_history",
            "description": "Get the requesting customer's account and return history.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_risk_score",
            "description": "Run the trained ML risk model to get a probability (0-1) that this return is abusive.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_cross_merchant_pattern",
            "description": "Check the payment network (across ALL merchants, not just this one) for whether this order's refund destination account is shared with orders from other merchants - a signal only a payment gateway can see, invisible to any single merchant.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_decision",
            "description": "Record the final decision and reasoning for this order. Call exactly once, as the LAST action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "action": {"type": "string", "enum": ["approve", "flag_for_review", "request_more_evidence"]},
                    "reasoning": {"type": "string"},
                },
                "required": ["order_id", "action", "reasoning"],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a Return Risk Verifier agent for an e-commerce platform.

For each order, investigate using the tools available: check the order details,
the customer's history, the ML risk score, and the cross-merchant pattern
check. Then decide one of THREE actions:
- "approve": let the return proceed normally
- "flag_for_review": hold it for a human to manually review, likely abuse
- "request_more_evidence": signals are genuinely mixed or evidence is
  incomplete - don't force a binary call, ask the customer for more proof
  (e.g. a photo) before deciding

On this dataset, the ML risk score is highly confident (near 0 or near 1,
rarely in between), so probability alone rarely signals genuine uncertainty.
Real uncertainty instead comes from EVIDENCE GAPS or SIGNAL CONFLICTS:
- The return reason implies physical damage/defect (e.g. "damaged", "broken",
  "defective") but has_photo_evidence is false -> request_more_evidence
  (can't confirm or deny a damage claim with no photo)
- The ML abuse_probability is low (looks legitimate) BUT
  check_cross_merchant_pattern reports cross_merchant_pattern_detected=true
  (a payment-network red flag) -> request_more_evidence, since these two
  signals disagree and the network signal deserves a closer look before
  either approving or flagging outright
- Otherwise, when signals AGREE (ML score and cross-merchant check point the
  same direction, and evidence is sufficient), decide directly:
  - abuse_probability above 0.43 (derived by sweeping cutoffs against actual
    FP/FN dollar costs - see cost_threshold.py) -> flag_for_review
  - cross_merchant_pattern_detected=true -> flag_for_review regardless of
    ML score, since it's a payment-network-level signal no single merchant
    could ever see on their own
  - otherwise -> approve

Always call check_cross_merchant_pattern for every order. Always call
log_decision exactly once as your final action, with reasoning a human
could audit, explaining which signals agreed, conflicted, or were missing.
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


def run_agent_on_order(order_id: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Review order {order_id} and decide whether to approve or flag it for review."},
    ]
    final_decision = None

    for _ in range(6):
        response = ollama.chat(model=OLLAMA_MODEL, messages=messages, tools=TOOLS,
                               options=DETERMINISTIC_OPTIONS)
        message = response["message"]
        messages.append(message)

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            break

        for call in tool_calls:
            name = call["function"]["name"]
            args = call["function"]["arguments"]
            if isinstance(args, str):  # some models return args as a JSON string
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

    # --- Policy enforcement -------------------------------------------------
    # The model's verdict is reconciled against the shared policy. Where they
    # disagree the policy wins and the override is recorded, so the audit log
    # shows both what the model concluded and what was actually done. This is
    # why the system does not depend on a 7B local model reliably obeying a
    # paragraph of prompt instructions to move money correctly.
    policy = policy_decision(order_id)
    action, reasoning = reconcile(final_decision, policy, order_id)

    if final_decision is not None and action == final_decision["action"]:
        return final_decision  # already logged by the model's own tool call

    return log_decision(order_id, action, reasoning)["entry"]


def run_demo():
    df = pd.read_csv("test.csv").sample(5, random_state=1)
    load_orders_from_df(df)
    for order_id in df["order_id"]:
        print(f"\n--- Reviewing {order_id} ---")
        decision = run_agent_on_order(order_id)
        print(json.dumps(decision, indent=2))


def run_evaluation(n: int):
    df = pd.read_csv("test.csv").sample(n, random_state=1)
    load_orders_from_df(df)
    y_true, actions = [], []
    for i, (_, row) in enumerate(df.iterrows()):
        print(f"Processing order {i+1}/{n}...")
        decision = run_agent_on_order(row["order_id"])
        y_true.append(int(row["is_abuse"]))
        actions.append(decision["action"] if decision else "approve")

    from collections import Counter
    print("\nAction distribution:", dict(Counter(actions)))

    # For precision/recall (binary metrics), treat request_more_evidence as
    # its own outcome, NOT silently folded into approve or flag - report it
    # separately so it doesn't inflate or deflate the binary numbers.
    binary_pairs = [(t, a) for t, a in zip(y_true, actions) if a != "request_more_evidence"]
    escalated = sum(1 for a in actions if a == "request_more_evidence")
    print(f"Orders escalated to request_more_evidence: {escalated} of {n} "
          f"(excluded from binary precision/recall below, reported separately)")

    if binary_pairs:
        y_true_bin = [t for t, a in binary_pairs]
        y_pred_bin = [1 if a == "flag_for_review" else 0 for t, a in binary_pairs]
        from sklearn.metrics import precision_score, recall_score, confusion_matrix
        print("\nAgent decision precision (on non-escalated orders):", precision_score(y_true_bin, y_pred_bin))
        print("Agent decision recall (on non-escalated orders):", recall_score(y_true_bin, y_pred_bin))
        print("Confusion matrix:\n", confusion_matrix(y_true_bin, y_pred_bin))

    # Of the escalated ones, report how many were actually abuse vs legit,
    # since that tells you whether escalation is catching genuinely hard cases
    if escalated:
        escalated_true = [t for t, a in zip(y_true, actions) if a == "request_more_evidence"]
        abuse_rate_in_escalated = sum(escalated_true) / len(escalated_true)
        print(f"Of escalated orders, {sum(escalated_true)} of {escalated} "
              f"({abuse_rate_in_escalated:.0%}) were actually abuse - a rate near 50%"
              f" would suggest escalation is correctly targeting genuinely hard cases.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true")
    parser.add_argument("--evaluate", type=int, default=0)
    args = parser.parse_args()

    if args.demo:
        run_demo()
    elif args.evaluate:
        run_evaluation(args.evaluate)
    else:
        print("Use --demo or --evaluate N")
