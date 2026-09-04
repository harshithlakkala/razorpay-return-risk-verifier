"""
app.py
Runs your Return-Risk Verifier agent as a local WEBPAGE instead of a terminal
script. Same agent, same tools, same logic - just reachable through a browser
like http://localhost:5000 instead of typing PowerShell commands each time.

This runs on YOUR machine (like a local website only you can see) - it is not
deployed publicly on the internet. That's normal and expected for a hackathon
demo; showing "type an order ID into a webpage and watch the agent reason
live" is exactly the kind of thing a judge or pitch video wants to see.

SETUP (one-time):
  pip install flask ollama pandas scikit-learn joblib

USAGE:
  python app.py
  Then open http://localhost:5000 in your browser.
"""
import inspect
import json
import joblib
import pandas as pd
import ollama
from datetime import datetime
from flask import Flask, jsonify, request, Response

from decision_policy import decide, reconcile

app = Flask(__name__)

# Specialised model built from Modelfile.risk-agent (policy + worked examples
# + deterministic parameters baked in). Rebuild with:
#   ollama create risk-agent -f Modelfile.risk-agent
TEXT_MODEL = "risk-agent"
VISION_MODEL = "moondream"

# Same order + same evidence must always yield the same verdict (see
# risk_agent_local.py for why). Greedy decoding, no sampling, fixed seed.
DETERMINISTIC_OPTIONS = {"temperature": 0, "top_p": 1, "top_k": 1, "seed": 42}
MODEL = joblib.load("risk_model.joblib")
TEST_DF = pd.read_csv("test.csv").set_index("order_id")
try:
    RAZORPAY_IDS_DF = pd.read_csv("razorpay_ids.csv").set_index("order_id")
    # Reverse lookup: a Razorpay employee has the payment_id, not the
    # merchant's internal order number - so payment_id is the real search key
    PAYMENT_TO_ORDER = RAZORPAY_IDS_DF.reset_index().set_index("razorpay_payment_id")["order_id"].to_dict()
except FileNotFoundError:
    RAZORPAY_IDS_DF = None
    PAYMENT_TO_ORDER = {}
try:
    NETWORK_DF = pd.read_csv("cross_merchant_network.csv")
except FileNotFoundError:
    NETWORK_DF = None

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

# ---------------- TOOL IMPLEMENTATIONS ----------------

def get_order_details(order_id: str) -> dict:
    if order_id not in TEST_DF.index:
        return {"error": f"No order found with id {order_id}"}
    o = TEST_DF.loc[order_id]
    result = {
        "order_id": order_id,
        "customer_segment": o["customer_segment"],
        "product_category": o["product_category"],
        "avg_order_value_usd": float(o["avg_order_value_usd"]),
        "refund_amount_requested_usd": float(o["refund_amount_requested_usd"]),
        "return_reason": o["return_reason"],
        "days_to_return": int(o["days_to_return"]),
        "payment_method": o["payment_method"],
        "has_photo_evidence": bool(o["photo_evidence_provided"]),
    }
    if RAZORPAY_IDS_DF is not None and order_id in RAZORPAY_IDS_DF.index:
        r = RAZORPAY_IDS_DF.loc[order_id]
        result["razorpay_order_id"] = r["razorpay_order_id"]
        result["razorpay_payment_id"] = r["razorpay_payment_id"]
        result["razorpay_refund_id"] = r["razorpay_refund_id"]
    return result


def get_customer_history(order_id: str) -> dict:
    if order_id not in TEST_DF.index:
        return {"error": f"No order found with id {order_id}"}
    o = TEST_DF.loc[order_id]
    return {
        "account_age_days": int(o["account_age_days"]),
        "total_orders_lifetime": int(o["total_orders_lifetime"]),
        "total_returns_lifetime": int(o["total_returns_lifetime"]),
        "return_rate_pct": float(o["return_rate_pct"]),
        "previous_dispute_count": int(o["previous_dispute_count"]),
        "customer_support_contacts": int(o["customer_support_contacts"]),
        "multiple_accounts_flag": bool(o["multiple_accounts_flag"]),
    }


def get_risk_score(order_id: str) -> dict:
    if order_id not in TEST_DF.index:
        return {"error": f"No order found with id {order_id}"}
    row = pd.DataFrame([{col: TEST_DF.loc[order_id, col] for col in FEATURE_COLUMNS}])
    proba = MODEL.predict_proba(row)[0][1]
    return {"order_id": order_id, "abuse_probability": round(float(proba), 4)}


def check_cross_merchant_pattern(order_id: str) -> dict:
    if NETWORK_DF is None:
        return {"result": "Network simulation data not found."}
    match = NETWORK_DF[NETWORK_DF["order_id"] == order_id]
    if match.empty:
        return {"result": "No network record for this order."}
    account = match.iloc[0]["refund_destination_account"]
    this_merchant = match.iloc[0]["merchant_id"]
    same = NETWORK_DF[NETWORK_DF["refund_destination_account"] == account]
    merchants = sorted(same["merchant_id"].unique().tolist())
    return {
        "order_id": order_id,
        "refund_destination_account": account,
        "this_order_merchant": this_merchant,
        "distinct_merchants_using_this_account": len(merchants),
        "merchants": merchants,
        "cross_merchant_pattern_detected": len(merchants) > 1,
    }


_DECISION_HOLDER = {}


def log_decision(order_id: str, action: str, reasoning: str) -> dict:
    entry = {"timestamp": datetime.utcnow().isoformat(), "order_id": order_id, "action": action, "reasoning": reasoning}
    with open("audit_log.txt", "a") as f:
        f.write(json.dumps(entry) + "\n")
    _DECISION_HOLDER["decision"] = entry
    return {"status": "logged", "entry": entry}


TOOL_FUNCTIONS = {
    "get_order_details": get_order_details,
    "get_customer_history": get_customer_history,
    "get_risk_score": get_risk_score,
    "check_cross_merchant_pattern": check_cross_merchant_pattern,
    "log_decision": log_decision,
}

TOOLS = [
    {"type": "function", "function": {"name": "get_order_details", "description": "Get order-level details.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "get_customer_history", "description": "Get customer account/return history.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "get_risk_score", "description": "Get ML abuse probability (0-1).",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "check_cross_merchant_pattern", "description": "Check if refund account is shared across merchants.",
     "parameters": {"type": "object", "properties": {"order_id": {"type": "string"}}, "required": ["order_id"]}}},
    {"type": "function", "function": {"name": "log_decision", "description": "Record final decision + reasoning. Call once, last.",
     "parameters": {"type": "object", "properties": {
         "order_id": {"type": "string"},
         "action": {"type": "string", "enum": ["approve", "flag_for_review", "request_more_evidence"]},
         "reasoning": {"type": "string"}}, "required": ["order_id", "action", "reasoning"]}}},
]

SYSTEM_PROMPT = """You are a Return Risk Verifier agent for an e-commerce platform.
Investigate using the tools available, then decide one of THREE actions:
approve, flag_for_review, or request_more_evidence (for mixed/incomplete signals).
If abuse_probability is above 0.43, lean toward flag_for_review unless context
explains it. Always call check_cross_merchant_pattern - a true result is a
strong flag signal regardless of ML score. Always call log_decision exactly
once as your final action, with reasoning a human could audit.
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
    _DECISION_HOLDER.pop("decision", None)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Review order {order_id} and decide."},
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
        if "decision" in _DECISION_HOLDER:
            break

    # --- Policy enforcement -------------------------------------------------
    # Same shared policy the CLI agent and the evaluator use. The model's
    # verdict is reconciled against it; where they disagree the policy wins and
    # the override is recorded in the audit log.
    agent_decision = _DECISION_HOLDER.get("decision")
    policy = policy_decision(order_id)
    action, reasoning = reconcile(agent_decision, policy, order_id)

    if agent_decision is not None and action == agent_decision["action"]:
        return agent_decision  # already logged by the model's own tool call

    return log_decision(order_id, action, reasoning)["entry"]


# ---------------- WEB ROUTES ----------------

def resolve_order_id(search_id: str):
    """Resolve either a Razorpay payment ID (pay_...) or a merchant order
    reference to the internal order_id used to look up test data."""
    if search_id.startswith("pay_") and RAZORPAY_IDS_DF is not None:
        match = RAZORPAY_IDS_DF[RAZORPAY_IDS_DF["razorpay_payment_id"] == search_id]
        if not match.empty:
            return match.index[0]
        return None
    if search_id in TEST_DF.index:
        return search_id
    return None


def review_order(order_id: str) -> dict:
    """The actual integration point: given an order, run the full agent
    investigation and return a decision. Both the human search box and the
    machine-triggered webhook below call this exact same function - a real
    system would have Razorpay's refund pipeline call this directly (or an
    equivalent internal function call, no HTTP needed) the moment a return
    is requested, with no human ever typing an ID into a page."""
    details = get_order_details(order_id)
    history = get_customer_history(order_id)
    risk = get_risk_score(order_id)
    network = check_cross_merchant_pattern(order_id)
    actual_label = "ABUSE" if int(TEST_DF.loc[order_id, "is_abuse"]) == 1 else "LEGITIMATE"
    decision = run_agent_on_order(order_id)
    return {
        "order_id": order_id,
        "actual_label": actual_label,
        "details": details,
        "history": history,
        "risk": risk,
        "network": network,
        "decision": decision,
    }


@app.route("/api/review")
def api_review():
    """HUMAN-FACING: a person types an ID into the search box on the webpage."""
    search_id = request.args.get("order_id", "").strip()
    order_id = resolve_order_id(search_id)
    if order_id is None:
        return jsonify({"error": f"No payment found matching '{search_id}'. "
                                  f"Use a Razorpay payment ID like pay_xxxxxxxxxxxxxx."}), 404
    return jsonify(review_order(order_id))


@app.route("/webhook/return-requested", methods=["POST"])
def webhook_return_requested():
    """
    MACHINE-FACING: this is what real integration looks like. No human types
    anything - Razorpay's own refund-processing system would call this
    automatically, the instant a customer requests a return, the same way
    any internal service calls another. This simulates that trigger.

    Example of what Razorpay's system would send automatically:
      POST /webhook/return-requested
      { "razorpay_payment_id": "pay_y9dzm9nWb2UF5Z" }

    Try it yourself from PowerShell:
      Invoke-RestMethod -Uri http://localhost:5000/webhook/return-requested `
        -Method Post -ContentType "application/json" `
        -Body '{"razorpay_payment_id": "pay_y9dzm9nWb2UF5Z"}'
    """
    payload = request.get_json(silent=True) or {}
    search_id = payload.get("razorpay_payment_id") or payload.get("order_id", "")
    search_id = str(search_id).strip()

    order_id = resolve_order_id(search_id)
    if order_id is None:
        return jsonify({"error": f"No payment found matching '{search_id}'"}), 404

    result = review_order(order_id)

    # In a real integration, THIS is the part that would matter most: instead
    # of returning JSON for a page to render, the decision would be written
    # back into Razorpay's own systems - e.g. holding the payout, opening a
    # fraud-ops ticket, or auto-releasing the refund. We simulate that here
    # by returning what that downstream action WOULD be, explicitly.
    action = result["decision"]["action"] if result["decision"] else "approve"
    downstream_action = {
        "approve": "refund released automatically, no human involved",
        "flag_for_review": "payout held, ticket created in fraud-ops queue",
        "request_more_evidence": "customer notified, asked to submit photo evidence before refund proceeds",
    }.get(action, "unknown action")

    result["simulated_downstream_action"] = downstream_action
    return jsonify(result)


@app.route("/")
def home():
    return Response(INDEX_HTML, mimetype="text/html")


INDEX_HTML = """
<!DOCTYPE html>
<html><head><meta charset="UTF-8"><title>Return-Risk Verifier</title>
<style>
  :root{--bg:#0B0E14;--panel:#131826;--line:rgba(255,255,255,0.08);--text:#E7E9EE;
  --muted:#8891A3;--amber:#D9A441;--teal:#3FA796;--mono:"SF Mono",Consolas,Menlo,monospace;
  --sans:-apple-system,"Segoe UI",Inter,Helvetica,Arial,sans-serif;}
  *{box-sizing:border-box;}
  body{background:var(--bg);color:var(--text);font-family:var(--sans);margin:0;padding:40px 20px;}
  .wrap{max-width:640px;margin:0 auto;}
  h1{font-family:Georgia,serif;font-weight:500;font-size:28px;margin-bottom:6px;}
  p.sub{color:var(--muted);margin-top:0;}
  .searchbar{display:flex;gap:8px;margin:24px 0;}
  input{flex:1;background:var(--panel);border:1px solid var(--line);color:var(--text);
  padding:12px 14px;border-radius:4px;font-family:var(--mono);font-size:14px;}
  button{background:var(--amber);color:#1a1300;border:none;padding:12px 20px;border-radius:4px;
  font-weight:600;cursor:pointer;font-size:14px;}
  button:disabled{opacity:0.5;cursor:default;}
  .card{background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:20px;margin-top:10px;display:none;}
  .row{display:flex;justify-content:space-between;padding:7px 0;border-bottom:1px solid var(--line);font-size:13.5px;}
  .row .k{color:var(--muted);}
  .row .v{font-family:var(--mono);}
  .verdict{margin-top:14px;padding:14px;border-radius:3px;border-left:3px solid var(--amber);background:rgba(217,164,65,0.08);}
  .verdict.approve{border-color:var(--teal);background:rgba(63,167,150,0.08);}
  .action{font-family:var(--mono);font-weight:600;text-transform:uppercase;font-size:13px;}
  .loading{color:var(--muted);font-family:var(--mono);font-size:13px;margin-top:14px;display:none;}
  .label-badge{font-family:var(--mono);font-size:11px;padding:3px 8px;border-radius:10px;background:var(--line);}
</style></head>
<body>
<div class="wrap">
  <h1>Return-Risk Verifier</h1>
  <p class="sub">Internal Razorpay risk console. Enter a payment ID to investigate a return/refund request.</p>
  <div class="searchbar">
    <input id="orderId" placeholder="e.g. pay_y9dzm9nWb2UF5Z" />
    <button id="go">Review</button>
  </div>
  <div class="loading" id="loading">Agent is investigating (this can take 30-90 seconds on a local model)...</div>
  <div class="card" id="card"></div>
</div>
<script>
const go = document.getElementById('go');
const input = document.getElementById('orderId');
const card = document.getElementById('card');
const loading = document.getElementById('loading');

async function review(){
  const id = input.value.trim();
  if(!id) return;
  card.style.display='none';
  loading.style.display='block';
  go.disabled = true;
  try{
    const res = await fetch('/api/review?order_id=' + encodeURIComponent(id));
    const data = await res.json();
    loading.style.display='none';
    go.disabled = false;
    if(data.error){ card.innerHTML = '<p style="color:#D9A441">' + data.error + '</p>'; card.style.display='block'; return; }
    const d = data.details, h = data.history, r = data.risk, n = data.network, dec = data.decision;
    const isApprove = dec && dec.action === 'approve';
    card.innerHTML = `
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:10px;">
        <strong>${d.razorpay_payment_id || data.order_id}</strong>
        <span class="label-badge">actual label: ${data.actual_label}</span>
      </div>
      <div class="row"><span class="k">Razorpay order ID</span><span class="v">${d.razorpay_order_id || '-'}</span></div>
      <div class="row"><span class="k">Merchant's order ref</span><span class="v">${data.order_id}</span></div>
      <div class="row"><span class="k">Product</span><span class="v">${d.product_category}</span></div>
      <div class="row"><span class="k">Return reason</span><span class="v">${d.return_reason}</span></div>
      <div class="row"><span class="k">Return rate (lifetime)</span><span class="v">${h.return_rate_pct}%</span></div>
      <div class="row"><span class="k">ML abuse probability</span><span class="v">${r.abuse_probability}</span></div>
      <div class="row"><span class="k">Refund destination</span><span class="v">${n.refund_destination_account || '-'}</span></div>
      <div class="row"><span class="k">Cross-merchant pattern</span><span class="v">${n.cross_merchant_pattern_detected}</span></div>
      <div class="verdict ${isApprove ? 'approve' : ''}">
        <div class="action">→ ${dec ? dec.action : 'no decision returned'}</div>
        <div style="margin-top:6px;font-size:13.5px;color:#C7CBD6;">${dec ? dec.reasoning : ''}</div>
      </div>
    `;
    card.style.display='block';
  }catch(e){
    loading.style.display='none';
    go.disabled = false;
    card.innerHTML = '<p style="color:#D9A441">Request failed: ' + e + '</p>';
    card.style.display='block';
  }
}
go.addEventListener('click', review);
input.addEventListener('keydown', e => { if(e.key==='Enter') review(); });
</script>
</body></html>
"""

if __name__ == "__main__":
    print("Open http://localhost:5000 in your browser")
    app.run(debug=False, port=5000)
