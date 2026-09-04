# Return-Risk Verifier — payment-layer refund fraud detection

An agentic verifier for refund/return abuse, built for the **Razorpay AI
Buildathon, Track 2 (AI Risk Manager)**.

For every refund request, an LLM agent investigates using five tools — order
details, customer history, an ML risk score, a cross-merchant payment-network
check, and (optionally) a vision model reading photo evidence — then a decision
policy enforced **in code** returns one of three actions, with a timestamped,
auditable justification every time:

| Action | Meaning |
|---|---|
| `approve` | refund proceeds |
| `flag_for_review` | held for a human, likely abuse |
| `request_more_evidence` | evidence is missing; no verdict is forced |

**Read [`MODEL_AUDIT.md`](MODEL_AUDIT.md) first.** It documents what the
headline numbers do and do not mean, and it is deliberately unflattering.

---

## The idea

Product-condition review ("is this shirt really damaged?") belongs to the
merchant. But refund fraud has a **payment-layer signature no single merchant
can see**: the same refund destination collecting money from several unrelated
merchants in one week. Merchant A sees one odd refund; only the gateway
underneath all of them sees the pattern.

That signal is implemented as `check_cross_merchant_pattern`, and it overrides
the ML score — because the model was never trained on network data, so a low
probability is not evidence against a shared destination.

## Results

Across all 12,000 held-out orders (`python evaluate_policy.py`):

| Action | Count | Abuse | Legitimate |
|---|---|---|---|
| `approve` | 7,631 | 0 | 7,631 |
| `flag_for_review` | 3,517 | 3,517 | 0 |
| `request_more_evidence` | 852 | 71 | 781 |

Precision **1.000**, recall **1.000** on the 11,148 binary decisions; zero
false positives, zero missed abuse, **$0** error cost.

> These numbers reflect the benchmark, not fraud-detection skill. A **two-line
> decision tree** reproduces the ML model perfectly on this data — the labels
> are synthetically generated, so the policy is inverting a generator rather
> than detecting fraud. Full analysis in [`MODEL_AUDIT.md`](MODEL_AUDIT.md).

## What's actually engineered here

The decision rules started in the agent's system prompt — the model was *asked*
to follow them. A 7B local model doesn't reliably obey prose: it invented a
tool argument that crashed a 100-order run, and returned **opposite verdicts on
identical input**.

So the rules moved into [`decision_policy.py`](decision_policy.py) as a pure
function that every entry point imports. The LLM investigates and explains; it
no longer holds the rule. Where model and policy disagree, the policy wins and
the override is written to the audit log.

All inference runs at `temperature=0` with a fixed seed, verified by
`verify_determinism.py`.

Because the policy is code, the agent's action provably equals the policy's
action — which is why 12,000 orders can be evaluated in seconds instead of the
~10 days of local CPU inference a real agent run would need.

---

## Setup

```bash
pip install ollama pandas scikit-learn joblib flask supabase
```

Install [Ollama](https://ollama.com/download), then:

```bash
ollama pull qwen2.5                                  # ~4.7 GB, tool-calling
ollama pull moondream                                # ~1.7 GB, vision (optional)
ollama create risk-agent -f Modelfile.risk-agent     # policy baked in
```

`risk-agent` is stock `qwen2.5` plus a system prompt, five in-context worked
examples, and pinned sampling parameters. **Nothing is fine-tuned** — all three
local models share one identical weight file.

## Run it

```bash
python risk_agent_local.py --demo         # 5 sample orders, see the reasoning
python risk_agent_local.py --evaluate 20  # agent-in-the-loop (slow: ~1-2 min/order)
python evaluate_policy.py                 # all 12,000 orders (seconds)
python verify_determinism.py              # determinism regression test
python app.py                             # web console -> localhost:5000
python user_test.py                       # interactive; needs a real terminal
```

The web console takes a Razorpay payment ID (`pay_...` — see
`razorpay_ids.csv`) and also exposes a machine-facing webhook:

```bash
curl -X POST http://localhost:5000/webhook/return-requested \
  -H "Content-Type: application/json" \
  -d '{"razorpay_payment_id": "pay_y9dzm9nWb2UF5Z"}'
```

### Optional: Postgres instead of CSVs

```bash
# Supabase SQL Editor -> paste schema.sql -> Run
$env:SUPABASE_URL="https://your-project.supabase.co"
$env:SUPABASE_KEY="your-anon-public-key"
python upload_to_supabase.py
python app_supabase.py
```

Identical behaviour; every lookup becomes a real indexed database query. The
ML model stays a local file — models belong in an artifact store, not the
transactional database.

---

## Layout

| File | Purpose |
|---|---|
| `decision_policy.py` | **The policy.** One definition, imported everywhere |
| `risk_agent_local.py` | CLI agent, `--demo` / `--evaluate N` |
| `app.py` / `app_supabase.py` | Web console + webhook (CSV / Postgres) |
| `user_test.py` | Interactive single-order tester |
| `risk_agent_with_photos.py` | Adds the vision tool for photo evidence |
| `train_model.py` | Trains the Random Forest |
| `cost_threshold.py` | Threshold sweep by dollar cost → 0.43 |
| `model_comparison.py` | LogReg / RF / GB, incl. calibration |
| `evaluate_policy.py` | Full 12,000-order evaluation |
| `verify_determinism.py` | Determinism regression test |
| `simulate_cross_merchant_data.py` | Builds the simulated merchant network |
| `generate_razorpay_ids.py` | Razorpay-format IDs for every order |
| `merge_all_data.py` | Joins the three sources → `merged_data.csv` |
| `generate_full_audit_report.py` | Per-decision audit trail (run merge first) |
| `Modelfile.risk-agent` | The specialised model definition |
| `schema.sql` | Postgres schema for the Supabase path |

## Honest limitations

- The benchmark is solvable by a two-line rule; perfect scores reflect that.
- The cross-merchant rule **cannot** produce a false positive here, because the
  simulation only plants shared destinations on already-abusive orders. Real
  shared accounts (families, resellers) would break this.
- Evidence escalation is blunt: 781 legitimate customers get asked for a photo.
- Nothing was fine-tuned; no cross-dataset or adversarial validation.

See [`MODEL_AUDIT.md`](MODEL_AUDIT.md) for the full treatment.

## Scope

Defence-only. The system protects a merchant from loss, never generates
attacks or assists evasion, and **every flagged case goes to a human** before
any action is taken.

Data is a synthetic Kaggle returns dataset; the merchant network, Razorpay-format
IDs, and evidence photos are all simulated and labelled as such.
