"""
verify_determinism.py
Proves the agent is reproducible: runs the SAME order N times and checks every
run produced an identical action.

Why this exists: before temperature was pinned to 0, the agent gave opposite
verdicts on identical input - ORD2013623 was approved in several runs and
flagged in another, with no change to the underlying evidence. For a fraud
system that is a real defect: an audit log you cannot reproduce is an audit
log you cannot defend. This script is the regression test for that fix.

Usage:
  python verify_determinism.py [order_id] [runs]
"""
import sys
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import risk_agent_local as agent

order_id = sys.argv[1] if len(sys.argv) > 1 else "ORD2013623"
runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3

df = pd.read_csv("test.csv")
row = df[df["order_id"] == order_id]
if row.empty:
    raise SystemExit(f"Order {order_id} not found in test.csv")

agent.load_orders_from_df(row)

print(f"Running {order_id} {runs} times with temperature=0...\n")
actions, reasonings = [], []
for i in range(runs):
    decision = agent.run_agent_on_order(order_id)
    action = decision["action"] if decision else "NO_DECISION"
    actions.append(action)
    reasonings.append(decision["reasoning"] if decision else "")
    print(f"  run {i + 1}: {action}")

print()
if len(set(actions)) == 1:
    print(f"DETERMINISTIC: all {runs} runs returned '{actions[0]}'")
    if len(set(reasonings)) == 1:
        print("Reasoning text was also byte-identical across every run.")
    else:
        print(f"NOTE: action was stable, but reasoning wording varied "
              f"({len(set(reasonings))} distinct texts).")
else:
    print(f"NON-DETERMINISTIC: got {sorted(set(actions))} across {runs} runs")
