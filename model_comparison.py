"""
model_comparison.py
Compares Random Forest, Logistic Regression, and Gradient Boosting - not just
on accuracy/precision/recall, but on CALIBRATION: when a model says "73%
probability of abuse," is that actually right 73% of the time?

This matters because your agent's guardrail threshold (0.43) only means what
you think it means if the probabilities are trustworthy. A model can have
perfect precision/recall on this test set while still producing poorly
calibrated probabilities - e.g. always spitting out 0.99 or 0.01 with nothing
in between, which is confident but not necessarily well-calibrated.

Metrics used:
- Precision/Recall: discrimination (does it separate abuse from legit?)
- Brier score: calibration (lower = better; measures how close predicted
  probabilities are to actual outcomes, penalizing overconfidence)
- Log loss: also penalizes confident-but-wrong predictions heavily
"""
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score, recall_score, brier_score_loss, log_loss, roc_auc_score
)
from sklearn.calibration import calibration_curve

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

NUMERIC_FEATURES = [
    "age", "account_age_days", "avg_order_value_usd",
    "refund_amount_requested_usd", "is_high_value_item", "discount_used",
    "days_to_return", "total_orders_lifetime", "total_returns_lifetime",
    "return_rate_pct", "item_returned_opened", "return_packaging_intact",
    "photo_evidence_provided", "tracking_number_valid",
    "address_change_before_delivery", "refund_to_different_account",
    "multiple_accounts_flag", "customer_support_contacts",
    "previous_dispute_count", "wishlist_to_cart_time_hrs",
    "review_left_after_return",
]
CATEGORICAL_FEATURES = [
    "customer_segment", "country", "platform", "device_type",
    "payment_method", "product_category", "return_reason", "shipping_carrier",
]
TARGET = "is_abuse"

X_train = train_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_train = train_df[TARGET]
X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y_test = test_df[TARGET]

# Logistic Regression needs scaled numeric features; tree models don't care,
# but scaling doesn't hurt them either, so we use one preprocessor for all three.
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.1, random_state=42
    ),
}

results = []
fitted_models = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("classifier", clf)])
    pipe.fit(X_train, y_train)
    fitted_models[name] = pipe

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.43).astype(int)  # use our tuned threshold for everyone, fair comparison

    results.append({
        "model": name,
        "precision": round(precision_score(y_test, pred), 4),
        "recall": round(recall_score(y_test, pred), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
        "brier_score": round(brier_score_loss(y_test, proba), 5),  # LOWER = better calibrated
        "log_loss": round(log_loss(y_test, proba), 5),             # LOWER = better calibrated
    })

results_df = pd.DataFrame(results)
print("=" * 80)
print("MODEL COMPARISON: discrimination AND calibration")
print("=" * 80)
print(results_df.to_string(index=False))

best_calibrated = results_df.loc[results_df["brier_score"].idxmin(), "model"]
print(f"\nBest-calibrated model (lowest Brier score): {best_calibrated}")

# Save calibration curve data for each model (predicted vs actual probability, binned)
print("\nCalibration curves (predicted probability bin vs actual fraction abusive):")
calibration_data = {}
for name, pipe in fitted_models.items():
    proba = pipe.predict_proba(X_test)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    calibration_data[name] = {"predicted": mean_pred.tolist(), "actual": frac_pos.tolist()}
    print(f"\n{name}:")
    for p, a in zip(mean_pred, frac_pos):
        print(f"  predicted ~{p:.2f}  ->  actual {a:.2f}")

results_df.to_csv("model_comparison_results.csv", index=False)
print("\nSaved to model_comparison_results.csv")

# Keep the best-calibrated model's probabilities available if you want to swap
# risk_model.joblib for a better-calibrated one later.
import json
with open("calibration_curves.json", "w") as f:
    json.dump(calibration_data, f, indent=2)
print("Calibration curve data saved to calibration_curves.json")
