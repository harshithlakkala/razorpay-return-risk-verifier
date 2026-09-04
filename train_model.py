"""
train_model.py
Trains a classifier to predict whether a return is likely abusive
(policy abuse / fraudulent return / wardrobing) vs legitimate.

This model becomes the "get_risk_score" tool the agent calls later.
"""
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_score, recall_score

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Columns we actually feed the model.
# We deliberately EXCLUDE order_id, customer_id, order_date, return_date
# (identifiers, not predictive signals) and abuse_type/abuse_label
# (that would be leaking the answer straight into the model).
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

# Turn text categories (like "Crypto", "Wardrobing") into numbers the model can use
preprocessor = ColumnTransformer(transformers=[
    ("num", "passthrough", NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])

model = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200, max_depth=12, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1
    )),
])

model.fit(X_train, y_train)

# --- Evaluate on the HELD-OUT test set (model has never seen this data) ---
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]  # probability of "abuse"

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("=" * 50)
print("HELD-OUT TEST SET RESULTS (12,000 orders, unseen during training)")
print("=" * 50)
print(f"Precision: {precision:.3f}  (of orders we flagged, how many were really abuse)")
print(f"Recall:    {recall:.3f}  (of real abuse cases, how many we caught)")
print()
print("Confusion matrix:")
print(f"                  Predicted Legit   Predicted Abuse")
print(f"Actual Legit       {cm[0][0]:>6}            {cm[0][1]:>6}   <- false positives")
print(f"Actual Abuse       {cm[1][0]:>6}            {cm[1][1]:>6}   <- false negatives (missed)")
print()
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Abuse"]))

# --- False-positive cost estimate ---
# A false positive = a genuine customer gets flagged/delayed.
# We estimate the cost as the average order value of those wrongly-flagged orders.
fp_mask = (y_test.values == 0) & (y_pred == 1)
fp_cost = test_df.loc[fp_mask, "avg_order_value_usd"].sum()
fn_mask = (y_test.values == 1) & (y_pred == 0)
fn_cost = test_df.loc[fn_mask, "refund_amount_requested_usd"].sum()

print(f"False positives: {fp_mask.sum()} orders, ${fp_cost:,.2f} in friction/lost trust exposure")
print(f"False negatives: {fn_mask.sum()} orders, ${fn_cost:,.2f} in abuse that slipped through")

joblib.dump(model, "risk_model.joblib")
print("\nModel saved to risk_model.joblib")
