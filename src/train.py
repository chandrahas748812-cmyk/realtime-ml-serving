"""Train a demo XGBoost fraud model on synthetic data. Run once before serving."""

from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from .features import FEATURE_NAMES, MERCHANT_RISK

rng = np.random.default_rng(42)
N = 20_000

# ---- synthetic transaction data ----
amount = rng.lognormal(mean=4.0, sigma=1.2, size=N)
hour = rng.integers(0, 24, size=N)
merchants = rng.choice(list(MERCHANT_RISK.keys()), size=N, p=[0.2, 0.15, 0.15, 0.2, 0.1, 0.08, 0.05, 0.02, 0.05])
avg30 = amount * rng.uniform(0.5, 1.5, size=N)
days_last = rng.exponential(scale=3.0, size=N)

# fraud probability rises with: high amount vs avg, night hours, risky merchant
logit = (
    -4.0
    + 1.5 * np.log1p(amount / np.maximum(avg30, 1))
    + 0.8 * ((hour < 6) | (hour >= 23)).astype(float)
    + 2.0 * np.array([MERCHANT_RISK[m] for m in merchants])
    + rng.normal(0, 0.5, size=N)
)
fraud = (rng.random(N) < 1 / (1 + np.exp(-logit))).astype(int)
print(f"Fraud rate: {fraud.mean():.3%}")

X = pd.DataFrame({
    "amount": amount,
    "amount_log": np.log1p(amount),
    "hour_of_day": hour,
    "is_night": ((hour < 6) | (hour >= 23)).astype(float),
    "days_since_last": days_last,
    "avg_amount_30d": avg30,
    "amount_vs_avg": amount / np.maximum(avg30, 1),
    "merchant_risk": [MERCHANT_RISK[m] for m in merchants],
})[FEATURE_NAMES]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model = XGBClassifier(
    n_estimators=200, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
)
model.fit(X_train, y_train)

auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
print(f"Test ROC-AUC: {auc:.4f}")

version = "fraud-xgb-" + datetime.utcnow().strftime("%Y%m%d")
bundle = {
    "model": model,
    "version": version,
    "threshold": 0.5,
    "feature_names": FEATURE_NAMES,
    "baseline_means": X_train.mean().tolist(),
    "baseline_stds": X_train.std().tolist(),
}
out = Path("artifacts")
out.mkdir(exist_ok=True)
joblib.dump(bundle, out / "fraud_model.joblib")
print(f"Saved artifacts/fraud_model.joblib (version {version})")
