"""
Step 2 — Predict customer behaviour (will they buy again?) and value (how much?),
then translate the predictions into marketing actions and measurable impact.

Two models on the same RFM-style features:
  1. CHURN / repeat-purchase classifier  -> probability a customer buys next quarter
  2. Value regressor                      -> expected next-quarter spend

Business outputs:
  * Gains chart: revenue captured by targeting the top X% of predicted-value customers.
  * At-risk list: historically high-value customers the model expects to lapse.

Run:  python 02_model_ltv.py   ->  figures + metrics.json in ./outputs
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_absolute_error, roc_auc_score, average_precision_score,
                             roc_curve)

sns.set_theme(style="whitegrid")
OUT = Path("outputs"); OUT.mkdir(exist_ok=True)
RS = 42
M = {}

df = pd.read_csv("data/customer_features.csv")
FEATURES = ["recency_days", "tenure_days", "frequency", "monetary_total",
            "avg_order_value", "total_items", "distinct_products",
            "active_months", "recent_revenue_90d"]
X = df[FEATURES]
y_value = df["future_revenue"]
y_active = (df["future_revenue"] > 0).astype(int)      # behaviour target
M["n_customers"] = int(len(df))
M["active_next_quarter_pct"] = round(float(y_active.mean()), 4)

idx = np.arange(len(df))
itr, ite = train_test_split(idx, test_size=0.25, random_state=RS, stratify=y_active)

# --- skewed, zero-inflated target -----------------------------------------
plt.figure(figsize=(6, 4))
sns.histplot(np.log1p(y_value), bins=40, color="#1f78b4")
plt.xlabel("log(1 + next-quarter revenue)"); plt.ylabel("customers")
plt.title("Next-quarter revenue is skewed and zero-inflated")
plt.tight_layout(); plt.savefig(OUT / "01_target_distribution.png", dpi=120); plt.close()

# ===========================================================================
# MODEL 1 — will the customer buy again next quarter? (behaviour)
# ===========================================================================
clf = HistGradientBoostingClassifier(max_depth=6, learning_rate=0.06,
                                     max_iter=400, random_state=RS)
clf.fit(X.iloc[itr], y_active.iloc[itr])
p_active = clf.predict_proba(X.iloc[ite])[:, 1]
M["churn_roc_auc"] = round(float(roc_auc_score(y_active.iloc[ite], p_active)), 3)
M["churn_pr_auc"] = round(float(average_precision_score(y_active.iloc[ite], p_active)), 3)

fpr, tpr, _ = roc_curve(y_active.iloc[ite], p_active)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color="#e31a1c", lw=2, label=f"Model (AUC={M['churn_roc_auc']:.2f})")
plt.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
plt.title("Predicting who returns next quarter"); plt.legend()
plt.tight_layout(); plt.savefig(OUT / "04_churn_roc.png", dpi=120); plt.close()

# ===========================================================================
# MODEL 2 — how much will they spend? (value)  [log target]
# ===========================================================================
reg = HistGradientBoostingRegressor(max_depth=6, learning_rate=0.06, max_iter=500,
                                    l2_regularization=1.0, random_state=RS)
reg.fit(X.iloc[itr], np.log1p(y_value.iloc[itr]))
pred_spend = np.clip(np.expm1(reg.predict(X.iloc[ite])), 0, None)
M["value_mae"] = round(float(mean_absolute_error(y_value.iloc[ite], pred_spend)), 2)
M["value_spearman"] = round(float(spearmanr(y_value.iloc[ite], pred_spend).statistic), 3)

# Expected value = P(buy) x predicted spend  -> the ranking score for targeting
exp_value = p_active * pred_spend

# ===========================================================================
# BUSINESS IMPACT 1 — gains chart (revenue captured by targeting top X%)
# ===========================================================================
te = pd.DataFrame({"actual": y_value.iloc[ite].values,
                   "score": exp_value,
                   "past_spend": X.iloc[ite]["monetary_total"].values})
total = te["actual"].sum()

def gains(col):
    d = te.sort_values(col, ascending=False)
    return (np.arange(1, len(d) + 1) / len(d)), (d["actual"].cumsum().values / total)

cf, g_model = gains("score")
_, g_past = gains("past_spend")
cap = lambda frac, g: float(g[np.searchsorted(cf, frac) - 1])
M["model_capture_top20"] = round(cap(0.20, g_model), 3)
M["pastspend_capture_top20"] = round(cap(0.20, g_past), 3)
M["lift_top20_vs_random"] = round(M["model_capture_top20"] / 0.20, 2)

plt.figure(figsize=(7, 6))
plt.plot(cf, g_model, color="#e31a1c", lw=2.5, label="Target by predicted value (model)")
plt.plot(cf, g_past, color="#1f78b4", lw=2, ls="--", label="Target by past spend")
plt.plot([0, 1], [0, 1], color="grey", lw=1, ls=":", label="Random targeting")
plt.scatter([0.20], [M["model_capture_top20"]], color="#e31a1c", zorder=5)
plt.annotate(f"  Top 20% -> {M['model_capture_top20']*100:.0f}% of next-quarter revenue",
             (0.20, M["model_capture_top20"]), fontsize=11, va="center")
plt.xlabel("Fraction of customers targeted")
plt.ylabel("Fraction of next-quarter revenue captured")
plt.title("Gains chart: model-guided targeting vs. past spend vs. random")
plt.legend(loc="lower right"); plt.tight_layout()
plt.savefig(OUT / "02_gains_chart.png", dpi=120); plt.close()

# Decile lift
te["decile"] = pd.qcut(te["score"].rank(method="first"), 10,
                       labels=[f"D{i}" for i in range(1, 11)])
dec = te.groupby("decile", observed=True)["actual"].mean().reindex([f"D{i}" for i in range(1, 11)])
plt.figure(figsize=(7, 4))
dec.plot(kind="bar", color="#1f78b4")
plt.ylabel("Mean actual next-quarter revenue (GBP)")
plt.xlabel("Predicted-value decile (D10 = highest)")
plt.title("Higher predicted value -> higher actual revenue")
plt.tight_layout(); plt.savefig(OUT / "03_decile_lift.png", dpi=120); plt.close()
M["top_decile_vs_bottom_decile_x"] = round(float(dec["D10"] / max(dec["D1"], 1e-9)), 1)

# ===========================================================================
# BUSINESS IMPACT 2 — high-value customers at risk of lapsing
# (something "rank by past spend" alone cannot surface)
# ===========================================================================
hv_cut = float(df["monetary_total"].quantile(0.75))
test = df.iloc[ite].copy(); test["p_active"] = p_active
hv = test[test["monetary_total"] >= hv_cut]
at_risk = hv[hv["p_active"] < 0.5]
M["high_value_threshold_gbp"] = round(hv_cut, 2)
M["n_high_value_test"] = int(len(hv))
M["n_at_risk_test"] = int(len(at_risk))
M["revenue_at_risk_test_gbp"] = round(float(at_risk["monetary_total"].sum()), 2)
M["at_risk_share_of_high_value"] = round(float(len(at_risk) / max(len(hv), 1)), 3)

# Feature importance for the behaviour model (what predicts returning)
imp = permutation_importance(clf, X.iloc[ite], y_active.iloc[ite],
                             n_repeats=5, random_state=RS, n_jobs=-1)
order = np.argsort(imp.importances_mean)
plt.figure(figsize=(7, 4.2))
plt.barh(np.array(FEATURES)[order], imp.importances_mean[order], color="#1f78b4")
plt.xlabel("Permutation importance"); plt.title("What predicts a repeat purchase")
plt.tight_layout(); plt.savefig(OUT / "05_feature_importance.png", dpi=120); plt.close()

with open(OUT / "metrics.json", "w") as f:
    json.dump(M, f, indent=2)
print(json.dumps(M, indent=2))
print("\nDone. Figures + metrics.json in ./outputs")
