"""
hyperparameter_tuning.py
Task 9 -- Hyperparameter Tuning (without Cluster feature)
RandomizedSearchCV applied to Random Forest and XGBoost.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)

# ─────────────────────────────────────────────
# 1. Load preprocessed data from xlsx (4 sheets)
# ─────────────────────────────────────────────
print("=" * 60)
print("Loading data ...")
print("=" * 60)

xlsx_path = "HR_Processed_Dataset_All_In_One.xlsx"

X_train = pd.read_excel(xlsx_path, sheet_name="X_train_balanced", engine="openpyxl")
y_train = pd.read_excel(xlsx_path, sheet_name="y_train_balanced", engine="openpyxl").squeeze()
X_test  = pd.read_excel(xlsx_path, sheet_name="X_test_scaled",    engine="openpyxl")
y_test  = pd.read_excel(xlsx_path, sheet_name="y_test",           engine="openpyxl").squeeze()

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"X_test  shape : {X_test.shape}")
print(f"y_test  shape : {y_test.shape}")

# ─────────────────────────────────────────────
# 2. Round 1 baseline results (hardcoded for comparison)
# ─────────────────────────────────────────────
baseline_results = {
    "Logistic Regression": {
        "Accuracy": 0.7721, "Precision": 0.3936,
        "Recall":   0.7872, "F1":        0.5248, "ROC_AUC": 0.8062
    },
    "Random Forest (before)": {
        "Accuracy": 0.8537, "Precision": 0.5909,
        "Recall":   0.2766, "F1":        0.3768, "ROC_AUC": 0.7979
    },
    "XGBoost (before)": {
        "Accuracy": 0.8639, "Precision": 0.6296,
        "Recall":   0.3617, "F1":        0.4595, "ROC_AUC": 0.7928
    },
}

# ─────────────────────────────────────────────
# 3. Helper: evaluate model on test set
# ─────────────────────────────────────────────
def evaluate(model, X, y):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy":  round(accuracy_score(y, y_pred), 4),
        "Precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y, y_pred, zero_division=0), 4),
        "ROC_AUC":   round(roc_auc_score(y, y_prob), 4),
    }

# ─────────────────────────────────────────────
# 4. Random Forest -- RandomizedSearchCV
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Tuning Random Forest (n_iter=50, cv=5, scoring='f1') ...")
print("=" * 60)

rf_param_dist = {
    "n_estimators":      [100, 200, 300, 500],
    "max_depth":         [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf":  [1, 2, 4],
    "max_features":      ["sqrt", "log2"],
}

rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_distributions=rf_param_dist,
    n_iter=50,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
    verbose=1
)
rf_search.fit(X_train, y_train)

print(f"\nRandom Forest best parameters:")
for k, v in rf_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best CV F1 (train): {rf_search.best_score_:.4f}")

rf_best    = rf_search.best_estimator_
rf_metrics = evaluate(rf_best, X_test, y_test)

print(f"\nRandom Forest test set metrics after tuning:")
for k, v in rf_metrics.items():
    print(f"  {k}: {v}")

# ─────────────────────────────────────────────
# 5. XGBoost -- RandomizedSearchCV
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Tuning XGBoost (n_iter=50, cv=5, scoring='f1') ...")
print("=" * 60)

xgb_param_dist = {
    "n_estimators":     [100, 200, 300, 500],
    "max_depth":        [3, 5, 7, 9],
    "learning_rate":    [0.01, 0.05, 0.1, 0.2],
    "subsample":        [0.6, 0.8, 1.0],
    "colsample_bytree": [0.6, 0.8, 1.0],
    "scale_pos_weight": [1, 2, 3],
}

xgb_search = RandomizedSearchCV(
    estimator=XGBClassifier(
        eval_metric="logloss",
        verbosity=0,
        random_state=42,
        use_label_encoder=False
    ),
    param_distributions=xgb_param_dist,
    n_iter=50,
    cv=5,
    scoring="f1",
    random_state=42,
    n_jobs=-1,
    verbose=1
)
xgb_search.fit(X_train, y_train)

print(f"\nXGBoost best parameters:")
for k, v in xgb_search.best_params_.items():
    print(f"  {k}: {v}")
print(f"Best CV F1 (train): {xgb_search.best_score_:.4f}")

xgb_best    = xgb_search.best_estimator_
xgb_metrics = evaluate(xgb_best, X_test, y_test)

print(f"\nXGBoost test set metrics after tuning:")
for k, v in xgb_metrics.items():
    print(f"  {k}: {v}")

# ─────────────────────────────────────────────
# 6. Print before vs after comparison
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Before vs After Comparison")
print("=" * 60)

comparison_data = [
    ("Logistic Regression (no tuning)", baseline_results["Logistic Regression"]),
    ("Random Forest (before tuning)",   baseline_results["Random Forest (before)"]),
    ("Random Forest (after tuning)",    rf_metrics),
    ("XGBoost (before tuning)",         baseline_results["XGBoost (before)"]),
    ("XGBoost (after tuning)",          xgb_metrics),
]

metrics_cols = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
header = f"{'Model':<35}" + "".join(f"{m:>12}" for m in metrics_cols)
print(header)
print("-" * (35 + 12 * len(metrics_cols)))
for name, m in comparison_data:
    row = f"{name:<35}" + "".join(f"{m[k]:>12.4f}" for k in metrics_cols)
    print(row)

# ─────────────────────────────────────────────
# 7. Save tuned results to CSV
# ─────────────────────────────────────────────
tuning_df = pd.DataFrame([
    {"Model": "Random Forest (tuned)", **rf_metrics},
    {"Model": "XGBoost (tuned)",       **xgb_metrics},
])
tuning_df.to_csv("tuning_results.csv", index=False)
print("\nSaved: tuning_results.csv")

# ─────────────────────────────────────────────
# 8. Plot F1 and ROC-AUC before vs after bar chart
# ─────────────────────────────────────────────
models_label = ["RF Before", "RF After", "XGB Before", "XGB After"]

f1_values = [
    baseline_results["Random Forest (before)"]["F1"],
    rf_metrics["F1"],
    baseline_results["XGBoost (before)"]["F1"],
    xgb_metrics["F1"],
]
auc_values = [
    baseline_results["Random Forest (before)"]["ROC_AUC"],
    rf_metrics["ROC_AUC"],
    baseline_results["XGBoost (before)"]["ROC_AUC"],
    xgb_metrics["ROC_AUC"],
]

x = np.arange(len(models_label))
colors_bar = ["#5c85d6", "#2255b8", "#f0a050", "#c05010"]

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("Hyperparameter Tuning: Before vs After Comparison",
             fontsize=14, fontweight="bold")

for ax, values, title, ylim in zip(
    axes,
    [f1_values, auc_values],
    ["F1 Score Comparison", "ROC-AUC Score Comparison"],
    [(0, max(f1_values) * 1.25), (0.70, min(max(auc_values) * 1.08, 1.0))]
):
    bars = ax.bar(x, values, color=colors_bar, edgecolor="white", linewidth=0.8)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models_label, rotation=15, ha="right")
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005, f"{val:.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#5c85d6", label="RF Before"),
    Patch(facecolor="#2255b8", label="RF After"),
    Patch(facecolor="#f0a050", label="XGB Before"),
    Patch(facecolor="#c05010", label="XGB After"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.02), fontsize=10)

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig("tuning_comparison.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved: tuning_comparison.png")

print("\nTask 9 complete.")
