"""
Task 10 -- Stacking Ensemble (without Cluster feature)
IBM HR Employee Attrition Dataset
Base learners : Random Forest + XGBoost (tuned)
Meta-learner  : Logistic Regression
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score
)
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# 1. Load preprocessed data from xlsx (4 sheets)
# ─────────────────────────────────────────────
print("=" * 60)
print("Loading HR_Processed_Dataset_All_In_One.xlsx ...")
print("=" * 60)

xlsx_path = "HR_Processed_Dataset_All_In_One.xlsx"

X_train = pd.read_excel(xlsx_path, sheet_name="X_train_balanced", engine="openpyxl")
y_train = pd.read_excel(xlsx_path, sheet_name="y_train_balanced", engine="openpyxl")["Attrition"]
X_test  = pd.read_excel(xlsx_path, sheet_name="X_test_scaled",    engine="openpyxl")
y_test  = pd.read_excel(xlsx_path, sheet_name="y_test",           engine="openpyxl")["Attrition"]

print(f"X_train shape : {X_train.shape}")
print(f"y_train shape : {y_train.shape}")
print(f"X_test  shape : {X_test.shape}")
print(f"y_test  shape : {y_test.shape}")

# ─────────────────────────────────────────────
# 2. Base Learner 1: Random Forest (default parameters)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Base Learner 1: Random Forest (default parameters)")
print("=" * 60)

rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
print("Random Forest params:", rf_base.get_params())

# ─────────────────────────────────────────────
# 3. Base Learner 2: XGBoost (tuned with RandomizedSearchCV)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Base Learner 2: XGBoost (RandomizedSearchCV tuning)")
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

print("Running XGBoost RandomizedSearchCV (n_iter=50, cv=5) ...")
xgb_search.fit(X_train, y_train)

best_xgb_params = xgb_search.best_params_
print(f"\nXGBoost best parameters: {best_xgb_params}")
print(f"XGBoost best CV F1: {xgb_search.best_score_:.4f}")

xgb_best = XGBClassifier(
    eval_metric="logloss",
    verbosity=0,
    random_state=42,
    use_label_encoder=False,
    **best_xgb_params
)

# ─────────────────────────────────────────────
# 4. Build Stacking model
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Building Stacking model")
print("=" * 60)

estimators = [
    ("random_forest", rf_base),
    ("xgboost",       xgb_best),
]

stacking_clf = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=1000, random_state=42),
    cv=5,
    stack_method="predict_proba",
    n_jobs=-1
)

print("Stacking structure:")
print("  Base Learners : Random Forest, XGBoost (tuned)")
print("  Meta Learner  : Logistic Regression")
print("  cv=5, stack_method='predict_proba'")

# ─────────────────────────────────────────────
# 5. Train Stacking and evaluate on test set
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Training Stacking model ...")
print("=" * 60)

stacking_clf.fit(X_train, y_train)
print("Training complete.\n")

y_pred_stack  = stacking_clf.predict(X_test)
y_proba_stack = stacking_clf.predict_proba(X_test)[:, 1]

stack_accuracy  = accuracy_score(y_test, y_pred_stack)
stack_precision = precision_score(y_test, y_pred_stack, zero_division=0)
stack_recall    = recall_score(y_test, y_pred_stack, zero_division=0)
stack_f1        = f1_score(y_test, y_pred_stack, zero_division=0)
stack_roc_auc   = roc_auc_score(y_test, y_proba_stack)

print("=" * 60)
print("Stacking Model -- Test Set Results")
print("=" * 60)
print(f"  Accuracy  : {stack_accuracy:.4f}")
print(f"  Precision : {stack_precision:.4f}")
print(f"  Recall    : {stack_recall:.4f}")
print(f"  F1 Score  : {stack_f1:.4f}")
print(f"  ROC-AUC   : {stack_roc_auc:.4f}")

# ─────────────────────────────────────────────
# 6. Build full results summary and save to CSV
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("Saving final_results_summary.csv ...")
print("=" * 60)

summary_data = [
    {"Model": "Naive Baseline",    "Accuracy": 0.8401, "Precision": 0.0000, "Recall": 0.0000, "F1": 0.0000, "ROC_AUC": 0.5000},
    {"Model": "LR Round1",         "Accuracy": 0.7721, "Precision": 0.3936, "Recall": 0.7872, "F1": 0.5248, "ROC_AUC": 0.8062},
    {"Model": "RF Round1",         "Accuracy": 0.8537, "Precision": 0.5909, "Recall": 0.2766, "F1": 0.3768, "ROC_AUC": 0.7979},
    {"Model": "XGB Round1",        "Accuracy": 0.8639, "Precision": 0.6296, "Recall": 0.3617, "F1": 0.4595, "ROC_AUC": 0.7928},
    {"Model": "RF Tuned",          "Accuracy": 0.8469, "Precision": 0.5455, "Recall": 0.2553, "F1": 0.3478, "ROC_AUC": 0.7836},
    {"Model": "XGB Tuned",         "Accuracy": 0.8265, "Precision": 0.4643, "Recall": 0.5532, "F1": 0.5049, "ROC_AUC": 0.8132},
    {
        "Model":     "Stacking",
        "Accuracy":  round(stack_accuracy,  4),
        "Precision": round(stack_precision, 4),
        "Recall":    round(stack_recall,    4),
        "F1":        round(stack_f1,        4),
        "ROC_AUC":   round(stack_roc_auc,   4),
    },
]

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv("final_results_summary.csv", index=False)
print("Saved: final_results_summary.csv")
print(summary_df.to_string(index=False))

# ─────────────────────────────────────────────
# 7. Plot final comparison bar chart (F1 and ROC-AUC)
# ─────────────────────────────────────────────
print("\nPlotting final_comparison.png ...")

model_names = summary_df["Model"].tolist()
f1_vals     = summary_df["F1"].tolist()
auc_vals    = summary_df["ROC_AUC"].tolist()

x     = np.arange(len(model_names))
width = 0.35

fig, ax = plt.subplots(figsize=(13, 6))
bars1 = ax.bar(x - width / 2, f1_vals,  width, label="F1 Score",  color="#4C72B0", alpha=0.85)
bars2 = ax.bar(x + width / 2, auc_vals, width, label="ROC-AUC",   color="#DD8452", alpha=0.85)

for bar in list(bars1) + list(bars2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            h + 0.005, f"{h:.4f}",
            ha="center", va="bottom", fontsize=7.5, rotation=45)

ax.set_xlabel("Model", fontsize=12)
ax.set_ylabel("Score",  fontsize=12)
ax.set_title("Final Model Comparison: F1 Score & ROC-AUC\n(IBM HR Attrition -- No Cluster Feature)",
             fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(model_names, rotation=20, ha="right", fontsize=10)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=11)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("final_comparison.png", dpi=200)
plt.close()
print("Saved: final_comparison.png")

print("\n" + "=" * 60)
print("Task 10 complete. Stacking Ensemble final results:")
print("=" * 60)
print(f"  Accuracy  : {stack_accuracy:.4f}")
print(f"  Precision : {stack_precision:.4f}")
print(f"  Recall    : {stack_recall:.4f}")
print(f"  F1 Score  : {stack_f1:.4f}")
print(f"  ROC-AUC   : {stack_roc_auc:.4f}")
print("\nOutput files:")
print("  final_results_summary.csv")
print("  final_comparison.png")
