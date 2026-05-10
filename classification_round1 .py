"""
Classification Round 1 -- Without Cluster Label
CS439 Final Project | IBM HR Employee Attrition Prediction
Models: Logistic Regression, Random Forest, XGBoost
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, roc_curve,
    confusion_matrix, ConfusionMatrixDisplay
)
from xgboost import XGBClassifier

# ──────────────────────────────────────────────
# 1. Load preprocessed data from xlsx (4 sheets)
# ──────────────────────────────────────────────
print("=" * 60)
print("Classification Round 1 -- Without Cluster Label")
print("=" * 60)

XLSX_PATH = "HR_Processed_Dataset_All_In_One.xlsx"
print(f"\n[1] Loading data from: {XLSX_PATH}")

X_train = pd.read_excel(XLSX_PATH, sheet_name="X_train_balanced", engine="openpyxl")
y_train = pd.read_excel(XLSX_PATH, sheet_name="y_train_balanced", engine="openpyxl").squeeze()
X_test  = pd.read_excel(XLSX_PATH, sheet_name="X_test_scaled",    engine="openpyxl")
y_test  = pd.read_excel(XLSX_PATH, sheet_name="y_test",           engine="openpyxl").squeeze()

print(f"   X_train shape : {X_train.shape}")
print(f"   y_train shape : {y_train.shape}  (balanced with SMOTE)")
print(f"   X_test  shape : {X_test.shape}")
print(f"   y_test  shape : {y_test.shape}")
print(f"   Train label distribution:\n{y_train.value_counts().to_string()}")
print(f"   Test label distribution:\n{y_test.value_counts().to_string()}")

# ──────────────────────────────────────────────
# 2. Define three classification models
# ──────────────────────────────────────────────
print("\n[2] Initializing models")

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    )
}

# ──────────────────────────────────────────────
# 3. Train and evaluate each model
# ──────────────────────────────────────────────
print("\n[3] Training and evaluation")

results  = []
roc_data = {}
cm_data  = {}

for name, model in models.items():
    print(f"\n   -- {name}")

    model.fit(X_train, y_train)

    y_pred      = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score (y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score   (y_test, y_pred, zero_division=0)
    f1        = f1_score       (y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score  (y_test, y_pred_prob)

    print(f"      Accuracy  : {acc:.4f}")
    print(f"      Precision : {precision:.4f}")
    print(f"      Recall    : {recall:.4f}")
    print(f"      F1-score  : {f1:.4f}")
    print(f"      ROC-AUC   : {roc_auc:.4f}")

    results.append({
        "Model"    : name,
        "Accuracy" : round(acc,       4),
        "Precision": round(precision, 4),
        "Recall"   : round(recall,    4),
        "F1"       : round(f1,        4),
        "ROC_AUC"  : round(roc_auc,   4)
    })

    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_data[name] = (fpr, tpr, roc_auc)
    cm_data[name]  = confusion_matrix(y_test, y_pred)

# ──────────────────────────────────────────────
# 4. Plot ROC curves (all three models on one figure)
# ──────────────────────────────────────────────
print("\n[4] Plotting ROC curves -> roc_curves_round1.png")

COLORS = ["#2563EB", "#16A34A", "#DC2626"]
fig, ax = plt.subplots(figsize=(8, 6))

for (name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), COLORS):
    ax.plot(fpr, tpr, color=color, lw=2,
            label=f"{name}  (AUC = {auc_val:.4f})")

ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Baseline (AUC = 0.50)")
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.02])
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves -- Classification Round 1 (No Cluster Label)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves_round1.png", dpi=200)
plt.close()
print("   Saved: roc_curves_round1.png")

# ──────────────────────────────────────────────
# 5. Plot confusion matrices for all three models
# ──────────────────────────────────────────────
print("\n[5] Plotting confusion matrices -> confusion_matrices_round1.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Confusion Matrices -- Classification Round 1 (No Cluster Label)",
             fontsize=13, y=1.02)

display_labels = ["Stay (0)", "Leave (1)"]
for ax, (name, cm) in zip(axes, cm_data.items()):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=display_labels)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=9)
    ax.set_ylabel("True Label",      fontsize=9)

plt.tight_layout()
plt.savefig("confusion_matrices_round1.png", dpi=200, bbox_inches="tight")
plt.close()
print("   Saved: confusion_matrices_round1.png")

# ──────────────────────────────────────────────
# 6. Save metrics summary to CSV
# ──────────────────────────────────────────────
print("\n[6] Saving metrics -> classification_results_round1.csv")

results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"
])
results_df.to_csv("classification_results_round1.csv", index=False)

print("\n   Classification Results Round 1")
print("   " + "-" * 70)
print(results_df.to_string(index=False))
print("   " + "-" * 70)

print("\nRound 1 complete. Output files:")
print("   classification_results_round1.csv")
print("   roc_curves_round1.png")
print("   confusion_matrices_round1.png")
