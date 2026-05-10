"""
Classification Round 2 -- With Cluster Label as Feature
CS439 Final Project | Task 8
Identical models to Round 1, with the Cluster column added as an extra feature.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score,
    roc_curve, confusion_matrix, ConfusionMatrixDisplay
)
from imblearn.over_sampling import SMOTE

# ─────────────────────────────────────────────
# 1. Load preprocessed data from xlsx (4 sheets)
# ─────────────────────────────────────────────
print("Loading HR_Processed_Dataset_All_In_One.xlsx ...")
xlsx_path = "HR_Processed_Dataset_All_In_One.xlsx"

X_train = pd.read_excel(xlsx_path, sheet_name="X_train_balanced", engine="openpyxl")
y_train = pd.read_excel(xlsx_path, sheet_name="y_train_balanced", engine="openpyxl")["Attrition"]
X_test  = pd.read_excel(xlsx_path, sheet_name="X_test_scaled",    engine="openpyxl")
y_test  = pd.read_excel(xlsx_path, sheet_name="y_test",           engine="openpyxl")["Attrition"]

print(f"  X_train shape: {X_train.shape}")
print(f"  X_test  shape: {X_test.shape}")

# ─────────────────────────────────────────────
# 2. Extract Cluster column from employee_clusters.csv
#    Split using the same 80/20 stratified split as preprocessing
# ─────────────────────────────────────────────
print("\nLoading employee_clusters.csv, extracting Cluster column ...")
clusters_df = pd.read_csv("employee_clusters.csv")

# Map Attrition to binary for stratified split
attrition_binary = clusters_df["Attrition"].map({"Yes": 1, "No": 0})
cluster_col = clusters_df["Cluster"]

cluster_train_raw, cluster_test, _, _ = train_test_split(
    cluster_col,
    attrition_binary,
    test_size=0.2,
    random_state=42,
    stratify=attrition_binary
)

cluster_train_raw = cluster_train_raw.reset_index(drop=True)
cluster_test      = cluster_test.reset_index(drop=True)

print(f"  cluster_train_raw length: {len(cluster_train_raw)}")
print(f"  cluster_test      length: {len(cluster_test)}")

# ─────────────────────────────────────────────
# 3. Apply SMOTE to Cluster training column to match X_train_balanced
# ─────────────────────────────────────────────
print("\nApplying SMOTE to Cluster training column ...")

_, _, y_train_orig, _ = train_test_split(
    attrition_binary,
    attrition_binary,
    test_size=0.2,
    random_state=42,
    stratify=attrition_binary
)
y_train_orig = y_train_orig.reset_index(drop=True)

cluster_train_2d = cluster_train_raw.values.reshape(-1, 1)
smote = SMOTE(random_state=42)
cluster_train_balanced, _ = smote.fit_resample(cluster_train_2d, y_train_orig)

cluster_train_balanced = pd.Series(
    cluster_train_balanced.flatten(),
    name="Cluster"
).astype(int)

print(f"  Cluster train length after SMOTE: {len(cluster_train_balanced)}")
assert len(cluster_train_balanced) == len(X_train), \
    "Mismatch between SMOTE Cluster length and X_train rows!"

# ─────────────────────────────────────────────
# 4. Append Cluster column to train and test feature sets
# ─────────────────────────────────────────────
print("\nAppending Cluster column to feature matrices ...")

X_train_r2 = X_train.copy().reset_index(drop=True)
X_train_r2["Cluster"] = cluster_train_balanced.values

X_test_r2 = X_test.copy().reset_index(drop=True)
X_test_r2["Cluster"] = cluster_test.values

print(f"  X_train_r2 shape: {X_train_r2.shape}")
print(f"  X_test_r2  shape: {X_test_r2.shape}")

# ─────────────────────────────────────────────
# 5. Define three models (same parameters as Round 1)
# ─────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost":             XGBClassifier(
                               use_label_encoder=False,
                               eval_metric="logloss",
                               random_state=42,
                               verbosity=0
                           )
}

# ─────────────────────────────────────────────
# 6. Train, evaluate, and collect metrics
# ─────────────────────────────────────────────
print("\nTraining three models ...")

results  = []
roc_data = {}
cm_data  = {}

for model_name, model in models.items():
    print(f"\n  Training {model_name} ...")
    model.fit(X_train_r2, y_train)

    y_pred = model.predict(X_test_r2)
    y_prob = model.predict_proba(X_test_r2)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall    = recall_score(y_test, y_pred, zero_division=0)
    f1        = f1_score(y_test, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_test, y_prob)

    print(f"    Accuracy : {acc:.4f}")
    print(f"    Precision: {precision:.4f}")
    print(f"    Recall   : {recall:.4f}")
    print(f"    F1       : {f1:.4f}")
    print(f"    ROC-AUC  : {roc_auc:.4f}")

    results.append({
        "Model"    : model_name,
        "Accuracy" : round(acc, 4),
        "Precision": round(precision, 4),
        "Recall"   : round(recall, 4),
        "F1"       : round(f1, 4),
        "ROC_AUC"  : round(roc_auc, 4)
    })

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[model_name] = (fpr, tpr, roc_auc)
    cm_data[model_name]  = confusion_matrix(y_test, y_pred)

# ─────────────────────────────────────────────
# 7. Save metrics to CSV
# ─────────────────────────────────────────────
results_df = pd.DataFrame(results)
results_df.to_csv("classification_results_round2.csv", index=False)
print("\nSaved: classification_results_round2.csv")
print(results_df.to_string(index=False))

# ─────────────────────────────────────────────
# 8. Plot ROC curves (all three models on one figure)
# ─────────────────────────────────────────────
print("\nPlotting ROC curves ...")

fig, ax = plt.subplots(figsize=(8, 6))
colors = ["steelblue", "darkorange", "seagreen"]
for (model_name, (fpr, tpr, auc_val)), color in zip(roc_data.items(), colors):
    ax.plot(fpr, tpr, label=f"{model_name} (AUC = {auc_val:.4f})", color=color, lw=2)

ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random Classifier")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate",  fontsize=12)
ax.set_title("ROC Curves -- Classification Round 2 (With Cluster Feature)", fontsize=13)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curves_round2.png", dpi=200)
plt.close()
print("  Saved: roc_curves_round2.png")

# ─────────────────────────────────────────────
# 9. Plot confusion matrices (1 row, 3 columns)
# ─────────────────────────────────────────────
print("\nPlotting confusion matrices ...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Confusion Matrices -- Classification Round 2 (With Cluster Feature)", fontsize=13)

for ax, (model_name, cm) in zip(axes, cm_data.items()):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Stay", "Leave"]
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(model_name, fontsize=11)

plt.tight_layout()
plt.savefig("confusion_matrices_round2.png", dpi=200)
plt.close()
print("  Saved: confusion_matrices_round2.png")

print("\nClassification Round 2 complete.")
