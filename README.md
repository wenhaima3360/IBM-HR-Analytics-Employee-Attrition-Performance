# IBM-HR-Analytics-Employee-Attrition-Performance

# IBM HR Attrition Analysis Project

## CS439 Final Project

This project analyzes the IBM HR Employee Attrition dataset using data preprocessing, exploratory data analysis (EDA), clustering, and baseline machine learning evaluation.

The goal is to understand employee behavior patterns, identify attrition-related trends, and establish strong preprocessing and evaluation pipelines for future predictive modeling.

---
# Author
WenHai Ma (wm332)
Derek Sun (dds168)
Jolin Chen (jc3875)

# Dataset

Dataset:

IBM HR Analytics Employee Attrition Dataset (Kaggle)[https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

The dataset contains employee demographic information, compensation details, workplace satisfaction metrics, and attrition labels.

## Dataset Statistics

* Total Employees: 1470
* Features: 35
* Target Variable: Attrition
* Attrition Classes:

  * Yes (Employee Left)
  * No (Employee Stayed)

---

# Project Structure

```text
project/
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── preprocessing/
│   ├── hr_preprocessing_pipeline.py
│   └── HR_Processed_Dataset_All_In_One.xlsx
│
├── eda/
│   ├── eda_analysis.py
│   ├── EDA_Report.md
│   ├── correlation_matrix.csv
│   ├── class_imbalance_report.csv
│   └── figures/
│       ├── correlation_heatmap.png
│       └── class_imbalance_plot.png
│
├── clustering/
│   ├── kmeans_clustering_pipeline.py
│   ├── KMeans_Clustering_Report.md
│   ├── cluster_attrition_summary.csv
│   └── figures/
│       ├── elbow_plot.png
│       ├── silhouette_plot.png
│       └── pca_cluster_plot.png
│
├── baseline/
│   ├── naive_baseline.py
│   ├── Naive_Baseline_Report.md
│   └── naive_baseline_metrics.csv
│
└── README.md
```

---

# 1. Exploratory Data Analysis (EDA)

Before building machine learning models, exploratory analysis was performed to understand the structure and quality of the dataset.

## EDA Tasks

### Missing Value Analysis

Checked whether the dataset contains:

* Null values
* NaN values
* Incomplete rows

### Result

```python
Total Missing Values: 0
```

The dataset is clean and does not require imputation.

---

## Correlation Analysis

Correlation analysis was performed on numerical features.

Highly correlated variables were flagged using:

```python
|correlation| >= 0.70
```

### Saved Figures

* correlation_heatmap.png

### Saved Reports

* correlation_matrix.csv
* high_correlation_pairs.csv

---

## Class Imbalance Analysis

Attrition prediction is an imbalanced classification problem.

Most employees remain at the company, while relatively few employees leave.

### Why This Matters

A model can achieve high accuracy simply by predicting:

```python
"Employee stays"
```

Therefore, metrics such as Recall and F1 Score are more important than Accuracy alone.

### Saved Figure

* class_imbalance_plot.png

---

# 2. Data Preprocessing

A full preprocessing pipeline was implemented.

## Removed Columns

The following non-informative columns were removed:

```python
EmployeeCount
EmployeeNumber
Over18
StandardHours
```

---

## Categorical Encoding

Categorical features were encoded using:

```python
LabelEncoder
```

Examples:

```python
BusinessTravel
Department
EducationField
Gender
JobRole
MaritalStatus
OverTime
```

---

## Feature Scaling

Numerical features were standardized using:

```python
StandardScaler
```

Why scaling is necessary:

* K-Means uses Euclidean distance
* Different feature magnitudes can distort clustering
* Scaling improves convergence and model stability

---

## Stratified Train/Test Split

The dataset was divided using:

```python
train_test_split(stratify=y)
```

This preserves the original class distribution in both training and testing sets.

---

## Imbalance Handling

SMOTE was applied only to the training dataset.

```python
SMOTE(random_state=42)
```

### Important

SMOTE was NOT applied to the testing set.

Applying SMOTE to testing data would leak synthetic information into evaluation.

---

# 3. K-Means Clustering

K-Means clustering was used to group employees into hidden behavioral segments.

This is an unsupervised learning task.

The model does NOT use the Attrition label while clustering.

Attrition is analyzed afterward for interpretation.

---

## Why K-Means?

K-Means was selected because:

1. The goal is employee segmentation
2. The dataset becomes fully numeric after encoding
3. K-Means is interpretable and computationally efficient
4. Cluster centers summarize employee behavior patterns
5. It provides a strong baseline clustering algorithm

---

## Selecting the Best k

Two techniques were used:

### Elbow Method

Measures:

```python
Within-cluster sum of squares (Inertia)
```

Saved Figure:

* elbow_plot.png

---

### Silhouette Score

Measures cluster quality:

```python
s = (b - a) / max(a, b)
```

Where:

* a = average intra-cluster distance
* b = average nearest-cluster distance

Interpretation:

* Close to 1 → strong clustering
* Close to 0 → overlapping clusters
* Negative → incorrect assignment

Saved Figure:

* silhouette_plot.png

---

## PCA Visualization

Principal Component Analysis (PCA) was used to reduce dimensions for visualization.

The K-Means model was trained on the FULL scaled feature set.

PCA was used ONLY for 2D plotting.

Saved Figure:

* pca_cluster_plot.png

---

## K-Means Results

### Best k

```python
k = 2
```

### Best Silhouette Score

```python
0.1202
```

### Cluster Attrition Rates

| Cluster   | Attrition Rate |
| --------- | -------------- |
| Cluster 0 | 19.20%         |
| Cluster 1 | 9.46%          |

This suggests one employee group has substantially higher turnover risk.

---

# 4. Naive Baseline Classifier

A naive baseline classifier was created.

The classifier always predicts:

```python
Employee stays
```

or:

```python
Attrition = No
```

This establishes the minimum acceptable performance level.

All future supervised learning models must outperform this baseline.

---

## Baseline Results

| Metric    | Value  |
| --------- | ------ |
| Accuracy  | 0.8401 |
| Precision | 0.0000 |
| Recall    | 0.0000 |
| F1 Score  | 0.0000 |

---

## Interpretation

The baseline achieves high accuracy because the dataset is imbalanced.

However:

* Precision = 0
* Recall = 0
* F1 Score = 0

The classifier completely fails to detect employees who leave.

This demonstrates why:

```python
Accuracy alone is misleading
```

for imbalanced classification problems.

---

# 5. Evaluation Metrics

The following metrics are recommended for future models:

## Accuracy

Measures overall correctness.

---

## Precision

Measures:

```python
Of employees predicted to leave,
how many actually leave?
```

---

## Recall

Measures:

```python
Of employees who actually leave,
how many were detected?
```

---

## F1 Score

Harmonic mean of Precision and Recall.

Useful for imbalanced datasets.

---

## ROC-AUC

Measures ranking quality across thresholds.

Higher ROC-AUC indicates stronger separation between:

* employees who stay
* employees who leave

---

# 6. Future Work

Possible future supervised learning models:

* Logistic Regression
* Random Forest
* XGBoost
* LightGBM
* Support Vector Machine
* Neural Networks

Possible explainability methods:

* SHAP
* Feature Importance
* Permutation Importance
* Partial Dependence Plots

---

# 7. Installation

## Required Packages

```bash
pip install pandas numpy scikit-learn matplotlib imbalanced-learn openpyxl
```

---

# 8. Running the Project

## Run EDA

```bash
python eda_analysis.py
```

## Run Preprocessing Pipeline

```bash
python hr_preprocessing_pipeline.py
```

## Run K-Means Clustering

```bash
python kmeans_clustering_pipeline.py
```

## Run Naive Baseline

```bash
python naive_baseline.py
```

---

# 9. Key Takeaways

* The dataset is clean and contains no missing values.
* The attrition dataset is highly imbalanced.
* Accuracy alone is not sufficient for evaluation.
* SMOTE should only be applied to training data.
* K-Means identified employee groups with different attrition rates.
* One cluster has approximately double the attrition rate of the other.
* The naive baseline fails to detect employee resignation despite high accuracy.

---

# 10. Author Notes

This project was designed as a strong CS439 / machine learning final project foundation.

The pipeline follows standard machine learning practices:

* proper preprocessing
* stratified splitting
* imbalance handling
* unsupervised analysis
* baseline evaluation
* reproducible experiments

The project can be extended into a full research-style HR analytics system using advanced machine learning and explainability methods.
