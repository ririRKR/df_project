# df_project
DF Project for SAAS
# UCI Heart Disease Classification

A ML pproject that predicts the presence of heart disease using clinical features from the [UCI Heart Disease dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data). Implements and compares Random Forest and XGBoost Classifiers.

---

## Project Structure

```
├── final_df_projct.ipynb       # Main notebook
├── heart_disease_uci 2.csv     # Dataset (download from Kaggle link above)
└── README.md
```

---

## Objective

Predict whether a patient has heart disease (`target = 1`) or not (`target = 0`) using real clinical measurements such as cholesterol, resting blood pressure, max heart rate, and ECG results.

---

## Dataset

- **Source**: UCI Heart Disease (via Kaggle)
- **Target**: Binarized from `num` column — `0` = no disease, `1` = any disease (original values 1–4)
- **Features include**: age, sex, chest pain type (`cp`), resting blood pressure (`trestbps`), cholesterol (`chol`), fasting blood sugar (`fbs`), ECG results (`restecg`), max heart rate (`thalch`), exercise-induced angina (`exang`), ST depression (`oldpeak`), slope, and thalassemia type (`thal`)

---

## Setup

### Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost tqdm
```

### Running the Notebook

1. Download the dataset from Kaggle and place `heart_disease_uci 2.csv` in the same directory as the notebook.
2. Open `final_df_projct.ipynb` in Jupyter.
3. Run all cells in order (top to bottom).

> **Note:** If you don't have a GPU, remove `device='cuda'` from the XGBoost cells. The model will run fine on CPU with `tree_method='hist'`.

---

## Methodology

### 1. Data Cleaning
- Binarized the target variable
- Dropped non-predictive columns (`id`, `dataset`)
- Identified categorical and continuous features

### 2. Feature Engineering
Several clinical interaction features were derived:

| Feature | Description |
|---|---|
| `hr_reserve` | Heart-rate reserve: `(220 - age) - thalch` |
| `hr_ratio` | Chronotropic index proxy: `thalch / (220 - age)` |
| `st_severity` | ST depression weighted by slope severity |
| `high_chol` | Binary flag: cholesterol > 200 mg/dL |
| `hypertension` | Binary flag: resting BP > 140 mmHg |
| `age_angina` | Age × exercise angina interaction |

### 3. Preprocessing
- One-hot encoding of categorical features
- Mean imputation for missing values (`SimpleImputer`)
- Standard scaling (for XGBoost and stacking)
- 80/20 stratified train/test split

### 4. Models

| Model | Details |
|---|---|
| **Random Forest** | 500 trees, max depth 8, balanced class weights, 10-fold CV |
| **XGBoost** | Random search over 50 candidates, 10-fold CV, optimized for ROC-AUC |

### 5. Evaluation
- ROC-AUC (primary metric)
- Classification report (precision, recall, F1)
- Optimal decision threshold via Precision-Recall curve
- Confusion matrices for RF and XGBoost
- Hyperparameter heatmap and 3D surface plot

---

## Results

| Model | Test ROC-AUC | 10-Fold CV AUC |
|---|---|---|
| Random Forest | — | — |
| XGBoost (tuned) | — | — |
| Stacking Ensemble | — | — |

---

## Notes

- All random seeds are fixed at `RANDOM_STATE = 42` for reproducibility.
- The notebook is designed to be run sequentially — later sections depend on variables defined earlier.
