# Group 2 — HR Analytics: Employee Promotion Prediction
## Naive Bayes Machine Learning Model

---

## Project Information

- **Course:** CMSC 170 — Introduction to Artificial Intelligence
- **Algorithm:** Categorical Naive Bayes with Laplace Smoothing (α=1)
- **Dataset:** HR Analytics — Employee Promotion Prediction (`train.xlsx`)

---

## Group 2 Members

- Jerald Cabrera
- Jesse Keane Catedral
- Arnine Conejos
- Princess Jaena Marie O. De la Pena

---

## Project Overview

### Problem Statement

Every year, a company must decide which of its **54,808 employees** should be promoted. Currently, only about **8.5% of employees** are promoted annually — but identifying the right candidates is difficult, time-consuming, and can be influenced by unconscious bias.

This project solves the following problem:

> **Can we build a machine learning model that fairly and accurately predicts which employees are most likely to be promoted, based on objective data such as performance ratings, training scores, and KPI achievement?**

Without a data-driven system, HR teams risk:
- **Missing high-performing employees** who deserve promotion
- **Promoting based on bias** rather than merit
- **Wasting resources** on manual review of thousands of records

### Our Solution

We apply **Categorical Naive Bayes**, a probabilistic classifier, to learn patterns from historical promotion data. Given an employee's profile (department, ratings, KPIs, awards, etc.), the model computes the **probability of promotion** for each individual and flags the most likely candidates.

### Why Naive Bayes?

- **Fits the data perfectly** — the dataset is dominated by categorical and binary features (department, gender, `KPIs_met >80%`, `awards_won?`), which Categorical Naive Bayes handles natively
- **Interpretable** — every prediction traces back to computed prior and likelihood probabilities that HR staff can understand and audit
- **Handles imbalanced classes** — with only 8.5% of employees promoted, the model's probability output can be threshold-tuned for real HR decisions
- **Fast and scalable** — trains on 50,000+ records in seconds with no hyperparameter tuning required

---

## File Structure

```
Group 2 - HR/
├── train.xlsx                      <- Raw data (54,808 rows, 14 columns, WITH labels)
├── NaiveBayes_HR_Manual_.xlsx      <- Manual computation sample (50 employees)
│
├── naive_bayes_hr.py               <- MAIN SCRIPT: trains model, evaluates, saves results
├── generate_probability_table.py   <- Generates Excel probability spreadsheet
│
├── README.md                       <- This guide (how to run)
├── RESULTS_EXPLAINED.md            <- Full plain-English explanation of all results
│
└── output/                         <- Auto-created when scripts run
    ├── validation_predictions.csv  <- Predictions for 10,962 validation employees
    ├── probability_tables.xlsx     <- Naive Bayes probability tables (5 sheets)
    ├── evaluation_plots.png        <- Confusion matrix + ROC curve
    ├── feature_importance.png      <- Feature discrimination chart
    └── metrics_summary.txt         <- Plain-text metrics summary
```

---

## Manual Computation Sample

The file **`NaiveBayes_HR_Manual_.xlsx`** contains a **hand-computed Naive Bayes walkthrough** using a sample of **50 employees** drawn from `train.xlsx`.

It demonstrates step by step:
- How prior probabilities P(Promoted) and P(Not Promoted) are calculated from the dataset
- How likelihood tables P(feature | class) are built for each feature
- How the posterior probability is computed for each of the 50 sample employees
- How the final promotion prediction is made by comparing class posteriors

> This file is provided as a **manual verification** of the algorithm to show that the Python implementation correctly mirrors the mathematical computation by hand.

---

## Requirements

Python 3.8+ with the following packages:

- `pandas`
- `numpy`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `openpyxl`
- `xlsxwriter`

### Install all dependencies at once:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl xlsxwriter
```

---

## How to Run

> **Important:** Run both commands from inside the `Group 2 - HR` folder (where `train.xlsx` lives).

### Step 1 — Train the model and generate evaluation results

```bash
python -X utf8 naive_bayes_hr.py
```

This script will:

1. Load `train.xlsx`
2. Impute missing values (`education` → "Unknown"; `previous_year_rating` → median)
3. Encode categorical features with `LabelEncoder`
4. Discretise numerical features into 5 quantile bins for `CategoricalNB`
5. Split data: **80% training** (43,846 rows) / **20% validation** (10,962 rows), stratified
6. Train a **CategoricalNB** model with Laplace smoothing α=1
7. Evaluate: Accuracy, Precision, Recall, **F1-Score**, AUC-ROC
8. Run **5-fold stratified cross-validation**
9. Save `output/validation_predictions.csv`
10. Save `output/evaluation_plots.png` (confusion matrix + ROC curve)
11. Save `output/feature_importance.png`
12. Save `output/metrics_summary.txt`

**Expected console output:**

```
[4] Training Categorical Naive Bayes (alpha=1, Laplace smoothing)...

    -- Validation Metrics --
    Accuracy  : 0.9153
    Precision : 0.5153
    Recall    : 0.0899
    F1-Score  : 0.1531  <- PRIMARY METRIC
    AUC-ROC   : 0.7929

[5] 5-Fold Stratified Cross-Validation...
    CV Accuracy: 0.9139 +/- 0.0015
    CV F1      : 0.1324 +/- 0.0108
    CV AUC     : 0.7842 +/- 0.0090
```

---

### Step 2 — Generate the probability spreadsheet

```bash
python -X utf8 generate_probability_table.py
```

This creates `output/probability_tables.xlsx` with **5 worksheets**:

| Sheet | Contents |
|-------|----------|
| **1. Prior Probabilities** | P(Not Promoted) = 91.5% and P(Promoted) = 8.5% |
| **2. Categorical Likelihoods** | P(feature value \| class) for every category of all categorical/binary features |
| **3. Numerical Binned Tables** | P(bin \| class) for age, training score, etc. — includes ratio P(Promoted)/P(Not Promoted) per bin |
| **4. Sample Posteriors** | Step-by-step Bayes posterior calculation for 5 real employees |
| **5. Model Metrics** | All evaluation metrics with interpretations + confusion matrix |

---

## Dataset Variables

| Column | Type | Notes |
|--------|------|-------|
| `employee_id` | ID | Not used as a feature |
| `department` | Categorical | 9 departments |
| `region` | Categorical | 34 geographic regions |
| `education` | Categorical | Bachelor's / Master's & above / Below Secondary / Unknown (2,409 missing → imputed) |
| `gender` | Binary | m / f |
| `recruitment_channel` | Categorical | sourcing / other / referred |
| `no_of_trainings` | Numerical | Number of trainings attended (binned into 5 groups) |
| `age` | Numerical | Employee age (binned into 5 groups) |
| `previous_year_rating` | Numerical | Performance rating 1–5 (4,124 missing → imputed with median = 3.0) |
| `length_of_service` | Numerical | Years at company (binned) |
| `KPIs_met >80%` | Binary | 1 if employee met >80% of KPIs |
| `awards_won?` | Binary | 1 if employee won an award |
| `avg_training_score` | Numerical | Average training assessment score (binned) |
| `is_promoted` | **Target** | 1 = promoted, 0 = not promoted |

---

## Quick Results Reference

| Metric | Value |
|--------|-------|
| Accuracy | 91.53% |
| Precision (Promoted class) | 51.53% |
| Recall (Promoted class) | 8.99% |
| **F1-Score (PRIMARY)** | **15.31%** |
| AUC-ROC | 79.29% |
| CV F1 (5-fold mean) | 13.24% ± 1.08% |
| CV AUC (5-fold mean) | 78.42% ± 0.90% |

**Top 5 discriminating features:**

1. `awards_won?` — strongest predictor
2. `region` — promotion rates vary significantly by location
3. `previous_year_rating` — high performers are much more likely to be promoted
4. `avg_training_score` — training quality correlates strongly with promotion
5. `KPIs_met >80%` — meeting KPI targets is a strong positive signal

> See `RESULTS_EXPLAINED.md` for full explanations of every metric, the confusion matrix breakdown, and feature analysis.

---

*Group 2 | CMSC 170: Introduction to Artificial Intelligence | HR Analytics — Employee Promotion Prediction*
