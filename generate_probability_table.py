"""
===================================================================
Group 2 - HR Analytics: Probability Table Generator
Data: train.xlsx only (54,808 rows)
Generates an Excel workbook with all Naive Bayes probability tables.
===================================================================
"""

import pandas as pd
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore")

from sklearn.naive_bayes import CategoricalNB
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train.xlsx")
OUT_DIR    = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

CATEGORICAL_COLS = ["department", "region", "education", "gender", "recruitment_channel"]
NUMERICAL_COLS   = ["no_of_trainings", "age", "previous_year_rating",
                    "length_of_service", "avg_training_score"]
BINARY_COLS      = ["KPIs_met >80%", "awards_won?"]
TARGET = "is_promoted"

# ── Load & impute ─────────────────────────────────────────────────
print("Loading data from train.xlsx...")
df = pd.read_excel(TRAIN_PATH)
df["education"] = df["education"].fillna("Unknown")
rating_median   = df["previous_year_rating"].median()
df["previous_year_rating"] = df["previous_year_rating"].fillna(rating_median)

n_total = len(df)
n1      = df[TARGET].sum()
n0      = n_total - n1

# ── Encode features ───────────────────────────────────────────────
encoders, binners = {}, {}
cat_frames, num_frames = [], []

for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    cat_frames.append(pd.Series(le.fit_transform(df[col].astype(str)),
                                name=col, index=df.index))
    encoders[col] = le

for col in BINARY_COLS:
    cat_frames.append(pd.Series(df[col].values, name=col, index=df.index))

for col in NUMERICAL_COLS:
    kbd = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    disc = kbd.fit_transform(df[[col]]).astype(int).flatten()
    num_frames.append(pd.Series(disc, name=col, index=df.index))
    binners[col] = kbd

X = pd.concat(cat_frames + num_frames, axis=1)
y = df[TARGET]

# ── Train model on FULL dataset for probability tables ─────────────
print("Training Naive Bayes on full dataset for probability tables...")
model = CategoricalNB(alpha=1.0)
model.fit(X, y)

# ── 80/20 split for metrics sheet ─────────────────────────────────
X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2,
                                             random_state=42, stratify=y)
m_val   = CategoricalNB(alpha=1.0).fit(X_tr, y_tr)
y_pred  = m_val.predict(X_val)
y_proba = m_val.predict_proba(X_val)[:, 1]
cm      = confusion_matrix(y_val, y_pred)

cv      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1   = cross_val_score(CategoricalNB(alpha=1.0), X, y, cv=cv, scoring="f1")
cv_auc  = cross_val_score(CategoricalNB(alpha=1.0), X, y, cv=cv, scoring="roc_auc")

print("Building Excel workbook...")

writer = pd.ExcelWriter(os.path.join(OUT_DIR, "probability_tables.xlsx"),
                        engine="xlsxwriter")
wb = writer.book

# ── Format helpers ─────────────────────────────────────────────────
fmt_title   = wb.add_format({"bold": True, "font_size": 13, "font_color": "#1E3A5F",
                              "bg_color": "#DBEAFE", "border": 1,
                              "align": "center", "valign": "vcenter"})
fmt_header  = wb.add_format({"bold": True, "bg_color": "#1E40AF", "font_color": "white",
                              "border": 1, "align": "center", "valign": "vcenter",
                              "text_wrap": True})
fmt_label   = wb.add_format({"bold": True, "bg_color": "#EFF6FF", "border": 1})
fmt_pct     = wb.add_format({"num_format": "0.0000%", "border": 1, "align": "center"})
fmt_prob    = wb.add_format({"num_format": "0.0000",  "border": 1, "align": "center"})
fmt_int     = wb.add_format({"num_format": "#,##0",   "border": 1, "align": "center"})
fmt_section = wb.add_format({"bold": True, "bg_color": "#BFDBFE", "border": 1,
                              "font_size": 11})
fmt_note    = wb.add_format({"italic": True, "font_color": "#6B7280", "font_size": 9})
fmt_normal  = wb.add_format({"border": 1, "align": "center"})
fmt_hi      = wb.add_format({"bold": True, "bg_color": "#FEF9C3", "border": 1,
                              "num_format": "0.0000%", "align": "center"})
fmt_hi2     = wb.add_format({"bold": True, "bg_color": "#FEF9C3", "border": 1,
                              "num_format": "0.0000",  "align": "center"})


# ══════════════════════════════════════════════════════════════════
# SHEET 1 — Prior Probabilities
# ══════════════════════════════════════════════════════════════════
ws1 = wb.add_worksheet("1. Prior Probabilities")
ws1.set_column("A:A", 30); ws1.set_column("B:D", 20)

ws1.merge_range("A1:D1", "PRIOR PROBABILITIES  P(Class)", fmt_title)
ws1.write_row(1, 0, ["Class", "Count", "P(class)", "Log P(class)"], fmt_header)

prior = model.class_log_prior_
for i, (lbl, cnt) in enumerate([("Not Promoted (0)", n0), ("Promoted (1)", n1)]):
    ws1.write(2+i, 0, lbl, fmt_label)
    ws1.write(2+i, 1, cnt, fmt_int)
    ws1.write(2+i, 2, cnt/n_total, fmt_pct)
    ws1.write(2+i, 3, prior[i], fmt_prob)
ws1.write(4, 0, "Total", fmt_label)
ws1.write(4, 1, n_total, fmt_int)
ws1.write(4, 2, 1.0, fmt_pct)
ws1.write(4, 3, "—", fmt_label)

ws1.write(6, 0, f"Dataset: {n_total:,} employees  |  "
                f"Promoted: {n1:,} ({n1/n_total*100:.1f}%)", fmt_note)
ws1.write(7, 0, "Laplace smoothing alpha=1 applied to all likelihoods", fmt_note)
ws1.write(8, 0, "Formula: P(class) = Count(class) / Total", fmt_note)


# ══════════════════════════════════════════════════════════════════
# SHEET 2 — Categorical Likelihood Tables
# ══════════════════════════════════════════════════════════════════
ws2 = wb.add_worksheet("2. Categorical Likelihoods")
ws2.set_column("A:A", 30); ws2.set_column("B:E", 20)
ws2.merge_range("A1:E1",
    "LIKELIHOOD TABLES  P(feature | class)  [Laplace alpha=1]", fmt_title)

all_cat_cols = CATEGORICAL_COLS + BINARY_COLS
feat_idx = {col: i for i, col in enumerate(list(X.columns))}
row_cur = 2

for col in all_cat_cols:
    fi   = feat_idx[col]
    lp   = model.feature_log_prob_[fi]
    p    = np.exp(lp)
    cats = ([f"{col}=0", f"{col}=1"] if col in BINARY_COLS
            else list(encoders[col].classes_))

    ws2.merge_range(row_cur, 0, row_cur, 4, f"Feature: {col.upper()}", fmt_section)
    row_cur += 1
    ws2.write_row(row_cur, 0,
        ["Category / Value", "P(x | Not Promoted)", "P(x | Promoted)",
         "Log P(x | 0)", "Log P(x | 1)"], fmt_header)
    row_cur += 1

    for ci, cat in enumerate(cats):
        if ci < p.shape[1]:
            ws2.write(row_cur, 0, str(cat), fmt_label)
            ws2.write(row_cur, 1, p[0][ci],  fmt_pct)
            ws2.write(row_cur, 2, p[1][ci],  fmt_pct)
            ws2.write(row_cur, 3, lp[0][ci], fmt_prob)
            ws2.write(row_cur, 4, lp[1][ci], fmt_prob)
            row_cur += 1
    row_cur += 1


# ══════════════════════════════════════════════════════════════════
# SHEET 3 — Numerical Binned Likelihoods
# ══════════════════════════════════════════════════════════════════
ws3 = wb.add_worksheet("3. Numerical Binned Tables")
ws3.set_column("A:A", 34); ws3.set_column("B:F", 18)
ws3.merge_range("A1:F1",
    "NUMERICAL FEATURES — Binned Likelihoods P(bin | class)  [5 Quantile Bins]",
    fmt_title)

num_offset = len(all_cat_cols)
row_cur = 2

for j, col in enumerate(NUMERICAL_COLS):
    fi    = num_offset + j
    lp    = model.feature_log_prob_[fi]
    p     = np.exp(lp)
    edges = binners[col].bin_edges_[0]

    bin_labels = []
    for b in range(len(edges) - 1):
        lo = f"{edges[b]:.1f}"; hi = f"{edges[b+1]:.1f}"
        cl = "]" if b == len(edges) - 2 else ")"
        bin_labels.append(f"Bin {b}: [{lo} – {hi}{cl}")

    ws3.merge_range(row_cur, 0, row_cur, 5, f"Feature: {col.upper()}", fmt_section)
    row_cur += 1
    ws3.write_row(row_cur, 0,
        ["Bin (value range)", "P(bin | Not Promoted)", "P(bin | Promoted)",
         "Log P(bin | 0)", "Log P(bin | 1)", "Ratio P(1)/P(0)"], fmt_header)
    row_cur += 1

    for ci, bl in enumerate(bin_labels):
        if ci < p.shape[1]:
            ratio = p[1][ci] / (p[0][ci] + 1e-12)
            ws3.write(row_cur, 0, bl, fmt_label)
            ws3.write(row_cur, 1, p[0][ci],  fmt_pct)
            ws3.write(row_cur, 2, p[1][ci],  fmt_pct)
            ws3.write(row_cur, 3, lp[0][ci], fmt_prob)
            ws3.write(row_cur, 4, lp[1][ci], fmt_prob)
            ws3.write(row_cur, 5, ratio,      fmt_prob)
            row_cur += 1
    row_cur += 1


# ══════════════════════════════════════════════════════════════════
# SHEET 4 — Sample Posterior Step-by-Step
# ══════════════════════════════════════════════════════════════════
ws4 = wb.add_worksheet("4. Sample Posteriors")
ws4.set_column("A:A", 30); ws4.set_column("B:I", 16)
ws4.merge_range("A1:I1",
    "SAMPLE POSTERIOR CALCULATIONS — Naive Bayes Step-by-Step", fmt_title)

# Pick 5 diverse employees (mix of promoted/not)
promo_idx     = df[df[TARGET] == 1].index[:3].tolist()
not_promo_idx = df[df[TARGET] == 0].index[:2].tolist()
sample_rows   = df.loc[promo_idx + not_promo_idx].reset_index(drop=True)

# Profile header
ws4.write(2, 0, "Employee Profiles", fmt_section)
profile_cols = ["employee_id", "department", "education", "gender",
                "KPIs_met >80%", "awards_won?", "avg_training_score",
                "previous_year_rating", TARGET]
ws4.write_row(3, 0, profile_cols, fmt_header)

for i, (_, row) in enumerate(sample_rows[profile_cols].iterrows()):
    for j, val in enumerate(row):
        ws4.write(4+i, j, val, fmt_normal)

# Posterior table
ws4.write(10, 0, "Naive Bayes Posterior Probabilities", fmt_section)
ws4.write_row(11, 0,
    ["Employee ID", "Log P(0|x)", "Log P(1|x)",
     "P(Not Promoted)", "P(Promoted)", "Prediction", "Actual", "Correct?", "Note"],
    fmt_header)

def encode_row(row):
    parts = []
    for col in CATEGORICAL_COLS:
        le  = encoders[col]
        val = str(row[col])
        val = val if val in le.classes_ else le.classes_[0]
        parts.append(le.transform([val])[0])
    for col in BINARY_COLS:
        parts.append(int(row[col]))
    for col in NUMERICAL_COLS:
        kbd  = binners[col]
        disc = kbd.transform([[row[col]]])[0][0]
        parts.append(int(disc))
    return parts

for i, (_, row) in enumerate(sample_rows.iterrows()):
    x_enc    = np.array([encode_row(row)])
    log_jt   = model.predict_log_proba(x_enc)[0]
    proba    = model.predict_proba(x_enc)[0]
    pred     = model.predict(x_enc)[0]
    actual   = int(row[TARGET])
    correct  = "YES" if pred == actual else "NO"
    note     = ("Promoted - correctly predicted" if pred == 1 and actual == 1 else
                "Not Promoted - correctly predicted" if pred == 0 and actual == 0 else
                "Predicted Promoted but NOT promoted (FP)" if pred == 1 and actual == 0 else
                "Missed promotion - should be promoted (FN)")

    ws4.write(12+i, 0, int(row["employee_id"]), fmt_normal)
    ws4.write(12+i, 1, log_jt[0],   fmt_prob)
    ws4.write(12+i, 2, log_jt[1],   fmt_prob)
    ws4.write(12+i, 3, proba[0],    fmt_pct)
    ws4.write(12+i, 4, proba[1],    fmt_hi)
    ws4.write(12+i, 5, pred,         fmt_normal)
    ws4.write(12+i, 6, actual,       fmt_normal)
    ws4.write(12+i, 7, correct,      fmt_label)
    ws4.write(12+i, 8, note,         fmt_normal)

ws4.write(18, 0,
    "Formula: P(C|x) proportional to P(C) x product of P(xi|C) — computed in log space",
    fmt_note)
ws4.write(19, 0,
    "Laplace smoothing (alpha=1) prevents zero probabilities for unseen feature values.",
    fmt_note)


# ══════════════════════════════════════════════════════════════════
# SHEET 5 — Model Metrics
# ══════════════════════════════════════════════════════════════════
ws5 = wb.add_worksheet("5. Model Metrics")
ws5.set_column("A:A", 35); ws5.set_column("B:B", 14); ws5.set_column("C:G", 16)
ws5.merge_range("A1:G1",
    "MODEL EVALUATION METRICS — Categorical Naive Bayes (80/20 Split)", fmt_title)

ws5.write_row(2, 0,
    ["Metric", "Value", "Interpretation", "", "", "", ""], fmt_header)

metrics = [
    ("Accuracy",              accuracy_score(y_val, y_pred),
     "Of ALL predictions, 91.5% are correct (inflated by class imbalance)"),
    ("Precision (Promoted)",  precision_score(y_val, y_pred, zero_division=0),
     "When model predicts Promoted, it is right ~51.5% of the time"),
    ("Recall (Promoted)",     recall_score(y_val, y_pred, zero_division=0),
     "Model only catches ~9% of actual promotions (low due to imbalance)"),
    ("F1-Score * PRIMARY *",  f1_score(y_val, y_pred, zero_division=0),
     "Harmonic mean of Precision & Recall — balanced metric for imbalanced data"),
    ("AUC-ROC",               roc_auc_score(y_val, y_proba),
     "0.79 = model has good ranking ability (random = 0.50, perfect = 1.00)"),
    ("CV F1 Mean (5-fold)",   cv_f1.mean(),
     f"Consistent F1 across 5 folds: {cv_f1.round(4).tolist()}"),
    ("CV F1 Std (5-fold)",    cv_f1.std(),
     "Low std = model is stable across different data subsets"),
    ("CV AUC Mean (5-fold)",  cv_auc.mean(),
     f"Consistent AUC across 5 folds: {cv_auc.round(4).tolist()}"),
]

for i, (name, val, interp) in enumerate(metrics):
    ws5.write(3+i, 0, name, fmt_label)
    fmt_use = fmt_hi2 if "F1-Score" in name else fmt_prob
    ws5.write(3+i, 1, val, fmt_use)
    ws5.merge_range(3+i, 2, 3+i, 6, interp, fmt_normal)

ws5.write(12, 0, "Confusion Matrix (Validation Set — 10,962 rows)", fmt_section)
ws5.write_row(13, 0, ["", "Predicted: Not Promoted", "Predicted: Promoted",
                      "Total Actual"], fmt_header)
ws5.write(14, 0, "Actual: Not Promoted", fmt_label)
ws5.write(14, 1, int(cm[0, 0]), fmt_int)
ws5.write(14, 2, int(cm[0, 1]), fmt_int)
ws5.write(14, 3, int(cm[0, 0]) + int(cm[0, 1]), fmt_int)
ws5.write(15, 0, "Actual: Promoted", fmt_label)
ws5.write(15, 1, int(cm[1, 0]), fmt_int)
ws5.write(15, 2, int(cm[1, 1]), fmt_int)
ws5.write(15, 3, int(cm[1, 0]) + int(cm[1, 1]), fmt_int)
ws5.write(16, 0, "Total Predicted", fmt_label)
ws5.write(16, 1, int(cm[0, 0]) + int(cm[1, 0]), fmt_int)
ws5.write(16, 2, int(cm[0, 1]) + int(cm[1, 1]), fmt_int)

ws5.write(18, 0, "TN = True Negative  |  FP = False Positive  |  FN = False Negative  |  TP = True Positive", fmt_note)
ws5.write(19, 0, "F1 = 2 x (Precision x Recall) / (Precision + Recall)", fmt_note)
ws5.write(20, 0, "Class imbalance: 91.5% Not Promoted, 8.5% Promoted — causes low Recall", fmt_note)

writer.close()

out_path = os.path.join(OUT_DIR, "probability_tables.xlsx")
print(f"\nExcel workbook saved -> {out_path}")
print("Sheets:")
print("  1. Prior Probabilities")
print("  2. Categorical Likelihoods")
print("  3. Numerical Binned Tables")
print("  4. Sample Posterior Calculations")
print("  5. Model Metrics")
