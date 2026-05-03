"""
===================================================================
Group 2 - HR Analytics: Employee Promotion Prediction
Model: Naive Bayes (Categorical)
Data:  train.xlsx (54,808 rows) — split 80/20 internally
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
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns


# ─────────────────────────────────────────────
# 0. PATHS
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "train.xlsx")
OUT_DIR    = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("  HR ANALYTICS - EMPLOYEE PROMOTION PREDICTION")
print("  Model: Categorical Naive Bayes")
print("=" * 60)


# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("\n[1] Loading data...")
df = pd.read_excel(TRAIN_PATH)
print(f"    Shape          : {df.shape}")
print(f"    Promoted       : {df['is_promoted'].sum()} / {len(df)} "
      f"({df['is_promoted'].mean()*100:.1f}%)")
print(f"    Missing - education           : {df['education'].isna().sum()}")
print(f"    Missing - previous_year_rating: {df['previous_year_rating'].isna().sum()}")


# ─────────────────────────────────────────────
# 2. PRE-PROCESSING
# ─────────────────────────────────────────────
print("\n[2] Pre-processing...")

CATEGORICAL_COLS = ["department", "region", "education", "gender", "recruitment_channel"]
NUMERICAL_COLS   = ["no_of_trainings", "age", "previous_year_rating",
                    "length_of_service", "avg_training_score"]
BINARY_COLS      = ["KPIs_met >80%", "awards_won?"]
TARGET           = "is_promoted"

# -- Impute missing values --
df["education"] = df["education"].fillna("Unknown")
RATING_MEDIAN   = df["previous_year_rating"].median()
df["previous_year_rating"] = df["previous_year_rating"].fillna(RATING_MEDIAN)

# -- Label-encode categorical columns --
encoders = {}
cat_frames = []
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    cat_frames.append(pd.Series(le.fit_transform(df[col].astype(str)),
                                name=col, index=df.index))
    encoders[col] = le

# -- Binary columns (already 0/1) --
for col in BINARY_COLS:
    cat_frames.append(pd.Series(df[col].values, name=col, index=df.index))

# -- Discretise numerical columns into 5 quantile bins --
binners = {}
num_frames = []
for col in NUMERICAL_COLS:
    kbd = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="quantile")
    disc = kbd.fit_transform(df[[col]]).astype(int).flatten()
    num_frames.append(pd.Series(disc, name=col, index=df.index))
    binners[col] = kbd

X = pd.concat(cat_frames + num_frames, axis=1)
y = df[TARGET]

print(f"    Feature matrix shape : {X.shape}")
print(f"    Features             : {list(X.columns)}")


# ─────────────────────────────────────────────
# 3. TRAIN / VALIDATION SPLIT  (80 / 20)
# ─────────────────────────────────────────────
print("\n[3] Splitting 80/20 (stratified)...")
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y)

print(f"    Train rows : {len(X_train)}")
print(f"    Val   rows : {len(X_val)}")
print(f"    Promoted in train : {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"    Promoted in val   : {y_val.sum()} ({y_val.mean()*100:.1f}%)")


# ─────────────────────────────────────────────
# 4. TRAIN CATEGORICAL NAIVE BAYES
# ─────────────────────────────────────────────
print("\n[4] Training Categorical Naive Bayes (alpha=1, Laplace smoothing)...")
model = CategoricalNB(alpha=1.0)
model.fit(X_train, y_train)

y_pred  = model.predict(X_val)
y_proba = model.predict_proba(X_val)[:, 1]

acc  = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred, zero_division=0)
rec  = recall_score(y_val, y_pred, zero_division=0)
f1   = f1_score(y_val, y_pred, zero_division=0)
auc  = roc_auc_score(y_val, y_proba)

print(f"\n    -- Validation Metrics --")
print(f"    Accuracy  : {acc:.4f}")
print(f"    Precision : {prec:.4f}")
print(f"    Recall    : {rec:.4f}")
print(f"    F1-Score  : {f1:.4f}  <- PRIMARY METRIC")
print(f"    AUC-ROC   : {auc:.4f}")

print("\n    Full Classification Report:")
print(classification_report(y_val, y_pred,
                             target_names=["Not Promoted", "Promoted"]))


# ─────────────────────────────────────────────
# 5. CROSS-VALIDATION  (5-Fold Stratified)
# ─────────────────────────────────────────────
print("\n[5] 5-Fold Stratified Cross-Validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1  = cross_val_score(CategoricalNB(alpha=1.0), X, y, cv=cv, scoring="f1")
cv_auc = cross_val_score(CategoricalNB(alpha=1.0), X, y, cv=cv, scoring="roc_auc")
cv_acc = cross_val_score(CategoricalNB(alpha=1.0), X, y, cv=cv, scoring="accuracy")

print(f"    CV Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}")
print(f"    CV F1      : {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
print(f"    CV AUC     : {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}")


# ─────────────────────────────────────────────
# 6. EVALUATION PLOTS
# ─────────────────────────────────────────────
print("\n[6] Generating evaluation plots...")

fig = plt.figure(figsize=(14, 5))
fig.suptitle("Naive Bayes - Employee Promotion Prediction", fontsize=14, fontweight="bold")
gs  = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

# Confusion Matrix
ax1 = fig.add_subplot(gs[0])
cm  = confusion_matrix(y_val, y_pred)
labels = [["TN", "FP"], ["FN", "TP"]]
annot  = np.array([[f"{labels[i][j]}\n{cm[i,j]:,}" for j in range(2)] for i in range(2)])
sns.heatmap(cm, annot=annot, fmt="", cmap="Blues", ax=ax1,
            xticklabels=["Not Promoted", "Promoted"],
            yticklabels=["Not Promoted", "Promoted"])
ax1.set_title("Confusion Matrix (Validation Set)")
ax1.set_ylabel("Actual")
ax1.set_xlabel("Predicted")

# ROC Curve
ax2 = fig.add_subplot(gs[1])
fpr, tpr, _ = roc_curve(y_val, y_proba)
ax2.plot(fpr, tpr, color="#2563EB", lw=2, label=f"AUC = {auc:.4f}")
ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1.05])
ax2.set_xlabel("False Positive Rate")
ax2.set_ylabel("True Positive Rate")
ax2.set_title("ROC Curve (Validation Set)")
ax2.legend(loc="lower right")
ax2.grid(alpha=0.3)

plt_path = os.path.join(OUT_DIR, "evaluation_plots.png")
plt.savefig(plt_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved -> {plt_path}")


# ─────────────────────────────────────────────
# 7. FEATURE DISCRIMINATION POWER
# ─────────────────────────────────────────────
print("\n[7] Computing feature discrimination power...")

feature_names = list(X.columns)
importance = []
for i, fname in enumerate(feature_names):
    fl   = model.feature_log_prob_[i]          # (2, n_cats)
    disc = np.max(np.abs(fl[1] - fl[0]))
    importance.append((fname, disc))

importance.sort(key=lambda x: x[1], reverse=True)
imp_df = pd.DataFrame(importance, columns=["Feature", "Max_LogProb_Diff"])

print("\n    Feature Discrimination (log-prob diff, Class 0 vs 1):")
print(imp_df.to_string(index=False))

fig2, ax3 = plt.subplots(figsize=(9, 5))
colors = ["#1E40AF" if i < 5 else "#93C5FD" for i in range(len(imp_df))]
ax3.barh(imp_df["Feature"][::-1], imp_df["Max_LogProb_Diff"][::-1], color=colors[::-1])
ax3.set_xlabel("Max Log-Probability Difference (Promoted vs Not Promoted)")
ax3.set_title("Feature Discrimination Power")
ax3.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="0.5 threshold")
ax3.legend()
ax3.grid(axis="x", alpha=0.3)
fig2.tight_layout()
imp_path = os.path.join(OUT_DIR, "feature_importance.png")
plt.savefig(imp_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"    Saved -> {imp_path}")


# ─────────────────────────────────────────────
# 8. SAVE VALIDATION PREDICTIONS
# ─────────────────────────────────────────────
print("\n[8] Saving validation predictions...")

val_indices = X_val.index
results_df = pd.DataFrame({
    "employee_id"          : df.loc[val_indices, "employee_id"].values,
    "promotion_probability": np.round(y_proba, 4),
    "predicted_promoted"   : y_pred,
    "actual_promoted"      : y_val.values,
    "correct"              : (y_pred == y_val.values).astype(int)
})
results_path = os.path.join(OUT_DIR, "validation_predictions.csv")
results_df.to_csv(results_path, index=False)
print(f"    Saved -> {results_path}")
print(f"    Correct predictions: {results_df['correct'].sum():,} / {len(results_df):,}")


# ─────────────────────────────────────────────
# 9. SAVE METRICS TO FILE
# ─────────────────────────────────────────────
metrics_path = os.path.join(OUT_DIR, "metrics_summary.txt")
with open(metrics_path, "w") as f:
    f.write("HR ANALYTICS - NAIVE BAYES METRICS SUMMARY\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Model       : Categorical Naive Bayes (alpha=1)\n")
    f.write(f"Dataset     : train.xlsx ({len(df):,} rows)\n")
    f.write(f"Split       : 80% train / 20% validation (stratified)\n\n")
    f.write(f"VALIDATION SET METRICS\n")
    f.write(f"  Accuracy  : {acc:.4f}\n")
    f.write(f"  Precision : {prec:.4f}\n")
    f.write(f"  Recall    : {rec:.4f}\n")
    f.write(f"  F1-Score  : {f1:.4f}\n")
    f.write(f"  AUC-ROC   : {auc:.4f}\n\n")
    f.write(f"5-FOLD CROSS-VALIDATION\n")
    f.write(f"  CV Accuracy: {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}\n")
    f.write(f"  CV F1      : {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}\n")
    f.write(f"  CV AUC     : {cv_auc.mean():.4f} +/- {cv_auc.std():.4f}\n\n")
    f.write(f"CONFUSION MATRIX\n")
    f.write(f"  TN={cm[0,0]:,}  FP={cm[0,1]:,}\n")
    f.write(f"  FN={cm[1,0]:,}  TP={cm[1,1]:,}\n\n")
    f.write(f"FEATURE RANKING (by discrimination power)\n")
    for _, row in imp_df.iterrows():
        f.write(f"  {row['Feature']:25s}: {row['Max_LogProb_Diff']:.4f}\n")

print(f"    Saved -> {metrics_path}")


# ─────────────────────────────────────────────
# 10. SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY")
print("=" * 60)
print(f"  Dataset        : train.xlsx  ({len(df):,} total rows)")
print(f"  Training rows  : {len(X_train):,}   Validation rows: {len(X_val):,}")
print(f"  Accuracy       : {acc:.4f}")
print(f"  Precision      : {prec:.4f}")
print(f"  Recall         : {rec:.4f}")
print(f"  F1-Score       : {f1:.4f}  <- PRIMARY METRIC")
print(f"  AUC-ROC        : {auc:.4f}")
print(f"  CV F1 (5-fold) : {cv_f1.mean():.4f} +/- {cv_f1.std():.4f}")
print(f"\n  Output files -> {OUT_DIR}/")
print(f"    validation_predictions.csv")
print(f"    metrics_summary.txt")
print(f"    evaluation_plots.png")
print(f"    feature_importance.png")
print("=" * 60)
print("\nDone! Run 'python generate_probability_table.py' next.")
