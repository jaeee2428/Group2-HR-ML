# Results Explained — Group 2 HR Analytics
## Naive Bayes: Employee Promotion Prediction

---

## 1. What the Model Does

The model answers one question for each employee:

> **"Based on this employee's profile, how likely are they to be promoted?"**

It uses **Categorical Naive Bayes** — a probabilistic algorithm that applies Bayes' theorem to calculate the probability of promotion given all observable features. The model was trained on **43,846 employees** and evaluated on a held-out set of **10,962 employees**, all from `train.xlsx`.

---

## 2. The Data at a Glance

| Stat | Value |
|------|-------|
| Total employees | 54,808 |
| Promoted (label = 1) | 4,668 (8.5%) |
| Not Promoted (label = 0) | 50,140 (91.5%) |
| Missing: `education` | 2,409 → filled with "Unknown" |
| Missing: `previous_year_rating` | 4,124 → filled with median (3.0) |

> **Key insight:** Only 8.5% of employees are promoted each year. This is called **class imbalance**, and it heavily affects how we interpret the results.

---

## 3. Validation Metrics — What Each Number Means

The model was evaluated on **10,962 employees it had never seen** (the 20% validation split).

### Summary Table

| Metric | Value | Plain-English Meaning |
|--------|-------|-----------------------|
| **Accuracy** | 91.53% | 9 out of every 10 predictions are correct |
| **Precision** | 51.53% | When the model says "promoted", it is right about half the time |
| **Recall** | 8.99% | Out of all truly promoted employees, the model caught only 9% |
| **F1-Score** ⭐ | **15.31%** | Balanced score combining precision and recall |
| **AUC-ROC** | 79.29% | The model can distinguish promoted from non-promoted 79% of the time |

---

## 4. The Confusion Matrix — Breaking Down Predictions

The confusion matrix shows **what the model predicted vs. what actually happened** on the 10,962-employee validation set:

```
                     Predicted:        Predicted:
                   Not Promoted        Promoted
                 ┌──────────────────┬──────────────┐
Actual:          │  TN = 9,944      │  FP = 84     │
Not Promoted     │  (correctly said │  (wrongly    │
                 │   NOT promoted)  │   said prom) │
                 ├──────────────────┼──────────────┤
Actual:          │  FN = 850        │  TP = 84     │
Promoted         │  (missed these   │  (correctly  │
                 │   promotions)    │   said prom) │
                 └──────────────────┴──────────────┘
```

| Cell | Count | What It Means |
|------|-------|---------------|
| **TN** (True Negative) | 9,944 | Correctly predicted "not promoted" — the model's strongest suit |
| **FP** (False Positive) | 84 | Told employee they'd be promoted, but they weren't |
| **FN** (False Negative) | 850 | Missed 850 actual promotions — the model's main weakness |
| **TP** (True Positive) | 84 | Correctly predicted "promoted" |

### Why is Recall so low?
Because only 8.5% of employees get promoted, the model "plays it safe" by predicting "not promoted" most of the time. This gives high accuracy (91.5%) but misses most actual promotions. This is the **class imbalance problem**.

---

## 5. F1-Score — Why It's the Primary Metric

**Accuracy is misleading here.** A model that always predicts "not promoted" would be 91.5% accurate — yet be completely useless for HR decisions.

**F1-Score** is the right metric because it penalises both:
- Missing real promotions (low Recall)
- Calling too many false promotions (low Precision)

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
   = 2 × (0.5153 × 0.0899) / (0.5153 + 0.0899)
   = 15.31%
```

**Is 15.31% F1 bad?** — In context, no. Naive Bayes with no threshold tuning on an 8.5% minority class is a reasonable baseline. The AUC of 0.79 shows the model has genuine signal.

---

## 6. AUC-ROC — The Model's True Ranking Ability

**AUC = 0.79** means:

> If you pick one promoted employee and one non-promoted employee at random, the model ranks the promoted one higher **79% of the time**.

| AUC Range | Interpretation |
|-----------|---------------|
| 0.50 | Random guess (coin flip) |
| 0.70 – 0.79 | Acceptable |
| **0.79** | **Our model — Good** |
| 0.80 – 0.89 | Excellent |
| 0.90 – 1.00 | Outstanding |

The ROC curve plots True Positive Rate vs. False Positive Rate across all decision thresholds. A curve that bows toward the top-left corner is good — ours does this.

---

## 7. Cross-Validation — Is the Model Reliable?

To verify the model isn't just lucky on one split, we ran **5-fold stratified cross-validation** — dividing the data 5 different ways and averaging the scores.

| Metric | Mean | Std Dev | Meaning |
|--------|------|---------|---------|
| CV Accuracy | 91.39% | ±0.15% | Extremely stable across folds |
| **CV F1** | **13.24%** | ±1.08% | Consistent — not overfitting |
| CV AUC | 78.42% | ±0.90% | Consistent ranking ability |

The low standard deviations confirm the model is **stable and not overfitting** to any particular subset of the data.

---

## 8. Feature Discrimination Power — What Drives Promotions?

Features are ranked by how differently they behave between promoted and non-promoted employees (measured by max log-probability difference):

| Rank | Feature | Score | Interpretation |
|------|---------|-------|---------------|
| 🥇 1 | `awards_won?` | 2.107 | **Strongest signal** — winning an award dramatically increases promotion probability |
| 🥈 2 | `region` | 1.381 | Promotion rates vary significantly across geographic regions |
| 🥉 3 | `previous_year_rating` | 1.264 | Higher performance rating → much higher promotion chance |
| 4 | `avg_training_score` | 0.850 | Better training scores correlate with promotion |
| 5 | `KPIs_met >80%` | 0.822 | Meeting >80% of KPIs is a strong positive signal |
| 6 | `education` | 0.561 | Education level has moderate discriminating power |
| 7 | `department` | 0.479 | Some departments promote more than others |
| 8 | `recruitment_channel` | 0.403 | Referred candidates promoted at different rates |
| 9 | `age` | 0.199 | Weak but present signal |
| 10 | `gender` | 0.042 | Very weak — gender has little discriminating power |
| 11 | `length_of_service` | 0.032 | Very weak — tenure alone doesn't predict promotion |
| 12 | `no_of_trainings` | 0.000 | **No discriminating power** — number of trainings alone doesn't differentiate |

> **HR Takeaway:** Awards, performance ratings, and KPI achievement are the biggest drivers of promotion prediction in this model.

---

## 9. How Naive Bayes Calculates a Prediction (Step by Step)

For an employee with features x₁, x₂, …, x₁₂, the model computes:

```
P(Promoted | x) ∝ P(Promoted) × P(x₁|Promoted) × P(x₂|Promoted) × … × P(x₁₂|Promoted)
P(Not Promoted | x) ∝ P(Not Promoted) × P(x₁|Not Prom) × P(x₂|Not Prom) × … × P(x₁₂|Not Prom)
```

Then normalise to get actual probabilities summing to 1.

**In log space** (to prevent underflow with tiny numbers multiplied together):

```
log P(Promoted | x) = log P(Promoted) + Σ log P(xᵢ | Promoted)
log P(Not Promoted | x) = log P(Not Promoted) + Σ log P(xᵢ | Not Promoted)
```

**Laplace smoothing** (α=1) prevents zero probabilities:
```
P(xᵢ = v | Class) = (Count(xᵢ=v AND Class) + 1) / (Count(Class) + num_categories)
```

**Example — Prior probabilities:**
```
P(Promoted)     = 4,668 / 54,808 = 8.5%   → log = -2.466
P(Not Promoted) = 50,140 / 54,808 = 91.5% → log = -0.089
```

---

## 10. Output Files Reference

| File | Description |
|------|-------------|
| `output/validation_predictions.csv` | Promotion probability + prediction for each of the 10,962 validation employees |
| `output/probability_tables.xlsx` | Full Naive Bayes probability tables (5 sheets) |
| `output/evaluation_plots.png` | Confusion matrix + ROC curve side by side |
| `output/feature_importance.png` | Feature discrimination power bar chart |
| `output/metrics_summary.txt` | Plain-text metrics summary |

---

## 11. Limitations and Context

| Limitation | Explanation |
|-----------|-------------|
| **Low Recall (9%)** | The 8.5% promotion rate means the model rarely predicts "promoted" — most promotions are missed. Adjusting the probability threshold (e.g., predict promoted if P > 0.3 instead of 0.5) can improve recall at the cost of precision. |
| **Independence Assumption** | Naive Bayes assumes all features are independent given the class. In reality, `awards_won?` and `KPIs_met >80%` are correlated. This is a known limitation but the model still performs reasonably well. |
| **Numerical Discretisation** | Continuous features (age, score, etc.) are binned into 5 groups, which loses some information compared to using Gaussian Naive Bayes or tree-based models. |
| **No feature for time** | Promotion decisions may depend on recent trends not captured in a single-year snapshot. |

---

*Group 2 | Machine Learning | HR Analytics — Employee Promotion Prediction*
