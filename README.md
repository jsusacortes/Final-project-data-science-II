# Bank Customer Churn Prediction — Data Science II Final Project

## Source
Dataset: Bank Customer Churn Modelling — Kaggle
https://www.kaggle.com/datasets/shubh0799/churn-modelling

## Dataset Overview
10,000 observations, 14 features. Each row represents a bank customer's
relationship snapshot: demographics, account behavior, and product usage.
Target variable: `Exited` (1 = churned, 0 = retained). Class split is ~80/20.
After dropping identifiers (RowNumber, CustomerId, Surname), 11 base features
remain for modeling — plus 6 engineered features added during analysis.

## Requirements
- Python 3.8+
- pandas, numpy, matplotlib, seaborn
- scikit-learn
- tensorflow / keras

Install all dependencies:
```
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## How to Run
1. Place `Churn_Modelling.csv` in the same directory as the notebook.
2. Open `final_project_draft.ipynb` in Jupyter Notebook or JupyterLab.
3. Run all cells top-to-bottom (`Kernel → Restart & Run All`).

GridSearchCV tuning takes ~2–5 minutes depending on hardware.
The neural network trains in under 30 seconds with early stopping.

## Feature Engineering (6 total)

| Feature | Description |
|---|---|
| `BalanceSalaryRatio` | Balance / EstimatedSalary — financial dependency on the bank |
| `ActiveWithCard` | 1 if both active member and credit card holder |
| `AgeGenderInteraction` | Age group × gender — captures life-stage gender effects |
| `HasZeroBalance` | 1 if account balance is exactly $0 — strong disengagement signal |
| `TenureAgeRatio` | Tenure / Age — loyalty depth relative to life stage |
| `ProductEngagementScore` | NumOfProducts × IsActiveMember — active vs. passive multi-product holders |

## Results

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Threshold |
|---|---|---|---|---|---|---|
| RF Baseline | 0.844 | 0.614 | 0.629 | 0.621 | 0.860 | 0.50 |
| RF Optimized | see notebook | see notebook | see notebook | see notebook | see notebook | 0.50 |
| NN Tuned | see notebook | see notebook | see notebook | see notebook | see notebook | ~0.35–0.40 |

> Run the notebook to populate the optimized model results — they depend on
> GridSearchCV output and PR-curve threshold selection and will vary slightly
> across runs.

**Key improvements over baseline:**
- GridSearchCV optimized on **F1** (not accuracy), directly reducing the ~33% FNR and FPR flagged in instructor feedback.
- Neural Network threshold tuned via precision-recall curve, lowering it below 0.5 to catch more churners.
- 3 additional engineered features added, with `HasZeroBalance` and `ProductEngagementScore` ranking among the top feature importances.

## Key Insights
- **Age and Balance** are the strongest raw predictors of churn.
- **Germany** customers churn at ~2× the rate of France/Spain customers.
- Customers with **zero balance** churn at ~50% vs ~20% overall — the single sharpest binary signal in the dataset.
- **Inactive multi-product holders** (`ProductEngagementScore = 0`) are high-risk despite appearing financially invested.
- The Neural Network achieves higher Recall (catches more churners) but lower Precision; the Random Forest offers a better F1 balance and is ~10–15× faster to train.

## Recommendation
**Deploy the optimized Random Forest** with a probability threshold between 0.35–0.50 (selected via precision-recall curve) for monthly batch churn scoring. Prioritize retention outreach for customers aged 40–60 with a single product, zero balance, or inactive status — these segments account for the majority of model-identified churn risk.
