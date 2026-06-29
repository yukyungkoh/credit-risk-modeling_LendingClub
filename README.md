# Credit Risk Modeling — LendingClub

End-to-end credit risk modeling pipeline using LendingClub loan data (2007–2019), covering data preprocessing, exploratory analysis, feature engineering, default prediction with logistic regression and XGBoost, and a profit-maximizing threshold framework.

## Project Description

This project builds predictive models to estimate loan default risk using real loan-level data from [LendingClub](https://www.lendingclub.com), a peer-to-peer lending platform where individuals applied for unsecured personal loans. Borrowers span diverse income and credit profiles, and loans are graded based on risk.

The dataset includes detailed borrower attributes (income, employment, credit profile), loan characteristics (amount, term, interest rate), and realized outcomes (repayment status, default). Analysis is restricted to pre-COVID vintages (2007–2019) to ensure a stable economic environment.

Key questions addressed:
- Which features best predict loan default?
- How do feature selection strategies (intuition, filter, LASSO, permutation) affect model performance?
- How does XGBoost compare to logistic regression in capturing nonlinear credit risk?
- What is the profit-maximizing approval threshold for a risk-neutral lender?

## Project Structure

```
├── data/
│   ├── raw/                        # Raw LendingClub data and data dictionary
│   └── processed/                  # Cleaned and feature-engineered parquet files
├── notebooks/
│   ├── 1_data_preprocessing.ipynb
│   ├── 2_exploratory_data_analysis.ipynb
│   ├── 3_feature_engineering.ipynb
│   ├── 4a_default_prediction_logistic.ipynb
│   ├── 4b_default_prediction_xgboost.ipynb
│   └── 5_threshold_profit.ipynb
├── src/
│   └── plot_utils.py               # Reusable plotting utilities
└── outputs/
    ├── logistic_full_test_predictions.parquet
    └── threshold.json
```

## Notebooks

| # | Notebook | Description |
|---|----------|-------------|
| 1 | `1_data_preprocessing.ipynb` | Data cleaning, variable documentation, target definition |
| 2 | `2_exploratory_data_analysis.ipynb` | Univariate distributions, default rate analysis by feature |
| 3 | `3_feature_engineering.ipynb` | Feature construction, FICO/DTI banding, macroeconomic variables |
| 4a | `4a_default_prediction_logistic.ipynb` | Logistic regression with five feature selection strategies; coefficient table, permutation importance |
| 4b | `4b_default_prediction_xgboost.ipynb` | XGBoost with hyperparameter tuning; gain-based importance and SHAP analysis |
| 5 | `5_threshold_profit.ipynb` | Profit-maximizing threshold framework; sensitivity analysis to LGD and revenue assumptions |

## Key Results

| Model | ROC-AUC | Gini | PR-AUC |
|-------|---------|------|--------|
| Logistic — Full | 0.700 | 0.400 | 0.392 |
| Logistic — LASSO | 0.700 | 0.400 | 0.392 |
| Logistic — Permutation | 0.700 | 0.400 | 0.391 |
| Logistic — Filter | 0.695 | 0.389 | 0.384 |
| Logistic — Intuition | 0.683 | 0.367 | 0.371 |
| **XGBoost** | **0.724** | **0.448** | — |

The profit-maximizing threshold (0.158) approves 37.6% of loans at a 10.5% realized default rate, generating 27% higher total profit than the F1-optimal threshold and avoiding large losses from the naïve 0.5 cutoff.

## Source Modules

### `src/plot_utils.py`

- **`plot_hist_and_default(df, var_list, ...)`** — Histogram + default rate by quantile bin (continuous variables)
- **`plot_dist_and_default(df, var_list, ...)`** — Share of observations + default rate (continuous or categorical)

## Data

- **Source**: [LendingClub Loan Data](https://www.kaggle.com/datasets/wordsforthewise/lending-club) (2007–2020 Q3)
- **Raw**: `Loan_status_2007-2020Q3.gzip`, `LCDataDictionary.xlsx`
- **Processed**: `LendingClub_cleaned.parquet`, `LendingClub_features.parquet`

## Dependencies

```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost==1.7.6
shap
pyarrow
statsmodels
jupyter
```
