# Predicting Fan Travel Demand for the 2026 FIFA World Cup

**DS528 Final Project** — Machine learning pipeline for predicting World Cup fan travel intent with ROI-based business optimization.

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9-orange.svg)](https://scikit-learn.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.51-green.svg)](https://github.com/shap/shap)

---

## Project Overview

This project predicts whether a potential fan will travel to the **2026 FIFA World Cup** (USA, Canada, Mexico). It functions as a **business decision tool** for targeting fans with travel packages, hotel bookings, ticket offers, and sponsor campaigns.

**The core insight:** The model with the highest accuracy is NOT the best model for the business. By optimizing for **net dollar impact** rather than accuracy, we can increase ROI by up to **7×**.

---

## Real-World Data Injection

Unlike the original synthetic-only approach, this project now incorporates:

| Data Source | Description |
|---|---|
| **16 Host Cities** | Official 2026 venues with real coordinates (Atlanta, Boston, Dallas, Houston, Kansas City, Los Angeles, Miami, New York, Philadelphia, San Francisco, Seattle, Toronto, Vancouver, Guadalajara, Mexico City, Monterrey) |
| **32 Countries** | Real GDP per capita, population, and geographic centroids |
| **Haversine Distances** | Actual great-circle distances from country centroids to nearest host city |
| **Visa Requirements** | Real US/Canada/Mexico visa policies by nationality |
| **FIFA Rankings** | Approximate current FIFA world rankings |

---

## Engineered Features (10 New)

Based on EDA insights and domain knowledge:

| # | Feature | Description |
|---|---|---|
| 1 | `travel_affinity_score` | Sum of ticket + flight + hotel search counts |
| 2 | `engagement_composite` | Football engagement × social media engagement / 100 |
| 3 | `cost_income_index` | Trip cost normalized by income level |
| 4 | `distance_bucket` | Categorical: local (<1K km), regional (1K-5K), intercontinental (5K+) |
| 5 | `travel_barrier_score` | Weighted composite of distance + visa + cost |
| 6 | `search_intent_ratio` | Ticket searches vs accommodation searches |
| 7 | `days_urgency` | Categorical: last_minute, soon, planning, early |
| 8 | `fan_enthusiasm_score` | Engagement + prior attendance + team qualification |
| 9 | `trip_feasibility` | Sigmoid transformation over barrier scores |
| 10 | `team_engagement_interaction` | Team qualification × football engagement |

---

## Models & Methodology

### Models Compared
1. **Logistic Regression** — Linear baseline with L1/L2 regularization
2. **Random Forest** — Bagging ensemble (400 trees, balanced class weights)
3. **Gradient Boosting** — Sequential ensemble (200 estimators, learning rate 0.05)

### Evaluation Framework
- **5-Fold Stratified Cross-Validation** — Robust performance estimates
- **RandomizedSearchCV** — 30 iterations per model for hyperparameter tuning
- **Dual Evaluation:** Technical metrics (Accuracy, Precision, Recall, F1, ROC-AUC) + Business ROI (USD)

### Business Impact Framework

| Outcome | Business Meaning | USD Impact |
|---|---|---|
| True Positive | Interested fan correctly targeted | `revenue - campaign_cost` |
| False Positive | Uninterested fan targeted | `-campaign_cost` |
| False Negative | Interested fan missed | `-potential_revenue` |
| True Negative | Uninterested fan not targeted | `$0` |

---

## Key Results

### Cross-Validation (5-Fold)

| Model | ROC-AUC | Recall | F1 |
|---|---|---|---|
| Logistic Regression | **0.8518** ±0.0061 | 0.6532 ±0.0107 | 0.7067 ±0.0091 |
| Gradient Boosting | 0.8497 ±0.0062 | 0.6544 ±0.0108 | 0.7065 ±0.0098 |
| Random Forest | 0.8472 ±0.0062 | **0.7192** ±0.0120 | **0.7166** ±0.0097 |

### Test Set — Default Threshold (0.50)

| Model | Accuracy | Recall | Net Business Impact |
|---|---|---|---|
| Random Forest | 0.783 | **0.725** | **+$351,383** ✅ |
| Logistic Regression | **0.793** | 0.655 | -$154,796 ❌ |
| Gradient Boosting | 0.790 | 0.654 | -$179,175 ❌ |

### Threshold Optimization — Maximizing ROI

| Model | Best Threshold | Recall | Net Business Impact |
|---|---|---|---|
| Random Forest | 0.10 | **0.989** | **$2,437,973** 🏆 |
| Logistic Regression | 0.10 | 0.971 | $2,290,437 |
| Gradient Boosting | 0.10 | 0.969 | $2,268,524 |

> **At default threshold, only Random Forest is profitable. By lowering the threshold to 0.10, ALL models become highly profitable — Random Forest delivers $2.44M in net business impact.**

### Feature Importance (Top 5)

| Rank | Feature | Type | Importance |
|---|---|---|---|
| 1 | `trip_feasibility` | Engineered #9 | 0.341 |
| 2 | `travel_barrier_score` | Engineered #5 | 0.328 |
| 3 | `football_engagement_score` | Original | 0.113 |
| 4 | `fan_enthusiasm_score` | Engineered #8 | 0.049 |
| 5 | `gdp_per_capita_usd` | Real-world | 0.035 |

**Engineered features dominate** — `trip_feasibility` and `travel_barrier_score` together account for ~67% of model decisions.

---

## SHAP Interpretability

6 types of SHAP analyses are generated to explain model decisions:

1. **Beeswarm Plot** — Distribution of SHAP values per feature
2. **Bar Plot** — Global mean |SHAP| importance
3. **Dependence Plots** — Top 3 features with interaction coloring
4. **Waterfall Plots** — Individual explanations for TP, FP, FN examples
5. **Heatmap** — SHAP values across many predictions
6. **Interaction Matrix** — Feature interaction strengths

---

## Repository Structure

```
DS528/
├── data/
│   ├── synthetic_worldcup_fans.csv              # Main dataset (50K rows)
│   └── synthetic_worldcup_data_dictionary.csv    # Column descriptions
│
├── notebooks/
│   ├── 01_eda.ipynb                             # Exploratory Data Analysis
│   └── 02_modeling.ipynb                        # Full Modeling Pipeline
│
├── outputs/
│   ├── eda_*.png                                # 6 EDA figures
│   ├── cv_summary.png                           # Cross-validation results
│   ├── model_comparison.png                     # Model comparison
│   ├── feature_importance.png                   # RF + GB importance
│   ├── roi_by_threshold_plot.png                # Threshold optimization
│   ├── shap_*.png                               # 8 SHAP figures
│   ├── model_comparison_with_roi.csv
│   ├── cross_validation_results.csv
│   ├── roi_by_threshold.csv
│   ├── best_thresholds_by_model.csv
│   └── *_feature_importance.csv
│
├── src/
│   ├── generate_data.py                         # Dataset generator
│   └── main.py                                  # Full ML pipeline
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Setup

```bash
cd DS528
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Generate Dataset

```bash
python src/generate_data.py
```

### 3. Run Full Pipeline

```bash
python src/main.py
```

This executes: EDA → Feature Engineering → 5-Fold CV → Hyperparameter Tuning → Model Evaluation → Threshold Optimization → Feature Importance → SHAP Analysis.

### 4. Explore Notebooks

```bash
jupyter notebook notebooks/
```

- `01_eda.ipynb` — Data exploration and visualization
- `02_modeling.ipynb` — Complete modeling walkthrough with inline results

---

## Business Recommendations

1. **Select Random Forest** — Only model profitable at default threshold; best recall and ROI
2. **Use aggressive threshold (0.10-0.30)** — 7× ROI improvement over conservative threshold
3. **Prioritize high-feasibility fans** — `trip_feasibility` is the single strongest predictor
4. **Invest in visa/price support** — Reduce barriers for high-engagement fans from visa-requiring countries
5. **Segment campaigns by region** — North America > Europe > South America > Africa/Asia

---

## Requirements

- Python ≥ 3.10
- pandas ≥ 2.0
- numpy ≥ 1.24
- scikit-learn ≥ 1.3
- matplotlib ≥ 3.7
- seaborn ≥ 0.12
- shap ≥ 0.42
- jupyter ≥ 1.0

---

## Authors

- **Yagiz Kabdi** — Initial project setup
- **Mert Goker** — Real-world data injection, feature engineering, CV, hyperparameter tuning, SHAP analysis

---

*DS528 — Data Science for Business Decisions*
