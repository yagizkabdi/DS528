# Predicting Fan Travel Demand for the 2026 FIFA World Cup

## Project Overview

This project predicts whether a potential fan is likely to travel for the **2026 FIFA World Cup**.

The model is designed as a business decision tool for targeting fans with travel, hotel, ticketing, and sponsor campaigns. The main objective is not only to maximize model accuracy, but also to maximize **expected business value in USD**.

## Business Problem

Marketing budgets are limited. If a campaign is sent to every fan, money may be wasted on people who are not interested in traveling. If the campaign is too selective, truly interested fans may be missed.

Therefore, the model helps answer:

> Which fans should be targeted to maximize net business impact?

## AI / Machine Learning Problem

This is a **binary classification** problem.

Target variable:

| Variable | Meaning |
|---|---|
| `will_travel = 1` | Fan is likely to travel / interested in visiting the World Cup |
| `will_travel = 0` | Fan is unlikely to travel / low interest |

## Dataset

The project uses a synthetic fan-level dataset because real individual-level World Cup travel intent data is not publicly available.

Each row represents a potential fan.

Main dataset:

```text
data/synthetic_worldcup_fans.csv
```

Data dictionary:

```text
data/synthetic_worldcup_data_dictionary.csv
```

Important features include:

- `age`
- `country_region`
- `income_level`
- `distance_to_host_city_km`
- `favorite_team_qualified`
- `previous_worldcup_attendance`
- `football_engagement_score`
- `social_media_engagement`
- `ticket_search_count`
- `flight_search_count`
- `hotel_search_count`
- `estimated_trip_cost`
- `visa_required`
- `days_until_match`
- `match_importance`
- `campaign_cost_usd`
- `potential_net_revenue_usd`

Important note:

`travel_probability_synthetic` was used only to generate the synthetic target and is **not used as a model feature**, because that would cause data leakage.

## Models Used

The project compares three classification models:

1. Logistic Regression
2. Random Forest
3. Gradient Boosting

## Evaluation Metrics

Technical metrics:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Business metric:

- Net Business Impact in USD

## ROI and Misclassification Cost Framework

The business action is:

> Send a targeted World Cup campaign to fans predicted as positive.

Misclassification cost logic:

| Prediction Outcome | Business Meaning | USD Impact |
|---|---|---|
| True Positive | Interested fan correctly targeted | `potential_net_revenue_usd - campaign_cost_usd` |
| False Positive / Type-1 Error | Uninterested fan targeted | `-campaign_cost_usd` |
| False Negative / Type-2 Error | Interested fan missed | `-potential_net_revenue_usd` |
| True Negative | Uninterested fan not targeted | `0` |

## Main Business Insight

The best model is not necessarily the model with the highest accuracy.

In this project, missing a truly interested World Cup traveler creates a high opportunity cost. Therefore, the model should be selected based on **expected monetary value**, not only technical accuracy.

## Threshold Optimization

The default classification threshold is usually `0.50`.

However, because False Negatives are more expensive than False Positives in this business case, the project tests multiple classification thresholds and selects the threshold that maximizes net business impact.

Main conclusion:

> Lowering the classification threshold can increase ROI by reducing costly False Negatives.

## Repository Structure

```text
worldcup-travel-demand-prediction/
│
├── data/
│   ├── synthetic_worldcup_fans.csv
│   └── synthetic_worldcup_data_dictionary.csv
│
├── notebooks/
│   └── worldcup_travel_demand_modeling.ipynb
│
├── outputs/
│   ├── model_comparison_with_roi.csv
│   ├── roi_by_threshold.csv
│   ├── best_thresholds_by_model.csv
│   ├── roi_by_threshold_plot.png
│   ├── random_forest_feature_importance.csv
│   ├── random_forest_feature_importance.png
│   ├── gradient_boosting_feature_importance.csv
│   └── gradient_boosting_feature_importance.png
│
├── src/
│   └── worldcup_travel_demand_modeling.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Python script:

```bash
python src/worldcup_travel_demand_modeling.py
```

Or open the notebook:

```bash
notebooks/worldcup_travel_demand_modeling.ipynb
```

## Output Files

The project generates or includes:

- model comparison with ROI
- threshold optimization results
- best threshold by model
- ROI by threshold plot
- feature importance outputs

## Final Recommendation

Use the model as a campaign targeting decision tool. Select the classification threshold based on **ROI maximization** rather than using the default threshold or choosing a model only by accuracy.