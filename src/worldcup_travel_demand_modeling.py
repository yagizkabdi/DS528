"""
Predicting Fan Travel Demand for the 2026 FIFA World Cup

This script was exported from the project notebook.
It generates the model comparison and ROI-based threshold analysis.
"""


# ==============================================================================
#
# # Predicting Fan Travel Demand for the 2026 FIFA World Cup
#
# ## Project Objective
#
# This project predicts whether a potential fan is likely to travel for the 2026 FIFA World Cup.  
# The model output is used as a business decision tool for targeting fans with travel, hotel, ticketing, and sponsor campaigns.
#
# The key goal is not only to maximize model accuracy, but also to maximize expected business value in USD.
#
# ## Business Problem
#
# Marketing budgets are limited. If a campaign is sent to every fan, a large amount of money may be wasted on people who are not interested in traveling.  
# If the campaign is too selective, truly interested fans may be missed.
#
# Therefore, the model helps answer:
#
# > Which fans should be targeted to maximize net business impact?
#
# ## Target Variable
#
# `will_travel`
#
# - `1`: fan is likely to travel / interested in visiting the World Cup
# - `0`: fan is unlikely to travel / low interest
#
# ## Professor Feedback Addressed
#
# The project explicitly quantifies:
#
# - USD impact of correctly predicting an interested traveler
# - Cost of Type-1 error / False Positive
# - Cost of Type-2 error / False Negative
# - ROI-based model selection
# - Threshold optimization based on monetary value
# ==============================================================================


# ==============================================================================
#
# ## Synthetic Dataset
#
# Since real individual-level World Cup travel intent data is not publicly available, a synthetic dataset was generated.
#
# Each row represents one potential fan. The dataset includes:
#
# - demographic variables
# - travel barrier variables
# - football engagement variables
# - digital intent signals
# - match importance
# - estimated campaign cost
# - estimated potential net revenue
# - target variable: `will_travel`
#
# The synthetic target was generated using a realistic latent probability function.  
# Important note: `travel_probability_synthetic` is not used as a model feature because that would cause data leakage.
# ==============================================================================


import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import matplotlib.pyplot as plt

project_dir = Path("..")
data_path = project_dir / "data" / "synthetic_worldcup_fans.csv"

# If running directly from the exported folder structure used here:
if not data_path.exists():
    data_path = Path("/mnt/data/worldcup_project/synthetic_worldcup_fans.csv")

df = pd.read_csv(data_path)
df.head()




print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print("\nTarget distribution:")
print(df["will_travel"].value_counts(normalize=True).rename("ratio"))

df.describe(include="all").T.head(20)




target = "will_travel"

drop_cols = [
    "fan_id",
    "will_travel",
    "travel_probability_synthetic",
    "expected_value_usd"
]

X = df.drop(columns=drop_cols)
y = df[target]

categorical_features = ["country_region", "income_level", "match_importance"]
numeric_features = [col for col in X.columns if col not in categorical_features]

X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
    X,
    y,
    df,
    test_size=0.25,
    random_state=42,
    stratify=y
)

preprocess_for_lr = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

preprocess_for_tree = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)

models = {
    "Logistic Regression": Pipeline(steps=[
        ("preprocess", preprocess_for_lr),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ]),
    "Random Forest": Pipeline(steps=[
        ("preprocess", preprocess_for_tree),
        ("model", RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=30,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ]),
    "Gradient Boosting": Pipeline(steps=[
        ("preprocess", preprocess_for_tree),
        ("model", GradientBoostingClassifier(
            n_estimators=160,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ))
    ])
}




# ==============================================================================
#
# ## ROI and Misclassification Cost Framework
#
# The professor asked to quantify the business impact in USD.
#
# We define the business action as:
#
# > Send a targeted World Cup travel campaign to fans predicted as positive.
#
# ### Business impact assumptions
#
# - True Positive: an interested fan is correctly targeted.  
#   Net impact = `potential_net_revenue_usd - campaign_cost_usd`
#
# - False Positive / Type-1 Error: an uninterested fan is targeted.  
#   Net impact = `-campaign_cost_usd`
#
# - False Negative / Type-2 Error: an interested fan is missed.  
#   Net impact = `-potential_net_revenue_usd`
#
# - True Negative: an uninterested fan is not targeted.  
#   Net impact = `0`
#
# This means False Negatives are usually more costly than False Positives, because missing a truly interested traveler means losing a potential revenue opportunity.
# ==============================================================================


def calculate_business_impact(y_true, y_pred, test_frame):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    revenue = test_frame["potential_net_revenue_usd"].values
    cost = test_frame["campaign_cost_usd"].values

    tp_mask = (y_true == 1) & (y_pred == 1)
    fp_mask = (y_true == 0) & (y_pred == 1)
    fn_mask = (y_true == 1) & (y_pred == 0)
    tn_mask = (y_true == 0) & (y_pred == 0)

    tp_impact = np.sum(revenue[tp_mask] - cost[tp_mask])
    fp_impact = -np.sum(cost[fp_mask])
    fn_impact = -np.sum(revenue[fn_mask])
    tn_impact = 0

    net_impact = tp_impact + fp_impact + fn_impact + tn_impact

    return {
        "TP": int(tp_mask.sum()),
        "FP": int(fp_mask.sum()),
        "FN": int(fn_mask.sum()),
        "TN": int(tn_mask.sum()),
        "TP_impact_usd": round(tp_impact, 2),
        "FP_impact_usd": round(fp_impact, 2),
        "FN_impact_usd": round(fn_impact, 2),
        "Net_business_impact_usd": round(net_impact, 2)
    }




results = []
predictions = {}

for model_name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    impact = calculate_business_impact(y_test, y_pred, df_test)

    result = {
        "Model": model_name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred), 4),
        "Recall": round(recall_score(y_test, y_pred), 4),
        "F1": round(f1_score(y_test, y_pred), 4),
        "ROC_AUC": round(roc_auc_score(y_test, y_proba), 4),
        **impact
    }

    results.append(result)
    predictions[model_name] = {
        "model": model,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

results_df = pd.DataFrame(results).sort_values("Net_business_impact_usd", ascending=False)
results_df




# ==============================================================================
#
# ## Threshold Optimization
#
# The default classification threshold is usually 0.50.  
# However, in this business problem, missing a truly interested traveler is expensive.
#
# Therefore, we test multiple thresholds and select the threshold that maximizes net business impact.
# ==============================================================================


thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
threshold_results = []

for model_name, pred_data in predictions.items():
    y_proba = pred_data["y_proba"]

    for threshold in thresholds:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        impact = calculate_business_impact(y_test, y_pred_threshold, df_test)

        threshold_results.append({
            "Model": model_name,
            "Threshold": threshold,
            "Accuracy": round(accuracy_score(y_test, y_pred_threshold), 4),
            "Precision": round(precision_score(y_test, y_pred_threshold, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred_threshold, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred_threshold, zero_division=0), 4),
            **impact
        })

threshold_df = pd.DataFrame(threshold_results)

best_thresholds_df = (
    threshold_df
    .sort_values(["Model", "Net_business_impact_usd"], ascending=[True, False])
    .groupby("Model")
    .head(1)
    .sort_values("Net_business_impact_usd", ascending=False)
)

best_thresholds_df




plt.figure(figsize=(9, 6))
for model_name in threshold_df["Model"].unique():
    temp = threshold_df[threshold_df["Model"] == model_name]
    plt.plot(temp["Threshold"], temp["Net_business_impact_usd"], marker="o", label=model_name)

plt.title("Net Business Impact by Classification Threshold")
plt.xlabel("Classification Threshold")
plt.ylabel("Net Business Impact (USD)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()




# ==============================================================================
#
# ## Conclusion
#
# The results show that model selection should not be based only on technical metrics such as accuracy.
#
# In the default 0.50 threshold setting, Gradient Boosting had the highest accuracy, but its net business impact was negative because it missed too many truly interested fans.
#
# After threshold optimization, all models produced much higher business value.  
# The best-performing threshold was around 0.10–0.15, which means the business should target more fans to reduce expensive False Negatives.
#
# ### Main business insight
#
# > The best model is not necessarily the most accurate model.  
# > The best model is the one that creates the highest expected monetary value.
#
# ### Recommendation
#
# Use the model as a campaign targeting tool and select the classification threshold based on expected ROI rather than accuracy alone.
# ==============================================================================
