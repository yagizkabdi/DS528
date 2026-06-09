# DS528 Final Project
# Predicting Fan Travel Demand for the 2026 FIFA World Cup
# Author: Yağız Kaan Abdi
#
# This script trains classification models to predict fan travel interest
# and evaluates them with both technical metrics and ROI-based business impact.


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 2. PROJECT SETTINGS
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.25

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic_worldcup_fans.csv"
OUTPUT_DIR = BASE_DIR / "outputs"

OUTPUT_DIR.mkdir(exist_ok=True)

pd.set_option("display.max_columns", 100)


# ============================================================
# 3. LOAD DATA
# ============================================================

def load_data(path: Path) -> pd.DataFrame:
    """Load the synthetic World Cup fan dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path)
    return df


# ============================================================
# 4. PREPARE FEATURES
# ============================================================

def prepare_features(df: pd.DataFrame):
    """Prepare feature matrix, target vector, and feature groups."""

    target = "will_travel"

    # These columns are removed to avoid leakage or irrelevant information.
    drop_cols = [
        "fan_id",
        "will_travel",
        "travel_probability_synthetic",
        "expected_value_usd",
    ]

    X = df.drop(columns=drop_cols)
    y = df[target]

    categorical_features = [
        "country_region",
        "income_level",
        "match_importance",
    ]

    numeric_features = [
        col for col in X.columns
        if col not in categorical_features
    ]

    return X, y, categorical_features, numeric_features


# ============================================================
# 5. CREATE MODEL PIPELINES
# ============================================================

def create_models(categorical_features, numeric_features):
    """Create preprocessing and model pipelines."""

    # Logistic Regression benefits from scaled numeric features.
    preprocess_for_lr = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    # Tree-based models do not require numeric scaling.
    preprocess_for_tree = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", preprocess_for_lr),
                (
                    "model",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Random Forest": Pipeline(
            steps=[
                ("preprocess", preprocess_for_tree),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=10,
                        min_samples_leaf=30,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "Gradient Boosting": Pipeline(
            steps=[
                ("preprocess", preprocess_for_tree),
                (
                    "model",
                    GradientBoostingClassifier(
                        n_estimators=160,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }

    return models


# ============================================================
# 6. BUSINESS IMPACT FUNCTION
# ============================================================

def calculate_business_impact(y_true, y_pred, test_frame):
    """
    Calculate ROI-based business impact.

    TP: interested fan correctly targeted
    FP: uninterested fan targeted, campaign cost is wasted
    FN: interested fan missed, potential revenue is lost
    TN: uninterested fan not targeted, no direct cost
    """

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

    net_impact = tp_impact + fp_impact + fn_impact

    return {
        "TP": int(tp_mask.sum()),
        "FP": int(fp_mask.sum()),
        "FN": int(fn_mask.sum()),
        "TN": int(tn_mask.sum()),
        "TP_impact_usd": round(tp_impact, 2),
        "FP_impact_usd": round(fp_impact, 2),
        "FN_impact_usd": round(fn_impact, 2),
        "Net_business_impact_usd": round(net_impact, 2),
    }


# ============================================================
# 7. TRAIN AND EVALUATE MODELS
# ============================================================

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test, df_test):
    """Train all models and evaluate technical and business metrics."""

    results = []
    predictions = {}

    for model_name, model in models.items():
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        impact = calculate_business_impact(y_test, y_pred, df_test)

        results.append(
            {
                "Model": model_name,
                "Accuracy": round(accuracy_score(y_test, y_pred), 4),
                "Precision": round(precision_score(y_test, y_pred), 4),
                "Recall": round(recall_score(y_test, y_pred), 4),
                "F1": round(f1_score(y_test, y_pred), 4),
                "ROC_AUC": round(roc_auc_score(y_test, y_proba), 4),
                **impact,
            }
        )

        predictions[model_name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        "Net_business_impact_usd",
        ascending=False,
    )

    return results_df, predictions


# ============================================================
# 8. THRESHOLD OPTIMIZATION
# ============================================================

def optimize_thresholds(predictions, y_test, df_test):
    """
    Test different classification thresholds.

    The default threshold is 0.50, but it may not maximize ROI.
    Since False Negatives are costly, lower thresholds may perform better.
    """

    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    threshold_results = []

    for model_name, pred_data in predictions.items():
        y_proba = pred_data["y_proba"]

        for threshold in thresholds:
            y_pred_threshold = (y_proba >= threshold).astype(int)
            impact = calculate_business_impact(y_test, y_pred_threshold, df_test)

            threshold_results.append(
                {
                    "Model": model_name,
                    "Threshold": threshold,
                    "Accuracy": round(accuracy_score(y_test, y_pred_threshold), 4),
                    "Precision": round(
                        precision_score(y_test, y_pred_threshold, zero_division=0),
                        4,
                    ),
                    "Recall": round(
                        recall_score(y_test, y_pred_threshold, zero_division=0),
                        4,
                    ),
                    "F1": round(
                        f1_score(y_test, y_pred_threshold, zero_division=0),
                        4,
                    ),
                    **impact,
                }
            )

    threshold_df = pd.DataFrame(threshold_results)

    best_thresholds_df = (
        threshold_df
        .sort_values(["Model", "Net_business_impact_usd"], ascending=[True, False])
        .groupby("Model")
        .head(1)
        .sort_values("Net_business_impact_usd", ascending=False)
    )

    return threshold_df, best_thresholds_df


# ============================================================
# 9. FEATURE IMPORTANCE
# ============================================================

def get_feature_names_from_pipeline(pipeline, categorical_features, numeric_features):
    """Get feature names after one-hot encoding."""
    preprocessor = pipeline.named_steps["preprocess"]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(categorical_features)

    return np.concatenate([numeric_features, cat_names])


def calculate_feature_importance(predictions, categorical_features, numeric_features):
    """Calculate feature importance for Gradient Boosting model."""

    gb_pipeline = predictions["Gradient Boosting"]["model"]

    feature_names = get_feature_names_from_pipeline(
        gb_pipeline,
        categorical_features,
        numeric_features,
    )

    importances = gb_pipeline.named_steps["model"].feature_importances_

    feature_importance_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "importance": importances,
            }
        )
        .sort_values("importance", ascending=False)
    )

    return feature_importance_df


# ============================================================
# 10. PLOTS
# ============================================================

def plot_roi_by_threshold(threshold_df):
    """Plot net business impact by classification threshold."""

    plt.figure(figsize=(10, 6))

    for model_name in threshold_df["Model"].unique():
        temp = threshold_df[threshold_df["Model"] == model_name]

        plt.plot(
            temp["Threshold"],
            temp["Net_business_impact_usd"],
            marker="o",
            label=model_name,
        )

    plt.title("Net Business Impact by Classification Threshold")
    plt.xlabel("Classification Threshold")
    plt.ylabel("Net Business Impact (USD)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "roi_by_threshold_plot.png", dpi=200)
    plt.close()


def plot_feature_importance(feature_importance_df):
    """Plot top Gradient Boosting feature importances."""

    top_features = (
        feature_importance_df
        .head(12)
        .sort_values("importance", ascending=True)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(top_features["feature"], top_features["importance"])
    plt.title("Top Feature Importance - Gradient Boosting")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()

    plt.savefig(OUTPUT_DIR / "gradient_boosting_feature_importance.png", dpi=200)
    plt.close()


# ============================================================
# 11. SAVE OUTPUTS
# ============================================================

def save_outputs(results_df, threshold_df, best_thresholds_df, feature_importance_df):
    """Save result tables as CSV files."""

    results_df.to_csv(
        OUTPUT_DIR / "model_comparison_with_roi.csv",
        index=False,
    )

    threshold_df.to_csv(
        OUTPUT_DIR / "roi_by_threshold.csv",
        index=False,
    )

    best_thresholds_df.to_csv(
        OUTPUT_DIR / "best_thresholds_by_model.csv",
        index=False,
    )

    feature_importance_df.to_csv(
        OUTPUT_DIR / "gradient_boosting_feature_importance.csv",
        index=False,
    )


# ============================================================
# 12. MAIN FUNCTION
# ============================================================

def main():
    # Load dataset
    df = load_data(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")
    print()

    # Prepare features and target
    X, y, categorical_features, numeric_features = prepare_features(df)

    print("Target distribution:")
    print(y.value_counts(normalize=True).rename("ratio"))
    print()

    # Split dataset
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X,
        y,
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # Create and train models
    models = create_models(categorical_features, numeric_features)

    results_df, predictions = train_and_evaluate_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
        df_test,
    )

    print("Model comparison with ROI:")
    print(results_df)
    print()

    # Optimize thresholds based on net business impact
    threshold_df, best_thresholds_df = optimize_thresholds(
        predictions,
        y_test,
        df_test,
    )

    print("Best threshold by model:")
    print(best_thresholds_df)
    print()

    # Calculate feature importance
    feature_importance_df = calculate_feature_importance(
        predictions,
        categorical_features,
        numeric_features,
    )

    print("Top 10 Gradient Boosting features:")
    print(feature_importance_df.head(10))
    print()

    # Save outputs
    save_outputs(
        results_df,
        threshold_df,
        best_thresholds_df,
        feature_importance_df,
    )

    # Save plots
    plot_roi_by_threshold(threshold_df)
    plot_feature_importance(feature_importance_df)

    print("Project completed successfully.")
    print(f"Outputs saved to: {OUTPUT_DIR}")


# ============================================================
# 13. RUN SCRIPT
# ============================================================

if __name__ == "__main__":
    main()
