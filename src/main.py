"""
DS528 Final Project — Predicting Fan Travel Demand for the 2026 FIFA World Cup

This script:
  1. Loads the synthetic dataset (with real-world geography, visa, GDP data)
  2. Performs Exploratory Data Analysis (EDA)
  3. Engineers 10 new features
  4. Runs 5-fold Stratified Cross-Validation with RandomizedSearchCV hyperparameter tuning
  5. Trains and compares Logistic Regression, Random Forest, and Gradient Boosting
  6. Evaluates with both technical metrics and ROI-based business impact
  7. Optimizes classification thresholds for maximum net business impact
  8. Computes feature importance (built-in + SHAP)
  9. Generates SHAP interpretability plots (summary, beeswarm, dependence, waterfall, decision, interaction)

Output: All CSVs and plots saved to outputs/
"""

# ===========================================================================
# 1. IMPORTS
# ===========================================================================
import warnings
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold, RandomizedSearchCV, train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# For non-interactive matplotlib
matplotlib.use("Agg")

# ===========================================================================
# 2. CONFIGURATION
# ===========================================================================
RANDOM_STATE = 42
TEST_SIZE = 0.25
N_CV_FOLDS = 5
N_HYPERPARAM_ITER = 30  # iterations for RandomizedSearchCV

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "synthetic_worldcup_fans.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

pd.set_option("display.max_columns", 100)
pd.set_option("display.width", 200)

# ===========================================================================
# 3. DATA LOADING
# ===========================================================================
def load_data(path: Path) -> pd.DataFrame:
    """Load the synthetic World Cup fan dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


# ===========================================================================
# 4. EDA — EXPLORATORY DATA ANALYSIS
# ===========================================================================
def run_eda(df: pd.DataFrame):
    """Generate and save EDA visualizations."""
    print("\n" + "=" * 65)
    print("EXPLORATORY DATA ANALYSIS")
    print("=" * 65)

    # --- 4a. Dataset overview ---
    print(f"\nDataset: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Target: will_travel = 1 → {df['will_travel'].sum():,} "
          f"({df['will_travel'].mean():.1%})")

    # --- 4b. Target distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    counts = df["will_travel"].value_counts()
    bars = ax.bar(["Will NOT Travel (0)", "Will Travel (1)"], counts.values,
                  color=["#E74C3C", "#27AE60"], edgecolor="white", linewidth=1.2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 300,
                f"{val:,}\n({val / len(df):.1%})", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("Target Variable Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Fans")
    ax.set_ylim(0, counts.max() * 1.15)

    ax = axes[1]
    travel_by_region = (df.groupby("country_region")["will_travel"]
                        .agg(["mean", "count"])
                        .sort_values("mean", ascending=True))
    colors = plt.cm.RdYlGn(travel_by_region["mean"].values)
    bars = ax.barh(travel_by_region.index, travel_by_region["mean"], color=colors,
                   edgecolor="white", linewidth=1.2)
    for bar, val, cnt in zip(bars, travel_by_region["mean"].values,
                              travel_by_region["count"].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}  (n={cnt:,})", va="center", fontsize=9)
    ax.set_title("Travel Rate by Region", fontsize=14, fontweight="bold")
    ax.set_xlabel("Proportion Willing to Travel")
    ax.set_xlim(0, 1.1)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_target_distribution.png", dpi=200)
    plt.close()
    print("  ✓ eda_target_distribution.png")

    # --- 4c. Numerical feature distributions by target ---
    num_cols = [
        "age", "distance_to_host_city_km", "football_engagement_score",
        "social_media_engagement", "ticket_search_count", "flight_search_count",
        "hotel_search_count", "estimated_trip_cost", "days_until_match",
        "gdp_per_capita_usd",
    ]
    n_num = len(num_cols)
    fig, axes = plt.subplots((n_num + 2) // 3, 3, figsize=(16, 4 * ((n_num + 2) // 3)))
    axes = axes.flatten()

    for i, col in enumerate(num_cols):
        ax = axes[i]
        for label, color, lbl in [(1, "#27AE60", "Will Travel"),
                                   (0, "#E74C3C", "Won't Travel")]:
            subset = df[df["will_travel"] == label][col]
            ax.hist(subset, bins=40, alpha=0.5, color=color, label=lbl, density=True)
        ax.set_title(col.replace("_", " ").title(), fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    # Hide extra axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_numerical_distributions.png", dpi=200)
    plt.close()
    print("  ✓ eda_numerical_distributions.png")

    # --- 4d. Categorical feature analysis ---
    cat_cols = ["country_region", "income_level", "match_importance",
                "visa_required", "favorite_team_qualified",
                "previous_worldcup_attendance", "nearest_host_country"]
    fig, axes = plt.subplots((len(cat_cols) + 2) // 3, 3, figsize=(18, 5 * ((len(cat_cols) + 2) // 3)))
    axes = axes.flatten()

    for i, col in enumerate(cat_cols):
        ax = axes[i]
        grouped = (df.groupby(col)["will_travel"]
                   .agg(["mean", "count"])
                   .sort_values("mean", ascending=True))
        colors = plt.cm.RdYlGn(grouped["mean"].values)
        bars = ax.barh(grouped.index.astype(str), grouped["mean"],
                       color=colors, edgecolor="white", linewidth=1)
        for bar, val, cnt in zip(bars, grouped["mean"].values, grouped["count"].values):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{val:.1%}", va="center", fontsize=8)
        ax.set_title(f"{col.replace('_', ' ').title()}\n(Travel Rate)", fontweight="bold")
        ax.set_xlim(0, 1.15)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_categorical_analysis.png", dpi=200)
    plt.close()
    print("  ✓ eda_categorical_analysis.png")

    # --- 4e. Correlation heatmap ---
    corr_cols = [
        "age", "distance_to_host_city_km", "gdp_per_capita_usd",
        "football_engagement_score", "social_media_engagement",
        "ticket_search_count", "flight_search_count", "hotel_search_count",
        "estimated_trip_cost", "visa_required", "days_until_match",
        "favorite_team_qualified", "previous_worldcup_attendance",
        "campaign_cost_usd", "potential_net_revenue_usd", "will_travel",
    ]
    corr = df[corr_cols].corr()

    plt.figure(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, cbar_kws={"shrink": 0.8},
                annot_kws={"size": 7})
    plt.title("Feature Correlation Matrix", fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_correlation_heatmap.png", dpi=200)
    plt.close()
    print("  ✓ eda_correlation_heatmap.png")

    # --- 4f. Geographical distance vs travel ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for region in df["country_region"].unique():
        subset = df[df["country_region"] == region]
        rate = subset.groupby(pd.cut(subset["distance_to_host_city_km"],
                                     bins=10))["will_travel"].mean()
        midpoints = [iv.mid for iv in rate.index]
        ax.plot(midpoints, rate.values, marker="o", label=region, linewidth=2)
    ax.set_xlabel("Distance to Host City (km)", fontsize=12)
    ax.set_ylabel("Travel Rate", fontsize=12)
    ax.set_title("Travel Rate vs Distance by Region", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    visa_df = df.groupby(["country_region", "visa_required"])["will_travel"].mean().unstack()
    visa_df.columns = ["No Visa", "Visa Required"]
    visa_df.plot(kind="bar", ax=ax, color=["#27AE60", "#E74C3C"], edgecolor="white")
    ax.set_title("Travel Rate: Visa Impact by Region", fontweight="bold")
    ax.set_ylabel("Travel Rate")
    ax.legend(fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_geography_analysis.png", dpi=200)
    plt.close()
    print("  ✓ eda_geography_analysis.png")

    # --- 4g. Engagement patterns ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.scatter(df["football_engagement_score"], df["social_media_engagement"],
               c=df["will_travel"].map({0: "#E74C3C", 1: "#27AE60"}), alpha=0.15, s=5)
    ax.set_xlabel("Football Engagement Score")
    ax.set_ylabel("Social Media Engagement")
    ax.set_title("Engagement Scores Colored by Travel Intent", fontweight="bold")

    ax = axes[0, 1]
    df["search_total"] = df["ticket_search_count"] + df["flight_search_count"] + df["hotel_search_count"]
    for label, color, name in [(0, "#E74C3C", "Won't Travel"), (1, "#27AE60", "Will Travel")]:
        subset = df[df["will_travel"] == label]["search_total"]
        ax.hist(subset, bins=30, alpha=0.5, color=color, label=name, density=True)
    ax.set_xlabel("Total Search Count (ticket + flight + hotel)")
    ax.set_title("Search Behavior by Travel Intent", fontweight="bold")
    ax.legend()

    ax = axes[1, 0]
    bins = [0, 20, 30, 40, 50, 60, 100]
    labels = ["<20", "20-30", "30-40", "40-50", "50-60", "60+"]
    df["age_bucket"] = pd.cut(df["age"], bins=bins, labels=labels)
    age_engagement = df.groupby("age_bucket")["will_travel"].mean()
    ax.bar(range(len(age_engagement)), age_engagement.values,
           color=plt.cm.RdYlGn(age_engagement.values), edgecolor="white")
    ax.set_xticks(range(len(age_engagement)))
    ax.set_xticklabels(age_engagement.index)
    ax.set_title("Travel Rate by Age Group", fontweight="bold")
    ax.set_ylabel("Travel Rate")

    ax = axes[1, 1]
    importance_vals = df.groupby("match_importance")["will_travel"].agg(["mean", "count"])
    importance_vals = importance_vals.sort_values("mean", ascending=True)
    bars = ax.barh(importance_vals.index, importance_vals["mean"],
                   color=plt.cm.RdYlGn(importance_vals["mean"].values), edgecolor="white")
    for bar, val in zip(bars, importance_vals["mean"].values):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.1%}", va="center")
    ax.set_title("Travel Rate by Match Importance", fontweight="bold")
    ax.set_xlim(0, 1.15)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "eda_engagement_patterns.png", dpi=200)
    plt.close()
    print("  ✓ eda_engagement_patterns.png")

    print("EDA complete — 6 figures saved.\n")


# ===========================================================================
# 5. FEATURE ENGINEERING — 10 NEW FEATURES
# ===========================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 10 engineered features to the dataset.

    1.  travel_affinity_score     — standardized sum of search behaviors
    2.  engagement_composite      — interaction of football × social engagement
    3.  cost_income_index         — trip cost relative to income level
    4.  distance_bucket           — categorical distance tiers
    5.  travel_barrier_score      — composite of distance + visa + cost
    6.  search_intent_ratio       — ticket vs accommodation searches
    7.  days_urgency              — categorical urgency bins
    8.  fan_enthusiasm_score      — composite engagement + history + team
    9.  trip_feasibility          — sigmoid over barriers
    10. team_engagement_interaction — team_qualified × engagement
    """
    df = df.copy()

    # -- 1. travel_affinity_score --
    df["travel_affinity_score"] = (
        df["ticket_search_count"] + df["flight_search_count"] + df["hotel_search_count"]
    )

    # -- 2. engagement_composite --
    df["engagement_composite"] = (
        df["football_engagement_score"] * df["social_media_engagement"] / 100
    )

    # -- 3. cost_income_index --
    income_map = {"Low": 1, "Medium": 2, "High": 3}
    df["income_numeric"] = df["income_level"].map(income_map)
    df["cost_income_index"] = df["estimated_trip_cost"] / df["income_numeric"]

    # -- 4. distance_bucket (categorical) --
    def bucket_distance(km):
        if km < 1000:
            return "local"
        elif km < 5000:
            return "regional"
        else:
            return "intercontinental"
    df["distance_bucket"] = df["distance_to_host_city_km"].apply(bucket_distance)

    # -- 5. travel_barrier_score --
    from sklearn.preprocessing import StandardScaler as SS
    ss = SS()
    dist_norm = ss.fit_transform(df[["distance_to_host_city_km"]]).ravel()
    cost_norm = ss.fit_transform(df[["estimated_trip_cost"]]).ravel()
    df["travel_barrier_score"] = (
        0.40 * dist_norm + 0.35 * df["visa_required"] + 0.25 * cost_norm
    )

    # -- 6. search_intent_ratio --
    df["search_intent_ratio"] = (
        (df["ticket_search_count"] + 1)
        / (df["flight_search_count"] + df["hotel_search_count"] + 1)
    )

    # -- 7. days_urgency (categorical) --
    def urgency(days):
        if days < 30:
            return "last_minute"
        elif days < 90:
            return "soon"
        elif days < 180:
            return "planning"
        else:
            return "early"
    df["days_urgency"] = df["days_until_match"].apply(urgency)

    # -- 8. fan_enthusiasm_score --
    df["fan_enthusiasm_score"] = (
        df["engagement_composite"] / 100 * 0.5
        + df["previous_worldcup_attendance"] * 0.3
        + df["favorite_team_qualified"] * 0.2
    )

    # -- 9. trip_feasibility (sigmoid over barriers) --
    barrier_z = (df["travel_barrier_score"] - df["travel_barrier_score"].mean()) / df["travel_barrier_score"].std()
    df["trip_feasibility"] = 1 / (1 + np.exp(barrier_z))

    # -- 10. team_engagement_interaction --
    df["team_engagement_interaction"] = (
        df["favorite_team_qualified"] * df["football_engagement_score"]
    )

    return df


# ===========================================================================
# 6. PREPARE FEATURES FOR MODELING
# ===========================================================================
def prepare_features(df: pd.DataFrame):
    """Split into X/y, identify feature groups. Returns also df for ROI calcs."""

    target = "will_travel"

    # Columns to drop (leakage / identifiers / intermediate)
    drop_cols = [
        "fan_id",
        "will_travel",
        "travel_probability_synthetic",  # data leakage
        "expected_value_usd",            # derived from target
        "country",                        # too granular; region captures it
        "nearest_host_city",              # too granular
        "income_numeric",                 # intermediate
        "search_total",                   # intermediate (if exists)
        "age_bucket",                     # intermediate (if exists)
    ]
    # Only drop columns that exist
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target]

    categorical_features = [
        "country_region",
        "income_level",
        "match_importance",
        "distance_bucket",
        "days_urgency",
        "nearest_host_country",
    ]
    # Keep only those present
    categorical_features = [c for c in categorical_features if c in X.columns]

    numeric_features = [col for col in X.columns if col not in categorical_features]

    return X, y, categorical_features, numeric_features


# ===========================================================================
# 7. MODEL PIPELINES & HYPERPARAMETER GRIDS
# ===========================================================================
def build_pipelines_and_grids(categorical_features, numeric_features):
    """Build preprocessing + model pipelines and hyperparameter search grids."""

    # Preprocessor for linear models (needs scaling)
    preprocess_lr = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ])

    # Preprocessor for tree models (no scaling needed)
    preprocess_tree = ColumnTransformer([
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features),
    ])

    pipelines = {
        "Logistic Regression": Pipeline([
            ("preprocess", preprocess_lr),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", preprocess_tree),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("preprocess", preprocess_tree),
            ("model", GradientBoostingClassifier(random_state=RANDOM_STATE)),
        ]),
    }

    # Hyperparameter grids for RandomizedSearchCV
    param_grids = {
        "Logistic Regression": {
            "model__C": np.logspace(-3, 2, 20),
            "model__penalty": ["l1", "l2"],
            "model__solver": ["liblinear", "saga"],
            "model__class_weight": [None, "balanced"],
        },
        "Random Forest": {
            "model__n_estimators": [100, 150, 200, 300, 400],
            "model__max_depth": [5, 8, 10, 12, 15, 20, None],
            "model__min_samples_leaf": [10, 20, 30, 50, 100],
            "model__max_features": ["sqrt", "log2", None],
            "model__class_weight": [None, "balanced", "balanced_subsample"],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 150, 200, 300],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.15],
            "model__max_depth": [3, 4, 5, 6, 8],
            "model__subsample": [0.6, 0.8, 1.0],
            "model__min_samples_leaf": [10, 20, 30, 50],
        },
    }

    return pipelines, param_grids


# ===========================================================================
# 8. CROSS-VALIDATION + HYPERPARAMETER TUNING
# ===========================================================================
def hyperparameter_tuning_cv(pipelines, param_grids, X, y):
    """
    For each model: 5-fold Stratified CV + RandomizedSearchCV.
    Returns best estimators and CV results summary.
    """
    cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    best_models = {}
    cv_results_list = []

    for model_name in pipelines.keys():
        print(f"\n{'─' * 55}")
        print(f"Tuning: {model_name}")
        print(f"{'─' * 55}")

        search = RandomizedSearchCV(
            pipelines[model_name],
            param_grids[model_name],
            n_iter=N_HYPERPARAM_ITER,
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=0,
        )
        search.fit(X, y)

        best_models[model_name] = search.best_estimator_

        # Collect CV results
        cv_scores = search.cv_results_
        best_idx = search.best_index_
        mean_score = cv_scores["mean_test_score"][best_idx]
        std_score = cv_scores["std_test_score"][best_idx]

        print(f"Best CV ROC-AUC: {mean_score:.4f} (±{std_score:.4f})")
        print(f"Best params: {search.best_params_}")

        # Compute multiple metrics via cross_val (reusing the best model)
        from sklearn.model_selection import cross_validate
        cv_metrics = cross_validate(
            search.best_estimator_, X, y,
            cv=cv,
            scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
            n_jobs=-1,
        )

        cv_results_list.append({
            "Model": model_name,
            "CV_Accuracy_Mean": round(cv_metrics["test_accuracy"].mean(), 4),
            "CV_Accuracy_Std": round(cv_metrics["test_accuracy"].std(), 4),
            "CV_Precision_Mean": round(cv_metrics["test_precision"].mean(), 4),
            "CV_Precision_Std": round(cv_metrics["test_precision"].std(), 4),
            "CV_Recall_Mean": round(cv_metrics["test_recall"].mean(), 4),
            "CV_Recall_Std": round(cv_metrics["test_recall"].std(), 4),
            "CV_F1_Mean": round(cv_metrics["test_f1"].mean(), 4),
            "CV_F1_Std": round(cv_metrics["test_f1"].std(), 4),
            "CV_ROC_AUC_Mean": round(cv_metrics["test_roc_auc"].mean(), 4),
            "CV_ROC_AUC_Std": round(cv_metrics["test_roc_auc"].std(), 4),
        })

        print(f"CV Accuracy : {cv_metrics['test_accuracy'].mean():.4f} "
              f"(±{cv_metrics['test_accuracy'].std():.4f})")
        print(f"CV Recall   : {cv_metrics['test_recall'].mean():.4f} "
              f"(±{cv_metrics['test_recall'].std():.4f})")
        print(f"CV F1       : {cv_metrics['test_f1'].mean():.4f} "
              f"(±{cv_metrics['test_f1'].std():.4f})")

    cv_df = pd.DataFrame(cv_results_list)
    cv_df = cv_df.sort_values("CV_ROC_AUC_Mean", ascending=False)
    print(f"\n{'=' * 55}")
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print(f"{'=' * 55}")
    print(cv_df.to_string(index=False))

    return best_models, cv_df


# ===========================================================================
# 9. BUSINESS IMPACT CALCULATION
# ===========================================================================
def calculate_business_impact(y_true, y_pred, test_frame):
    """
    ROI-based business impact.

    TP: interested fan correctly targeted → revenue - campaign cost
    FP: uninterested fan targeted → wasted campaign cost
    FN: interested fan missed → lost potential revenue
    TN: uninterested fan not targeted → $0
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


# ===========================================================================
# 10. TRAIN & EVALUATE FINAL MODELS
# ===========================================================================
def train_and_evaluate(best_models, X_train, X_test, y_train, y_test, df_test):
    """Train final models on full training data, evaluate on test set."""
    results = []
    predictions = {}

    for model_name, model in best_models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        impact = calculate_business_impact(y_test, y_pred, df_test)

        results.append({
            "Model": model_name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall": round(recall_score(y_test, y_pred), 4),
            "F1": round(f1_score(y_test, y_pred), 4),
            "ROC_AUC": round(roc_auc_score(y_test, y_proba), 4),
            **impact,
        })

        predictions[model_name] = {
            "model": model,
            "y_pred": y_pred,
            "y_proba": y_proba,
        }

    results_df = pd.DataFrame(results).sort_values("Net_business_impact_usd", ascending=False)
    return results_df, predictions


# ===========================================================================
# 11. THRESHOLD OPTIMIZATION
# ===========================================================================
def optimize_thresholds(predictions, y_test, df_test):
    """Test thresholds from 0.10 to 0.90, find best per model."""
    thresholds = np.round(np.arange(0.10, 0.91, 0.05), 2)
    threshold_results = []

    for model_name, pred_data in predictions.items():
        y_proba = pred_data["y_proba"]
        for threshold in thresholds:
            y_pred_th = (y_proba >= threshold).astype(int)
            impact = calculate_business_impact(y_test, y_pred_th, df_test)
            threshold_results.append({
                "Model": model_name,
                "Threshold": threshold,
                "Accuracy": round(accuracy_score(y_test, y_pred_th), 4),
                "Precision": round(precision_score(y_test, y_pred_th, zero_division=0), 4),
                "Recall": round(recall_score(y_test, y_pred_th, zero_division=0), 4),
                "F1": round(f1_score(y_test, y_pred_th, zero_division=0), 4),
                **impact,
            })

    threshold_df = pd.DataFrame(threshold_results)

    best_thresholds_df = (
        threshold_df
        .sort_values(["Model", "Net_business_impact_usd"], ascending=[True, False])
        .groupby("Model").head(1)
        .sort_values("Net_business_impact_usd", ascending=False)
    )

    return threshold_df, best_thresholds_df


# ===========================================================================
# 12. FEATURE IMPORTANCE (BUILT-IN)
# ===========================================================================
def get_feature_names_from_pipeline(pipeline, categorical_features, numeric_features):
    """Extract feature names after one-hot encoding."""
    preprocessor = pipeline.named_steps["preprocess"]
    cat_encoder = preprocessor.named_transformers_["cat"]
    cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
    return list(numeric_features) + cat_names


def compute_feature_importance(predictions, categorical_features, numeric_features):
    """Compute built-in feature importance for RF and GB."""
    importance_dfs = {}

    for model_key in ["Random Forest", "Gradient Boosting"]:
        if model_key not in predictions:
            continue
        pipeline = predictions[model_key]["model"]
        feature_names = get_feature_names_from_pipeline(
            pipeline, categorical_features, numeric_features
        )
        importances = pipeline.named_steps["model"].feature_importances_

        importance_dfs[model_key] = (
            pd.DataFrame({"feature": feature_names, "importance": importances})
            .sort_values("importance", ascending=False)
        )

    return importance_dfs


# ===========================================================================
# 13. SHAP ANALYSIS
# ===========================================================================
def run_shap_analysis(predictions, X_train, X_test, y_test,
                      categorical_features, numeric_features):
    """
    Comprehensive SHAP analysis:

    1. Beeswarm plot — distribution of SHAP values per feature
    2. Bar plot — mean |SHAP| global importance
    3. Dependence plots — top 3 features
    4. Waterfall plots — individual explanations (TP, FP, FN examples)
    5. Heatmap plot — SHAP values across many predictions
    6. Interaction values — feature interaction matrix
    """
    print("\n" + "=" * 65)
    print("SHAP ANALYSIS")
    print("=" * 65)

    # Use Gradient Boosting for SHAP (fast with TreeExplainer)
    gb_pipeline = predictions["Gradient Boosting"]["model"]

    # Transform data through preprocessor
    X_test_transformed = gb_pipeline.named_steps["preprocess"].transform(X_test)
    feature_names = get_feature_names_from_pipeline(
        gb_pipeline, categorical_features, numeric_features
    )
    X_test_transformed = pd.DataFrame(X_test_transformed, columns=feature_names,
                                      index=X_test.index)

    print("  Computing SHAP values with TreeExplainer...")
    explainer = shap.TreeExplainer(
        gb_pipeline.named_steps["model"],
    )

    # Compute SHAP values on a sample for speed
    shap_sample_size = min(2000, len(X_test_transformed))
    X_shap = X_test_transformed.sample(shap_sample_size, random_state=RANDOM_STATE)
    shap_values = explainer(X_shap)

    # For manipulating SHAP values with feature names
    shap_values.feature_names = feature_names

    # --- 13a. SHAP Beeswarm (Summary) Plot ---
    print("  → SHAP Beeswarm plot")
    plt.figure(figsize=(14, 10))
    shap.plots.beeswarm(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_beeswarm.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 13b. SHAP Bar Plot (Mean |SHAP|) ---
    print("  → SHAP Bar plot")
    plt.figure(figsize=(12, 8))
    shap.plots.bar(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_bar.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 13c. SHAP Dependence Plots (top 3 features) ---
    top_features = X_shap.columns[np.argsort(
        np.abs(shap_values.values).mean(0))[-3:]][::-1]
    print(f"  → SHAP Dependence plots for: {list(top_features)}")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, feat in enumerate(top_features):
        ax = axes[i]
        shap.plots.scatter(shap_values[:, feat], ax=ax, show=False)
        ax.set_title(f"SHAP Dependence: {feat[:40]}", fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_dependence.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 13d. SHAP Waterfall Plots (individual examples) ---
    print("  → SHAP Waterfall plots (TP, FP, FN examples)")

    gb_model = gb_pipeline.named_steps["model"]
    y_shap_pred = gb_model.predict(X_shap.values)
    y_shap_true = y_test.loc[X_shap.index].values

    tp_idx = np.where((y_shap_true == 1) & (y_shap_pred == 1))[0]
    fp_idx = np.where((y_shap_true == 0) & (y_shap_pred == 1))[0]
    fn_idx = np.where((y_shap_true == 1) & (y_shap_pred == 0))[0]

    examples = []
    if len(tp_idx) > 0:
        examples.append((tp_idx[0], "shap_waterfall_tp.png",
                         "True Positive — Correctly Targeted Traveler"))
    if len(fp_idx) > 0:
        examples.append((fp_idx[0], "shap_waterfall_fp.png",
                         "False Positive — Wasted Campaign Spend"))
    if len(fn_idx) > 0:
        examples.append((fn_idx[0], "shap_waterfall_fn.png",
                         "False Negative — Missed Traveler Opportunity"))

    for idx, fname, title in examples:
        plt.figure(figsize=(12, 6))
        shap.plots.waterfall(shap_values[idx], max_display=10, show=False)
        plt.title(title, fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / fname, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"    ✓ {fname}")

    # --- 13e. SHAP Heatmap (alternative to decision plot) ---
    print("  → SHAP Heatmap")
    plt.figure(figsize=(14, 10))
    shap.plots.heatmap(shap_values, max_display=20, show=False)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_heatmap.png", dpi=200, bbox_inches="tight")
    plt.close()

    # --- 13f. SHAP Interaction Values (top features only) ---
    print("  → SHAP Interaction matrix")
    # Use a smaller sample for interaction computation
    interaction_sample = min(200, len(X_test_transformed))
    X_small = X_test_transformed.sample(interaction_sample, random_state=RANDOM_STATE)
    shap_interaction = explainer.shap_interaction_values(X_small)

    # Aggregate to feature-level interaction matrix
    interaction_matrix = np.abs(shap_interaction).sum(axis=0)
    interaction_matrix = interaction_matrix / interaction_matrix.max()

    # Plot top 12 features
    top_n = min(12, len(feature_names))
    top_indices = np.argsort(interaction_matrix.diagonal())[-top_n:][::-1]
    top_names = [feature_names[i] for i in top_indices]
    top_matrix = interaction_matrix[top_indices][:, top_indices]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        top_matrix,
        xticklabels=[n[:30] for n in top_names],
        yticklabels=[n[:30] for n in top_names],
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
    )
    plt.title("SHAP Feature Interaction Matrix (Top Features)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "shap_interaction_matrix.png", dpi=200)
    plt.close()

    print("  SHAP analysis complete — 8 figures saved.\n")


# ===========================================================================
# 14. PLOTTING FUNCTIONS
# ===========================================================================
def plot_model_comparison(results_df, cv_df):
    """Side-by-side comparison of test set vs CV metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Test set metrics ---
    ax = axes[0]
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    x = np.arange(len(metrics))
    width = 0.25
    for i, (_, row) in enumerate(results_df.iterrows()):
        vals = [row[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["Model"], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("Test Set Metrics", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # --- Business impact ---
    ax = axes[1]
    models = results_df["Model"].values
    impacts = results_df["Net_business_impact_usd"].values / 1000
    colors = ["#3498DB", "#E67E22", "#2ECC71"]
    bars = ax.bar(models, impacts, color=colors[:len(models)], edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, impacts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(impacts) * 0.02,
                f"${val:,.0f}K", ha="center", fontweight="bold", fontsize=11)
    ax.set_title("Net Business Impact (Test Set)", fontweight="bold")
    ax.set_ylabel("Net Business Impact (Thousands USD)")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=200)
    plt.close()
    print("  ✓ model_comparison.png")


def plot_roi_by_threshold(threshold_df):
    """Line plot: Net Business Impact vs Threshold for each model."""
    plt.figure(figsize=(10, 6))
    for model_name in threshold_df["Model"].unique():
        temp = threshold_df[threshold_df["Model"] == model_name]
        plt.plot(temp["Threshold"], temp["Net_business_impact_usd"],
                 marker="o", label=model_name, linewidth=2, markersize=6)
    plt.title("Net Business Impact by Classification Threshold", fontsize=14, fontweight="bold")
    plt.xlabel("Classification Threshold")
    plt.ylabel("Net Business Impact (USD)")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "roi_by_threshold_plot.png", dpi=200)
    plt.close()
    print("  ✓ roi_by_threshold_plot.png")


def plot_feature_importance(importance_dfs):
    """Horizontal bar plots for RF and GB feature importance."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    for ax, (model_name, imp_df) in zip(axes, importance_dfs.items()):
        top = imp_df.head(15).sort_values("importance", ascending=True)
        colors = plt.cm.Blues(0.3 + 0.7 * (top["importance"] / top["importance"].max()))
        ax.barh(top["feature"].str[:40], top["importance"], color=colors, edgecolor="gray", linewidth=0.5)
        ax.set_title(f"{model_name}\nTop 15 Feature Importance", fontweight="bold")
        ax.set_xlabel("Importance")
        ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=200)
    plt.close()
    print("  ✓ feature_importance.png (RF + GB)")


def plot_cv_summary(cv_df):
    """Plot cross-validation results summary."""
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]
    x = np.arange(len(metrics))
    width = 0.25
    for i, (_, row) in enumerate(cv_df.iterrows()):
        vals = [row[f"CV_{m}_Mean"] for m in metrics]
        ax.bar(x + i * width, vals, width, label=row["Model"], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title("5-Fold Cross-Validation Metrics (Mean)", fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "cv_summary.png", dpi=200)
    plt.close()
    print("  ✓ cv_summary.png")


# ===========================================================================
# 15. SAVE OUTPUTS
# ===========================================================================
def save_outputs(results_df, threshold_df, best_thresholds_df, importance_dfs, cv_df):
    """Save all result tables as CSV."""
    results_df.to_csv(OUTPUT_DIR / "model_comparison_with_roi.csv", index=False)
    threshold_df.to_csv(OUTPUT_DIR / "roi_by_threshold.csv", index=False)
    best_thresholds_df.to_csv(OUTPUT_DIR / "best_thresholds_by_model.csv", index=False)
    cv_df.to_csv(OUTPUT_DIR / "cross_validation_results.csv", index=False)

    for model_name, imp_df in importance_dfs.items():
        fname = model_name.lower().replace(" ", "_") + "_feature_importance.csv"
        imp_df.to_csv(OUTPUT_DIR / fname, index=False)

    print("\nAll CSVs saved to outputs/")


# ===========================================================================
# 16. MAIN
# ===========================================================================
def main():
    print("=" * 65)
    print("DS528 — WORLD CUP TRAVEL DEMAND PREDICTION")
    print("with Real-World Data, Feature Engineering, CV & SHAP")
    print("=" * 65)

    # ---- Load data ----
    df = load_data(DATA_PATH)
    print(f"\nLoaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # ---- EDA ----
    run_eda(df)

    # ---- Feature engineering ----
    print("=" * 65)
    print("FEATURE ENGINEERING — Adding 10 Engineered Features")
    print("=" * 65)
    df = engineer_features(df)
    print(f"After engineering: {df.shape[1]} columns")
    new_cols = [
        "travel_affinity_score", "engagement_composite", "cost_income_index",
        "distance_bucket", "travel_barrier_score", "search_intent_ratio",
        "days_urgency", "fan_enthusiasm_score", "trip_feasibility",
        "team_engagement_interaction",
    ]
    for c in new_cols:
        print(f"  + {c}")

    # ---- Prepare features ----
    X, y, categorical_features, numeric_features = prepare_features(df)
    print(f"\nFeatures: {X.shape[1]} total ({len(numeric_features)} numeric, "
          f"{len(categorical_features)} categorical)")
    print(f"Target: will_travel=1 → {y.sum():,} ({y.mean():.1%})")

    # ---- Train/test split ----
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    print(f"Train: {len(X_train):,} | Test: {len(X_test):,}")

    # ---- Build pipelines & grids ----
    pipelines, param_grids = build_pipelines_and_grids(categorical_features, numeric_features)

    # ---- Hyperparameter tuning with 5-fold CV ----
    print("\n" + "=" * 65)
    print(f"HYPERPARAMETER TUNING — {N_CV_FOLDS}-Fold CV × {N_HYPERPARAM_ITER} iterations")
    print("=" * 65)
    best_models, cv_df = hyperparameter_tuning_cv(pipelines, param_grids, X_train, y_train)

    # ---- Train final models & evaluate ----
    print("\n" + "=" * 65)
    print("FINAL MODEL EVALUATION (Test Set)")
    print("=" * 65)
    results_df, predictions = train_and_evaluate(
        best_models, X_train, X_test, y_train, y_test, df_test,
    )
    print(results_df.to_string(index=False))

    # ---- Threshold optimization ----
    print("\n" + "=" * 65)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 65)
    threshold_df, best_thresholds_df = optimize_thresholds(predictions, y_test, df_test)
    print("\nBest threshold by model:")
    print(best_thresholds_df.to_string(index=False))

    # ---- Feature importance ----
    print("\n" + "=" * 65)
    print("FEATURE IMPORTANCE")
    print("=" * 65)
    importance_dfs = compute_feature_importance(predictions, categorical_features, numeric_features)
    for model_name, imp_df in importance_dfs.items():
        print(f"\n{model_name} — Top 10:")
        print(imp_df.head(10).to_string(index=False))

    # ---- SHAP analysis ----
    run_shap_analysis(predictions, X_train, X_test, y_test,
                      categorical_features, numeric_features)

    # ---- Generate plots ----
    print("=" * 65)
    print("GENERATING PLOTS")
    print("=" * 65)
    plot_model_comparison(results_df, cv_df)
    plot_roi_by_threshold(threshold_df)
    plot_feature_importance(importance_dfs)
    plot_cv_summary(cv_df)

    # ---- Save outputs ----
    print("\n" + "=" * 65)
    print("SAVING OUTPUTS")
    print("=" * 65)
    save_outputs(results_df, threshold_df, best_thresholds_df, importance_dfs, cv_df)

    print("\n" + "=" * 65)
    print("PIPELINE COMPLETE")
    print(f"All outputs saved to: {OUTPUT_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
