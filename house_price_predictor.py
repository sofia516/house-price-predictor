"""
================================================================================
House Price Predictor — California Housing
================================================================================

A complete, end-to-end regression project.

Pipeline
--------
    1. Load data              -> local CSV (Aurelien Geron's version) OR sklearn
    2. EDA                    -> shape, dtypes, stats, missing, distributions, corr
    3. Cleaning               -> missing imputation, duplicates, outliers
    4. Feature engineering    -> ratios, interactions, log-transforms, geo-cluster
    5. Train / test split
    6. Preprocess             -> ColumnTransformer (numeric + categorical)
    7. Models                 -> Linear, Ridge, Lasso, RandomForest, GBR, XGBoost
    8. Cross-validation       -> 5-fold RMSE
    9. Hyperparam search      -> RandomizedSearchCV on the best family
   10. Stacking ensemble      -> the "Top 4% on LB" trick from the Kaggle ref
   11. Final test RMSE        -> report honestly
   12. Feature importance     -> permutation-based
   13. Save model             -> joblib

Dataset
-------
California Housing (regression target = median_house_value, in USD).
Source CSV (Aurelien Geron, O'Riley handson-ml2):
    https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv

Raw columns
    longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
    population, households, median_income, ocean_proximity (categorical)
    median_house_value (target)

Run
---
    python house_price_predictor.py

Requires
--------
    scikit-learn >= 1.3, pandas, numpy, matplotlib, seaborn, xgboost, joblib
================================================================================
"""

from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    StackingRegressor,
)
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    KFold,
    RandomizedSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# -----------------------------------------------------------------------------#
# 1. Load data
# -----------------------------------------------------------------------------#
LOCAL_CSV = Path(__file__).with_name("housing.csv")


def load_data() -> pd.DataFrame:
    """
    Load California Housing.

    Tries a local CSV first (more realistic: has missing values + categorical),
    then falls back to sklearn.datasets.fetch_california_housing.
    """
    if LOCAL_CSV.exists():
        df = pd.read_csv(LOCAL_CSV)
        df.rename(columns={"median_house_value": "Price"}, inplace=True)
        return df

    from sklearn.datasets import fetch_california_housing
    raw = fetch_california_housing(as_frame=True)
    df = raw.frame.copy()
    df.rename(columns={"MedHouseVal": "Price"}, inplace=True)
    return df


# -----------------------------------------------------------------------------#
# 2. Exploratory Data Analysis
# -----------------------------------------------------------------------------#
def eda(df: pd.DataFrame) -> None:
    """Quick EDA — print stats and a few plots."""
    print("\n========== EDA ==========")
    print(f"Shape       : {df.shape}")
    print(f"Duplicates  : {df.duplicated().sum()}")
    miss = df.isna().sum()
    miss = miss[miss > 0]
    print(f"Missing     :\n{miss if len(miss) else 'none'}")

    print("\nDescribe (numeric):")
    print(df.describe().T.round(3))

    print("\nSkewness:")
    print(df.skew(numeric_only=True).round(3))

    # Correlation with target
    corr = df.corr(numeric_only=True)["Price"].sort_values(ascending=False)
    print("\nCorrelation with Price:")
    print(corr.round(3))

    if "ocean_proximity" in df.columns:
        print("\nMedian price by ocean_proximity:")
        print(df.groupby("ocean_proximity")["Price"]
                .agg(["count", "mean", "median"])
                .round(0)
                .sort_values("median", ascending=False))

    # ---- Plots ----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # Target distribution
    sns.histplot(df["Price"], kde=True, ax=axes[0, 0], color="steelblue")
    axes[0, 0].set_title("Target distribution (Price)")

    # Correlation heatmap
    sns.heatmap(
        df.corr(numeric_only=True),
        annot=True,
        cmap="coolwarm",
        fmt=".2f",
        ax=axes[0, 1],
        cbar=False,
    )
    axes[0, 1].set_title("Correlation heatmap")

    # Geographic scatter
    sc = axes[1, 0].scatter(
        df["longitude"], df["latitude"],
        c=df["Price"], cmap="viridis", s=4, alpha=0.4,
    )
    axes[1, 0].set_xlabel("Longitude")
    axes[1, 0].set_ylabel("Latitude")
    axes[1, 0].set_title("Geography vs Price")
    plt.colorbar(sc, ax=axes[1, 0], label="Price")

    # Price vs median_income (strongest single feature)
    axes[1, 1].scatter(df["median_income"], df["Price"], s=4, alpha=0.3, color="teal")
    axes[1, 1].set_xlabel("Median income")
    axes[1, 1].set_ylabel("Price")
    axes[1, 1].set_title("Price vs Median Income")

    plt.tight_layout()
    plt.savefig("eda_overview.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("\nSaved: eda_overview.png")


# -----------------------------------------------------------------------------#
# 3. Cleaning
# -----------------------------------------------------------------------------#
def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Impute missing numeric values with the median (robust to outliers).
    - Drop duplicates.
    - Drop clearly insane rows (e.g. populations of 0 with thousands of rooms).
    - Flag the price cap.
    """
    print("\n========== CLEANING ==========")
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Dropped duplicates: {before - len(df)}")

    # Median imputation for any numeric column with NaN
    num_cols = df.select_dtypes(include=np.number).columns
    for col in num_cols:
        n_miss = df[col].isna().sum()
        if n_miss:
            med = df[col].median()
            df[col] = df[col].fillna(med)
            print(f"Imputed {col}: {n_miss} missing -> median ({med:.2f})")

    # Sanity: drop rows where population is tiny but rooms/households huge
    if "population" in df.columns:
        insane = (df["population"] < 5) & (
            (df.get("total_rooms", pd.Series([0])) > 5000)
            | (df.get("households", pd.Series([0])) > 1000)
        )
        n_insane = int(insane.sum())
        df = df[~insane].reset_index(drop=True)
        if n_insane:
            print(f"Dropped impossible rows: {n_insane}")

    # Flag the well-known price cap (creates a horizontal bar in plots)
    if df["Price"].max() > 480_000:
        capped = (df["Price"] >= 500_001).sum()
        print(f"Price >= 500,001 (capped): {capped} rows  ({capped/len(df):.1%})")

    print(f"Final shape: {df.shape}")
    return df


# -----------------------------------------------------------------------------#
# 4. Feature engineering
# -----------------------------------------------------------------------------#
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add features that capture real-estate intuition."""
    print("\n========== FEATURE ENGINEERING ==========")
    df = df.copy()

    # Per-household ratios (the raw counts are block-level; ratios are fairer)
    df["RoomsPerHousehold"] = df["total_rooms"] / df["households"].clip(lower=1)
    df["BedroomsPerHousehold"] = df["total_bedrooms"] / df["households"].clip(lower=1)
    df["BedroomRatio"] = df["total_bedrooms"] / df["total_rooms"].clip(lower=1)
    df["PopulationPerHousehold"] = df["population"] / df["households"].clip(lower=1)

    # Affordability / crowding
    df["IncomePerRoom"] = df["median_income"] / df["RoomsPerHousehold"].clip(lower=0.1)
    df["LogPopulation"] = np.log1p(df["population"])
    df["LogRooms"] = np.log1p(df["total_rooms"])

    # Non-linear age
    df["AgeSquared"] = df["housing_median_age"] ** 2

    # Geo: distance to LA, SF, and the closer of the two
    geo = df[["latitude", "longitude"]].values
    la = np.array([34.0522, -118.2437])
    sf = np.array([37.7749, -122.4194])
    df["DistLA"] = np.sqrt(((geo - la) ** 2).sum(axis=1))
    df["DistSF"] = np.sqrt(((geo - sf) ** 2).sum(axis=1))
    df["DistCoast"] = np.minimum(df["DistLA"], df["DistSF"])

    # K-means cluster on lat/lon — neighborhood effect
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
    df["GeoCluster"] = km.fit_predict(geo)

    # Map ocean_proximity to an ordinal "distance-from-coast" sense
    if "ocean_proximity" in df.columns:
        coast_order = {
            "NEAR BAY": 0, "NEAR OCEAN": 1, "<1H OCEAN": 2,
            "INLAND": 3, "ISLAND": 0,
        }
        df["CoastOrdinal"] = df["ocean_proximity"].map(coast_order).fillna(2)

    print(f"Final feature count: {df.shape[1]}")
    new_cols = [c for c in df.columns if c not in
                ["longitude", "latitude", "housing_median_age", "total_rooms",
                 "total_bedrooms", "population", "households", "median_income",
                 "ocean_proximity", "Price"]]
    print("New features:", new_cols)
    return df


# -----------------------------------------------------------------------------#
# 5. Train / test split
# -----------------------------------------------------------------------------#
def split(df: pd.DataFrame, test_size: float = 0.2):
    """Shuffle then split."""
    df = shuffle(df, random_state=RANDOM_STATE).reset_index(drop=True)
    y = df["Price"]
    X = df.drop(columns=["Price"])
    return train_test_split(X, y, test_size=test_size, random_state=RANDOM_STATE)


# -----------------------------------------------------------------------------#
# 6. Model zoo
# -----------------------------------------------------------------------------#
NUMERIC_FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "RoomsPerHousehold", "BedroomsPerHousehold", "BedroomRatio",
    "PopulationPerHousehold", "IncomePerRoom", "LogPopulation", "LogRooms",
    "AgeSquared", "DistLA", "DistSF", "DistCoast", "CoastOrdinal",
]
CATEGORICAL_FEATURES = ["GeoCluster", "ocean_proximity"]


def make_preprocess():
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )


def get_models() -> dict:
    """Return a dict of (name -> Pipeline)."""
    preprocess = make_preprocess()

    models = {
        "Linear": Pipeline([("prep", preprocess), ("mdl", LinearRegression())]),
        "Ridge":  Pipeline([("prep", preprocess), ("mdl", Ridge(alpha=1.0, random_state=RANDOM_STATE))]),
        "Lasso":  Pipeline([("prep", preprocess), ("mdl", Lasso(alpha=0.001, random_state=RANDOM_STATE, max_iter=10_000))]),
        "RF":     Pipeline([("prep", preprocess), ("mdl", RandomForestRegressor(
                        n_estimators=300, max_depth=None, min_samples_leaf=2,
                        n_jobs=-1, random_state=RANDOM_STATE))]),
        "GBR":    Pipeline([("prep", preprocess), ("mdl", GradientBoostingRegressor(
                        n_estimators=400, learning_rate=0.06, max_depth=4,
                        subsample=0.9, random_state=RANDOM_STATE))]),
        "XGB":    Pipeline([("prep", preprocess), ("mdl", XGBRegressor(
                        n_estimators=500, learning_rate=0.06, max_depth=6,
                        subsample=0.9, colsample_bytree=0.9,
                        n_jobs=-1, random_state=RANDOM_STATE,
                        tree_method="hist"))]),
    }
    return models


# -----------------------------------------------------------------------------#
# 7. Cross-validation
# -----------------------------------------------------------------------------#
def cv_evaluate(models: dict, X, y) -> pd.DataFrame:
    """5-fold CV with RMSE."""
    print("\n========== CROSS-VALIDATION (5-fold, RMSE) ==========")
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    for name, pipe in models.items():
        scores = cross_val_score(
            pipe, X, y,
            cv=kf, scoring="neg_root_mean_squared_error", n_jobs=-1,
        )
        rmse_mean = -scores.mean()
        rmse_std = scores.std()
        rows.append({"Model": name, "CV RMSE mean": rmse_mean, "CV RMSE std": rmse_std})
        print(f"  {name:8s}  RMSE = {rmse_mean:.4f}  (+/- {rmse_std:.4f})")
    return pd.DataFrame(rows).sort_values("CV RMSE mean")


# -----------------------------------------------------------------------------#
# 8. Hyperparam search (gradient boosting family)
# -----------------------------------------------------------------------------#
def tune_best(X, y):
    """Quick randomized search on XGBoost — usually the strongest base learner."""
    print("\n========== HYPERPARAM SEARCH (XGB) ==========")

    pipe = Pipeline([("prep", make_preprocess()), ("mdl", XGBRegressor(
        tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE))])

    param_dist = {
        "mdl__n_estimators":     [300, 500, 800, 1200],
        "mdl__learning_rate":    [0.03, 0.06, 0.1],
        "mdl__max_depth":        [4, 6, 8],
        "mdl__subsample":        [0.8, 0.9, 1.0],
        "mdl__colsample_bytree": [0.8, 0.9, 1.0],
        "mdl__min_child_weight": [1, 3, 5],
        "mdl__reg_alpha":        [0.0, 0.01, 0.1],
        "mdl__reg_lambda":       [0.5, 1.0, 2.0],
    }

    search = RandomizedSearchCV(
        pipe, param_distributions=param_dist,
        n_iter=15, scoring="neg_root_mean_squared_error",
        cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1, random_state=RANDOM_STATE, verbose=0, refit=True,
    )
    search.fit(X, y)
    print(f"  Best CV RMSE : {-search.best_score_:.4f}")
    print(f"  Best params  : {search.best_params_}")
    return search.best_estimator_


# -----------------------------------------------------------------------------#
# 9. Stacked ensemble (the "Top 4%" trick from the Kaggle reference)
# -----------------------------------------------------------------------------#
def stacking_ensemble(X, y):
    """Stack Ridge + RF + GBR + XGB with a Ridge meta-learner."""
    print("\n========== STACKING ENSEMBLE ==========")

    base = [
        ("ridge", Pipeline([("p", make_preprocess()), ("m", Ridge(alpha=1.0))])),
        ("rf",    Pipeline([("p", make_preprocess()), ("m", RandomForestRegressor(
                            n_estimators=300, min_samples_leaf=2, n_jobs=-1,
                            random_state=RANDOM_STATE))])),
        ("gbr",   Pipeline([("p", make_preprocess()), ("m", GradientBoostingRegressor(
                            n_estimators=400, learning_rate=0.06, max_depth=4,
                            random_state=RANDOM_STATE))])),
        ("xgb",   Pipeline([("p", make_preprocess()), ("m", XGBRegressor(
                            n_estimators=500, learning_rate=0.06, max_depth=6,
                            subsample=0.9, colsample_bytree=0.9,
                            tree_method="hist", n_jobs=-1,
                            random_state=RANDOM_STATE))])),
    ]

    stack = StackingRegressor(
        estimators=base,
        final_estimator=Ridge(alpha=1.0),
        cv=KFold(5, shuffle=True, random_state=RANDOM_STATE),
        n_jobs=-1,
    )

    # Just fit once — the inner CV already provides the stacking train signal.
    # Skip the extra outer CV (it duplicates work and is what made the script slow).
    stack.fit(X, y)
    # Quick sanity check on train (will be optimistic — for honest number see TEST SET)
    train_pred = stack.predict(X)
    train_rmse = float(np.sqrt(mean_squared_error(y, train_pred)))
    print(f"  Stacked TRAIN RMSE (optimistic): {train_rmse:.4f}")
    return stack


# -----------------------------------------------------------------------------#
# 10. Final test evaluation
# -----------------------------------------------------------------------------#
def report(name: str, y_true, y_pred) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    print(f"  {name:18s}  RMSE={rmse:.4f}  MAE={mae:.4f}  R^2={r2:.4f}")
    return {"Model": name, "Test RMSE": rmse, "Test MAE": mae, "Test R2": r2}


# -----------------------------------------------------------------------------#
# 11. Feature importance (permutation)
# -----------------------------------------------------------------------------#
def feature_importance(best_pipe, X, y, top_n: int = 15) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    print("\n========== PERMUTATION IMPORTANCE ==========")
    r = permutation_importance(
        best_pipe, X, y,
        n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )
    imp = (pd.DataFrame({
            "feature": X.columns,
            "importance": r.importances_mean,
            "std": r.importances_std,
           })
           .sort_values("importance", ascending=False)
           .head(top_n)
           .reset_index(drop=True))
    print(imp.round(4).to_string(index=False))
    return imp


# -----------------------------------------------------------------------------#
# Main
# -----------------------------------------------------------------------------#
def main() -> None:
    print("=" * 70)
    print("  HOUSE PRICE PREDICTOR — California Housing")
    print("=" * 70)

    # 1. Load
    df = load_data()
    print(f"Loaded {df.shape[0]} rows x {df.shape[1]} cols")

    # 2. EDA
    eda(df)

    # 3. Clean
    df = clean(df)

    # 4. Engineer
    df = engineer(df)

    # Make sure categorical columns are strings
    if "ocean_proximity" in df.columns:
        df["ocean_proximity"] = df["ocean_proximity"].astype(str)

    # 5. Split
    X_train, X_test, y_train, y_test = split(df)
    print(f"\nTrain: {X_train.shape}   Test: {X_test.shape}")

    # 6 + 7. CV
    models = get_models()
    cv_df = cv_evaluate(models, X_train, y_train)
    best_name = cv_df.iloc[0]["Model"]
    print(f"\n>>> Best CV model: {best_name}")

    # 8. Tune
    best_xgb = tune_best(X_train, y_train)

    # 9. Stack
    stack = stacking_ensemble(X_train, y_train)

    # 10. Final test evaluation
    print("\n========== TEST SET (held out) ==========")
    rows = []
    for name, pipe in models.items():
        pipe.fit(X_train, y_train)
        rows.append(report(name, y_test, pipe.predict(X_test)))
    rows.append(report("XGB (tuned)",  y_test, best_xgb.predict(X_test)))
    rows.append(report("Stack ensemble", y_test, stack.predict(X_test)))
    final = pd.DataFrame(rows).sort_values("Test RMSE").reset_index(drop=True)
    print("\nFinal leaderboard:")
    print(final.round(4).to_string(index=False))

    # 11. Importance
    importance = feature_importance(best_xgb, X_test, y_test)

    # 12. Save
    out_dir = Path("artifacts")
    out_dir.mkdir(exist_ok=True)
    joblib.dump(best_xgb, out_dir / "best_model.joblib")
    final.to_csv(out_dir / "leaderboard.csv", index=False)
    importance.to_csv(out_dir / "feature_importance.csv", index=False)
    print(f"\nSaved artifacts to {out_dir.resolve()}/")

    # Diagnostic plot: predicted vs actual for the best model
    pred = best_xgb.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, pred, s=4, alpha=0.4, color="teal")
    lo = min(y_test.min(), pred.min())
    hi = max(y_test.max(), pred.max())
    plt.plot([lo, hi], [lo, hi], "r--", lw=1)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Best model: XGB (tuned)  |  RMSE=${final.iloc[0]['Test RMSE']:,.0f}")
    plt.tight_layout()
    plt.savefig(out_dir / "pred_vs_actual.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("Saved: artifacts/pred_vs_actual.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
