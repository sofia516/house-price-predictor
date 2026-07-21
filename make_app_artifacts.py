"""
Build deployable artifacts for the Streamlit app.

- Refits the KMeans cluster on the same training data (so cluster IDs are stable)
- Wraps the saved best_model inside a full sklearn Pipeline that includes
  the FeatureEngineer, the ColumnTransformer, and the XGBoost model
- Saves:
    artifacts/kmeans.joblib         (the fitted KMeans)
    artifacts/full_pipeline.joblib  (engineer -> preprocess -> model)
"""
from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"
ART.mkdir(exist_ok=True)

NUMERIC_FEATURES = [
    "longitude", "latitude", "housing_median_age", "total_rooms",
    "total_bedrooms", "population", "households", "median_income",
    "RoomsPerHousehold", "BedroomsPerHousehold", "BedroomRatio",
    "PopulationPerHousehold", "IncomePerRoom", "LogPopulation", "LogRooms",
    "AgeSquared", "DistLA", "DistSF", "DistCoast", "CoastOrdinal",
]
CATEGORICAL_FEATURES = ["GeoCluster", "ocean_proximity"]


# -----------------------------------------------------------------------------#
# 1. Custom transformer — captures the same feature engineering as the training
# -----------------------------------------------------------------------------#
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """All the engineered features used during training."""

    def __init__(self, kmeans: KMeans | None = None):
        self.kmeans = kmeans

    def fit(self, X: pd.DataFrame, y=None):
        if self.kmeans is None:
            km = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
            km.fit(X[["latitude", "longitude"]].values)
            self.kmeans = km
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        X["RoomsPerHousehold"] = X["total_rooms"] / X["households"].clip(lower=1)
        X["BedroomsPerHousehold"] = X["total_bedrooms"] / X["households"].clip(lower=1)
        X["BedroomRatio"] = X["total_bedrooms"] / X["total_rooms"].clip(lower=1)
        X["PopulationPerHousehold"] = X["population"] / X["households"].clip(lower=1)

        X["IncomePerRoom"] = X["median_income"] / X["RoomsPerHousehold"].clip(lower=0.1)
        X["LogPopulation"] = np.log1p(X["population"])
        X["LogRooms"] = np.log1p(X["total_rooms"])
        X["AgeSquared"] = X["housing_median_age"] ** 2

        geo = X[["latitude", "longitude"]].values
        la, sf = np.array([34.0522, -118.2437]), np.array([37.7749, -122.4194])
        X["DistLA"] = np.sqrt(((geo - la) ** 2).sum(axis=1))
        X["DistSF"] = np.sqrt(((geo - sf) ** 2).sum(axis=1))
        X["DistCoast"] = np.minimum(X["DistLA"], X["DistSF"])

        X["GeoCluster"] = self.kmeans.predict(geo)

        if "ocean_proximity" in X.columns:
            coast_order = {
                "NEAR BAY": 0, "NEAR OCEAN": 1, "<1H OCEAN": 2,
                "INLAND": 3, "ISLAND": 0,
            }
            X["CoastOrdinal"] = X["ocean_proximity"].map(coast_order).fillna(2).astype(int)

        return X


# -----------------------------------------------------------------------------#
# 2. Rebuild the same training data to fit the KMeans reproducibly
# -----------------------------------------------------------------------------#
def load_clean_data() -> pd.DataFrame:
    csv = ROOT / "housing.csv"
    if not csv.exists():
        from sklearn.datasets import fetch_california_housing
        df = fetch_california_housing(as_frame=True).frame
        df.rename(columns={"MedHouseVal": "Price"}, inplace=True)
        # the sklearn version has no ocean_proximity; synthesize a coarse proxy
        df["ocean_proximity"] = np.where(df["Latitude"] > 36.5, "INLAND", "<1H OCEAN")
    else:
        df = pd.read_csv(csv)
        df.rename(columns={"median_house_value": "Price"}, inplace=True)
    df = df.drop_duplicates().reset_index(drop=True)
    if "total_bedrooms" in df.columns:
        df["total_bedrooms"] = df["total_bedrooms"].fillna(df["total_bedrooms"].median())
    if "ocean_proximity" in df.columns:
        df["ocean_proximity"] = df["ocean_proximity"].astype(str)
    return df


# -----------------------------------------------------------------------------#
# 3. Main
# -----------------------------------------------------------------------------#
def main() -> None:
    df = load_clean_data()
    print(f"Loaded {df.shape[0]} rows")

    # Fit KMeans on the same lat/lon distribution the training script saw
    km = KMeans(n_clusters=10, random_state=RANDOM_STATE, n_init=10)
    km.fit(df[["latitude", "longitude"]].values)
    joblib.dump(km, ART / "kmeans.joblib")
    print("Saved: artifacts/kmeans.joblib")

    # Build a single end-to-end pipeline: engineer -> preprocess -> model
    best_model = joblib.load(ART / "best_model.joblib")
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    full = Pipeline([
        ("engineer", FeatureEngineer(kmeans=km)),
        ("prep", preprocess),
        ("mdl", best_model.named_steps["mdl"]),
    ])

    # Sanity check: refit the pipeline's preprocess on the engineered data so
    # the saved bundle is a *trained* pipeline, ready to predict out of the box.
    engineered = FeatureEngineer(kmeans=km).fit_transform(df)
    X_for_prep = engineered.drop(columns=["Price"])
    full.named_steps["prep"].fit(X_for_prep)

    # Wrap into a single Pipeline we can save
    final_pipe = Pipeline([
        ("engineer", FeatureEngineer(kmeans=km)),
        ("prep", full.named_steps["prep"]),
        ("mdl", best_model.named_steps["mdl"]),
    ])
    joblib.dump(final_pipe, ART / "full_pipeline.joblib")
    print("Saved: artifacts/full_pipeline.joblib")

    # Quick sanity prediction
    sample = df.iloc[[0]].drop(columns=["Price"])
    pred = final_pipe.predict(sample)
    actual = df.iloc[0]["Price"]
    print(f"Sanity prediction: ${pred[0]:,.0f}  (actual: ${actual:,.0f})")


if __name__ == "__main__":
    from sklearn.pipeline import Pipeline
    main()
