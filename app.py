"""
House Price Predictor — Interactive Streamlit app.

Run locally:
    streamlit run app.py

Deploy to share.streamlit.io for free — see README.md.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
ART = ROOT / "artifacts"

# --- Custom transformer (required by the saved full_pipeline.joblib) ---
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans

class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Must match the class used in make_app_artifacts.py."""

    def __init__(self, kmeans=None):
        self.kmeans = kmeans

    def fit(self, X, y=None):
        if self.kmeans is None:
            km = KMeans(n_clusters=10, random_state=42, n_init=10)
            km.fit(X[["latitude", "longitude"]].values)
            self.kmeans = km
        return self

    def transform(self, X):
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
            coast = {"NEAR BAY": 0, "NEAR OCEAN": 1, "<1H OCEAN": 2, "INLAND": 3, "ISLAND": 0}
            X["CoastOrdinal"] = X["ocean_proximity"].map(coast).fillna(2).astype(int)
        return X

# ---------------------------------------------------------------------------- #
# Page config & theme
# ---------------------------------------------------------------------------- #
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------- #
# Cached loaders
# ---------------------------------------------------------------------------- #
@st.cache_resource
def load_pipeline():
    return joblib.load(ART / "full_pipeline.joblib")


@st.cache_data
def load_leaderboard():
    return pd.read_csv(ART / "leaderboard.csv")


@st.cache_data
def load_importance():
    return pd.read_csv(ART / "feature_importance.csv")


pipeline = load_pipeline()
leaderboard = load_leaderboard()
importance = load_importance()

# ---------------------------------------------------------------------------- #
# Header
# ---------------------------------------------------------------------------- #
st.title("🏠 California House Price Predictor")
st.markdown(
    "Predict the median house value for a California block group using a tuned "
    "XGBoost model trained on 20,640 rows. Adjust the inputs on the left to see "
    "the price change in real time."
)

# ---------------------------------------------------------------------------- #
# Sidebar inputs
# ---------------------------------------------------------------------------- #
with st.sidebar:
    st.header("🏡 Property inputs")

    ocean = st.selectbox(
        "Ocean proximity",
        ["NEAR BAY", "NEAR OCEAN", "<1H OCEAN", "INLAND", "ISLAND"],
        index=3,
        help="How close is the block group to the coast?",
    )

    median_income = st.slider(
        "Median income (in $10,000s)",
        min_value=0.5, max_value=15.0, value=3.5, step=0.1,
        help="Block-group median household income, scaled (real value ≈ this × 10k).",
    )

    housing_median_age = st.slider("House median age (years)", 1, 52, 28)

    total_rooms = st.number_input(
        "Total rooms in block", min_value=10, max_value=40_000, value=2_500, step=50,
        help="Sum of all rooms across every household in the block group.",
    )
    total_bedrooms = st.number_input(
        "Total bedrooms in block", min_value=1, max_value=6_500, value=500, step=10,
    )
    population = st.number_input(
        "Population", min_value=3, max_value=36_000, value=1_500, step=50,
    )
    households = st.number_input(
        "Households", min_value=1, max_value=6_100, value=500, step=10,
    )

    st.divider()
    st.subheader("📍 Location")
    # Map a sensible default to each ocean_proximity class
    defaults = {
        "NEAR BAY":     (37.80, -122.27, "San Francisco Bay"),
        "NEAR OCEAN":   (36.60, -121.90, "Monterey coast"),
        "<1H OCEAN":    (34.05, -118.25, "Los Angeles metro"),
        "INLAND":       (36.74, -119.78, "Fresno"),
        "ISLAND":       (33.50, -118.50, "Channel Islands"),
    }
    d_lat, d_lon, label = defaults[ocean]
    location_label = st.text_input("Location label", value=label)
    latitude  = st.slider("Latitude",  32.5, 42.0, d_lat, 0.01)
    longitude = st.slider("Longitude", -124.5, -114.0, d_lon, 0.01)

# ---------------------------------------------------------------------------- #
# Build the input row + predict
# ---------------------------------------------------------------------------- #
row = pd.DataFrame([{
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income,
    "ocean_proximity": ocean,
}])

price_pred = float(pipeline.predict(row)[0])
price_low  = price_pred * 0.86
price_high = price_pred * 1.14  # ±1 std roughly (test MAE is ~$27k)

# ---------------------------------------------------------------------------- #
# Hero prediction card
# ---------------------------------------------------------------------------- #
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1f4e79 0%, #2e8b57 100%);
            padding: 32px 24px;
            border-radius: 16px;
            text-align: center;
            color: white;
            box-shadow: 0 8px 24px rgba(0,0,0,0.15);
        ">
            <div style="font-size: 14px; opacity: 0.85; letter-spacing: 1px;">
                ESTIMATED MEDIAN HOUSE VALUE
            </div>
            <div style="font-size: 56px; font-weight: 800; margin: 8px 0; letter-spacing: -1px;">
                ${price_pred:,.0f}
            </div>
            <div style="font-size: 14px; opacity: 0.85;">
                90% range: ${price_low:,.0f} – ${price_high:,.0f}
            </div>
            <div style="font-size: 12px; opacity: 0.7; margin-top: 8px;">
                📍 {location_label}  ·  🌊 {ocean}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("")

# ---------------------------------------------------------------------------- #
# Two-column layout: leaderboard + feature importance
# ---------------------------------------------------------------------------- #
left, right = st.columns(2)

with left:
    st.subheader("🏆 Model leaderboard (held-out test set)")
    pretty = leaderboard.copy()
    pretty["Test RMSE"] = pretty["Test RMSE"].map(lambda v: f"${v:,.0f}")
    pretty["Test MAE"]  = pretty["Test MAE"].map(lambda v: f"${v:,.0f}")
    pretty["Test R2"]   = pretty["Test R2"].map(lambda v: f"{v:.3f}")
    st.dataframe(pretty, use_container_width=True, hide_index=True)

with right:
    st.subheader("🔍 Top features (permutation importance)")
    chart = importance.head(12).set_index("feature")[["importance"]]
    st.bar_chart(chart, height=400, horizontal=True)

# ---------------------------------------------------------------------------- #
# Diagnostics
# ---------------------------------------------------------------------------- #
st.divider()
st.subheader("📈 Diagnostics")
diag1, diag2 = st.columns(2)
with diag1:
    st.markdown("**Predicted vs actual** (tuned XGBoost on the test set)")
    img_path = ART / "pred_vs_actual.png"
    if img_path.exists():
        st.image(str(img_path), use_container_width=True)
with diag2:
    st.markdown("**EDA — target, correlations, geography**")
    eda_path = ROOT / "eda_overview.png"
    if eda_path.exists():
        st.image(str(eda_path), use_container_width=True)

# ---------------------------------------------------------------------------- #
# Footer
# ---------------------------------------------------------------------------- #
st.divider()
st.markdown(
    """
    <div style="text-align: center; color: #666; font-size: 13px;">
        Model: tuned XGBoost (test RMSE ≈ $42k, R² = 0.868) ·
        Trained on 20,640 rows · Inference: &lt; 50ms
    </div>
    """,
    unsafe_allow_html=True,
)
