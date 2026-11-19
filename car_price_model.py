"""
Used Car Price Prediction - Craigslist Cars & Trucks dataset
Dataset: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

Usage:
    pip install -r requirements.txt
    python car_price_model.py --csv vehicles.csv

    # Quick test on a subset of rows
    python car_price_model.py --csv vehicles.csv --nrows 50000

Outputs (in ./artifacts):
    car_price_model.joblib   model + category mappings + UI metadata + residual intervals
    metrics.json             test-set metrics
    feature_importance.csv   feature importance (gain)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
RANDOM_STATE = 42
ARTIFACT_DIR = Path("artifacts")

DROPPED_COLUMNS = ["id", "url", "region_url", "image_url", "county"]

CATEGORICAL_COLS = [
    "region", "manufacturer", "model", "model_base", "condition", "fuel",
    "title_status", "transmission", "drive", "size", "type", "paint_color", "state",
]
NUMERIC_COLS = [
    "year", "odometer", "cylinders", "lat", "long",
    "posting_year", "posting_month", "posting_dayofweek",
    "car_age", "miles_per_year", "description_length", "has_vin",
]
FEATURE_COLS = CATEGORICAL_COLS + NUMERIC_COLS

INPUT_FIELDS = [
    "region", "year", "manufacturer", "model", "condition", "cylinders", "fuel",
    "odometer", "title_status", "transmission", "vin", "drive", "size", "type",
    "paint_color", "description", "state", "lat", "long", "posting_date",
]

FEATURE_LABELS = {
    "region": "Local Region",
    "manufacturer": "Manufacturer",
    "model": "Model / Trim",
    "model_base": "Base Model",
    "condition": "Condition",
    "fuel": "Fuel Type",
    "title_status": "Title Status",
    "transmission": "Transmission",
    "drive": "Drivetrain",
    "size": "Vehicle Size",
    "type": "Body Type",
    "paint_color": "Paint Color",
    "state": "State",
    "year": "Model Year",
    "odometer": "Odometer (Miles)",
    "cylinders": "Cylinders",
    "lat": "Latitude",
    "long": "Longitude",
    "posting_year": "Listing Year",
    "posting_month": "Listing Month",
    "posting_dayofweek": "Listing Day",
    "car_age": "Vehicle Age",
    "miles_per_year": "Miles per Year",
    "description_length": "Description Detail",
    "has_vin": "VIN Provided",
}

# Training-data filters
MIN_PRICE, MAX_PRICE = 500, 100_000
PLACEHOLDER_PRICES = {1234, 12345, 99999}
MIN_YEAR, MAX_YEAR = 1970, 2022
MAX_ODOMETER = 500_000
MIN_CATEGORY_COUNT = 30
MIN_MODEL_PER_MAKE_COUNT = 5


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------
def _clean_text(series: pd.Series) -> pd.Series:
    """Lower-case, strip and convert empty-like tokens to NaN."""
    s = series.astype(str).str.strip().str.lower()
    return s.replace({"nan": np.nan, "none": np.nan, "<na>": np.nan, "": np.nan})


def prepare_features(df: pd.DataFrame, default_posting_date=None) -> pd.DataFrame:
    """
    Convert raw dataset-style columns into the model feature table.
    Used consistently for both training and inference.
    """
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    for col in INPUT_FIELDS:
        if col not in df.columns:
            df[col] = np.nan

    out = pd.DataFrame(index=df.index)

    # Categorical columns (raw dataset fields)
    raw_cat_cols = [c for c in CATEGORICAL_COLS if c != "model_base"]
    for col in raw_cat_cols:
        out[col] = _clean_text(df[col])

    # Base model token (e.g., "f-150 xlt supercrew" -> "f-150")
    out["model_base"] = out["model"].str.split().str[0]

    # Numeric columns
    out["year"] = pd.to_numeric(df["year"], errors="coerce")
    out["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")
    out["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    out["long"] = pd.to_numeric(df["long"], errors="coerce")
    out["cylinders"] = pd.to_numeric(
        df["cylinders"].astype(str).str.extract(r"(\d+)")[0], errors="coerce"
    )

    # Posting date features
    posting = pd.to_datetime(df["posting_date"], errors="coerce", utc=True)
    if default_posting_date is not None:
        posting = posting.fillna(pd.Timestamp(default_posting_date))
    out["posting_year"] = posting.dt.year
    out["posting_month"] = posting.dt.month
    out["posting_dayofweek"] = posting.dt.dayofweek

    # Age & annual usage
    out["car_age"] = (out["posting_year"] - out["year"]).clip(lower=0)
    out["miles_per_year"] = out["odometer"] / np.maximum(out["car_age"], 1.0)

    # Description and VIN derived features
    desc = df["description"]
    out["description_length"] = desc.astype(str).str.len().where(desc.notna(), 0)
    vin = df["vin"]
    out["has_vin"] = (
        vin.notna() & (vin.astype(str).str.strip().str.len() >= 11)
    ).astype(int)

    return out[FEATURE_COLS]


def clean_training_data(df: pd.DataFrame) -> pd.DataFrame:
    """Row-level filtering and deduplication applied to training data only."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.drop(columns=[c for c in DROPPED_COLUMNS if c in df.columns])

    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["odometer"] = pd.to_numeric(df["odometer"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["price", "year"])
    df = df[df["price"].between(MIN_PRICE, MAX_PRICE)]
    df = df[~df["price"].isin(PLACEHOLDER_PRICES)]
    df = df[df["year"].between(MIN_YEAR, MAX_YEAR)]
    df = df[df["odometer"].isna() | df["odometer"].between(0, MAX_ODOMETER)]

    # Treat suspicious 0-10 mile entries on older vehicles as missing
    suspicious_odo = (df["odometer"] <= 10) & (df["year"] < 2021)
    df.loc[suspicious_odo, "odometer"] = np.nan

    # Drop exact duplicates
    df = df.drop_duplicates()

    # Deduplicate cross-posted dealer listings sharing the same valid VIN
    if "vin" in df.columns:
        if "posting_date" in df.columns:
            df = df.sort_values("posting_date", ascending=False, na_position="last")
        vin_clean = _clean_text(df["vin"])
        has_valid_vin = vin_clean.notna() & (vin_clean.str.len() >= 11)
        dup_vin = has_valid_vin & vin_clean.duplicated(keep="first")
        df = df[~dup_vin]

    print(f"Rows after cleaning & deduplication: {len(df):,} (removed {before - len(df):,})")
    return df.reset_index(drop=True)


def fit_categories(X: pd.DataFrame) -> dict[str, list[str]]:
    """Learn allowed categories (frequency threshold) from the training split."""
    categories = {}
    for col in CATEGORICAL_COLS:
        counts = X[col].value_counts()
        categories[col] = sorted(counts[counts >= MIN_CATEGORY_COUNT].index.tolist())
    return categories


def apply_categories(X: pd.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Cast categorical columns using fixed categories. Unseen values become NaN."""
    X = X.copy()
    for col in CATEGORICAL_COLS:
        known = X[col].where(X[col].isin(categories[col]))
        X[col] = pd.Categorical(known, categories=categories[col])
    return X


def fit_ui_metadata(X_train_raw: pd.DataFrame, categories: dict[str, list[str]]) -> dict:
    """Extract cascading dropdown mappings and default coordinates from training data."""
    # Manufacturer -> known models
    known_models = set(categories["model"])
    mm_counts = (
        X_train_raw.dropna(subset=["manufacturer", "model"])
        .groupby(["manufacturer", "model"])
        .size()
    )
    manufacturer_models: dict[str, list[str]] = {}
    for (make, mdl), cnt in mm_counts.items():
        if make in categories["manufacturer"] and mdl in known_models and cnt >= MIN_MODEL_PER_MAKE_COUNT:
            manufacturer_models.setdefault(make, []).append(mdl)
    manufacturer_models = {k: sorted(v) for k, v in manufacturer_models.items()}

    # State -> known regions
    known_regions = set(categories["region"])
    sr_pairs = (
        X_train_raw.dropna(subset=["state", "region"])[["state", "region"]]
        .drop_duplicates()
    )
    state_regions: dict[str, list[str]] = {}
    for state, reg in sr_pairs.itertuples(index=False):
        if state in categories["state"] and reg in known_regions:
            state_regions.setdefault(state, []).append(reg)
    state_regions = {k: sorted(v) for k, v in state_regions.items()}

    # Median coordinates per region and state (used to auto-fill lat/long in UI)
    region_coords = {
        reg: (float(row["lat"]), float(row["long"]))
        for reg, row in X_train_raw.dropna(subset=["region", "lat", "long"])
        .groupby("region")[["lat", "long"]]
        .median()
        .iterrows()
        if reg in known_regions
    }
    state_coords = {
        st_code: (float(row["lat"]), float(row["long"]))
        for st_code, row in X_train_raw.dropna(subset=["state", "lat", "long"])
        .groupby("state")[["lat", "long"]]
        .median()
        .iterrows()
        if st_code in categories["state"]
    }

    return {
        "manufacturer_models": manufacturer_models,
        "state_regions": state_regions,
        "region_coords": region_coords,
        "state_coords": state_coords,
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train(csv_path: str, nrows: int | None = None) -> dict:
    ARTIFACT_DIR.mkdir(exist_ok=True)

    print(f"Loading {csv_path} ...")
    raw = pd.read_csv(csv_path, nrows=nrows, low_memory=False)
    print(f"Raw shape: {raw.shape}")

    df = clean_training_data(raw)
    y = np.log1p(df["price"].values)

    default_posting_date = pd.to_datetime(
        df["posting_date"], errors="coerce", utc=True
    ).median()
    if pd.isna(default_posting_date):
        default_posting_date = pd.Timestamp("2021-05-01", tz="UTC")

    X_raw = prepare_features(
        df.drop(columns=["price"]), default_posting_date=default_posting_date
    )

    X_train_full, X_test_raw, y_train_full, y_test = train_test_split(
        X_raw, y, test_size=0.20, random_state=RANDOM_STATE
    )
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.10, random_state=RANDOM_STATE
    )

    categories = fit_categories(X_train_raw)
    ui_metadata = fit_ui_metadata(X_train_raw, categories)

    X_train = apply_categories(X_train_raw, categories)
    X_val = apply_categories(X_val_raw, categories)
    X_test = apply_categories(X_test_raw, categories)

    model = lgb.LGBMRegressor(
        objective="regression",
        n_estimators=5000,
        learning_rate=0.05,
        num_leaves=127,
        min_child_samples=20,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        cat_smooth=20,
        min_data_per_group=50,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="l2",
        categorical_feature=CATEGORICAL_COLS,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )

    # Calibrated 70% prediction interval from out-of-sample validation residuals
    val_log_pred = model.predict(X_val)
    val_residuals = y_val - val_log_pred
    q_low, q_high = np.quantile(val_residuals, [0.15, 0.85])

    # Evaluation on the untouched test split (original dollar scale)
    pred = np.clip(np.expm1(model.predict(X_test)), MIN_PRICE, MAX_PRICE)
    actual = np.expm1(y_test)
    metrics = {
        "MAE": float(mean_absolute_error(actual, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, pred))),
        "R2": float(r2_score(actual, pred)),
        "R2_log_price": float(r2_score(y_test, np.log1p(pred))),
        "MedianAbsError": float(np.median(np.abs(actual - pred))),
        "MAPE_percent": float(np.mean(np.abs(actual - pred) / actual) * 100),
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
    }
    print("\nTest metrics:")
    for k, v in metrics.items():
        print(f"  {k:>16}: {v:,.4f}" if isinstance(v, float) else f"  {k:>16}: {v:,}")

    # Feature importance
    importance = (
        pd.DataFrame({
            "feature": FEATURE_COLS,
            "label": [FEATURE_LABELS.get(c, c) for c in FEATURE_COLS],
            "gain": model.booster_.feature_importance(importance_type="gain"),
        })
        .sort_values("gain", ascending=False)
        .reset_index(drop=True)
    )
    importance["gain_percent"] = 100.0 * importance["gain"] / importance["gain"].sum()
    importance.to_csv(ARTIFACT_DIR / "feature_importance.csv", index=False)
    print("\nTop 10 features:")
    print(importance[["feature", "gain_percent"]].head(10).to_string(index=False))

    # Save bundle
    artifact = {
        "model": model,
        "categories": categories,
        "feature_cols": FEATURE_COLS,
        "feature_labels": FEATURE_LABELS,
        "default_posting_date": str(default_posting_date),
        "metrics": metrics,
        "price_range": (MIN_PRICE, MAX_PRICE),
        "residual_quantiles": (float(q_low), float(q_high)),
        "ui_metadata": ui_metadata,
        "feature_importance": importance.to_dict(orient="records"),
    }
    joblib.dump(artifact, ARTIFACT_DIR / "car_price_model.joblib", compress=3)
    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved artifacts to {ARTIFACT_DIR.resolve()}")
    return artifact


# --------------------------------------------------------------------------
# Prediction & Explainability
# --------------------------------------------------------------------------
def load_artifact(path: str | Path = "artifacts/car_price_model.joblib") -> dict:
    return joblib.load(path)


def _enrich_coordinates(df: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Fill missing lat/long using median coordinates for the region or state."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    meta = artifact.get("ui_metadata", {})
    region_coords = meta.get("region_coords", {})
    state_coords = meta.get("state_coords", {})

    if "lat" not in df.columns:
        df["lat"] = np.nan
    if "long" not in df.columns:
        df["long"] = np.nan

    for idx, row in df.iterrows():
        if pd.isna(row["lat"]) or pd.isna(row["long"]):
            reg = str(row.get("region", "")).strip().lower()
            st_code = str(row.get("state", "")).strip().lower()
            coords = region_coords.get(reg) or state_coords.get(st_code)
            if coords is not None:
                df.at[idx, "lat"] = coords[0]
                df.at[idx, "long"] = coords[1]
    return df


def predict_price(inputs: dict | list[dict], artifact: dict) -> float | list[float]:
    """
    Predict the price for one record (dict) or many records (list of dicts).
    Backwards-compatible with the original API.
    """
    single = isinstance(inputs, dict)
    df = pd.DataFrame([inputs] if single else inputs)
    df = _enrich_coordinates(df, artifact)
    X = prepare_features(df, default_posting_date=artifact["default_posting_date"])
    X = apply_categories(X, artifact["categories"])
    prices = np.expm1(artifact["model"].predict(X))
    low, high = artifact["price_range"]
    prices = np.clip(prices, low, high)
    return float(prices[0]) if single else prices.tolist()


def predict_price_with_details(record: dict, artifact: dict) -> dict:
    """
    Predict price, fair market range, and per-feature TreeSHAP explanations
    for a single car record.
    """
    df = _enrich_coordinates(pd.DataFrame([record]), artifact)
    X = prepare_features(df, default_posting_date=artifact["default_posting_date"])
    X = apply_categories(X, artifact["categories"])

    model = artifact["model"]
    low_cap, high_cap = artifact["price_range"]
    q_low, q_high = artifact.get("residual_quantiles", (-0.15, 0.15))

    pred_log = float(model.predict(X)[0])
    price = float(np.clip(np.expm1(pred_log), low_cap, high_cap))
    low_price = float(np.clip(np.expm1(pred_log + q_low), low_cap, high_cap))
    high_price = float(np.clip(np.expm1(pred_log + q_high), low_cap, high_cap))

    # Native LightGBM TreeSHAP contributions in log space
    contribs = model.predict(X, pred_contrib=True)[0]
    base_log = float(contribs[-1])
    base_price = float(np.expm1(base_log))
    feat_contribs = contribs[:-1]

    total_log_delta = pred_log - base_log
    total_dollar_delta = price - base_price
    labels = artifact.get("feature_labels", FEATURE_LABELS)

    breakdown = []
    for feat, c in zip(artifact["feature_cols"], feat_contribs):
        c = float(c)
        pct_impact = float(np.expm1(c) * 100.0)
        if abs(total_log_delta) > 1e-9:
            dollar_impact = float((c / total_log_delta) * total_dollar_delta)
        else:
            dollar_impact = 0.0
        breakdown.append({
            "feature": feat,
            "label": labels.get(feat, feat),
            "log_impact": c,
            "pct_impact": pct_impact,
            "dollar_impact": dollar_impact,
        })

    breakdown.sort(key=lambda d: abs(d["dollar_impact"]), reverse=True)

    return {
        "price": price,
        "low_price": min(low_price, price),
        "high_price": max(high_price, price),
        "base_price": base_price,
        "contributions": breakdown,
    }


def predict_batch_with_ranges(df_inputs: pd.DataFrame, artifact: dict) -> pd.DataFrame:
    """Predict point estimates and fair market ranges for a DataFrame of cars."""
    df_enriched = _enrich_coordinates(df_inputs, artifact)
    X = prepare_features(df_enriched, default_posting_date=artifact["default_posting_date"])
    X = apply_categories(X, artifact["categories"])

    low_cap, high_cap = artifact["price_range"]
    q_low, q_high = artifact.get("residual_quantiles", (-0.15, 0.15))

    pred_log = artifact["model"].predict(X)
    out = df_inputs.copy()
    out["predicted_price"] = np.round(np.clip(np.expm1(pred_log), low_cap, high_cap), 0)
    out["fair_range_low"] = np.round(np.clip(np.expm1(pred_log + q_low), low_cap, high_cap), 0)
    out["fair_range_high"] = np.round(np.clip(np.expm1(pred_log + q_high), low_cap, high_cap), 0)
    return out


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Craigslist used car price model")
    parser.add_argument("--csv", default="vehicles.csv", help="path to vehicles.csv")
    parser.add_argument("--nrows", type=int, default=None, help="read only the first N rows")
    args = parser.parse_args()

    art = train(args.csv, args.nrows)

    sample = {
        "year": 2018,
        "manufacturer": "toyota",
        "model": "camry",
        "condition": "good",
        "cylinders": "4 cylinders",
        "fuel": "gas",
        "odometer": 65000,
        "title_status": "clean",
        "transmission": "automatic",
        "drive": "fwd",
        "type": "sedan",
        "paint_color": "white",
        "state": "ca",
    }
    res = predict_price_with_details(sample, art)
    print(
        f"\nSample prediction: ${res['price']:,.0f} "
        f"(Fair range: ${res['low_price']:,.0f} - ${res['high_price']:,.0f})"
    )