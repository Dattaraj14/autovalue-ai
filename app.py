from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from car_price_model import (
    load_artifact,
    predict_batch_with_ranges,
    predict_price,
    predict_price_with_details,
)

ARTIFACT_PATH = Path("artifacts/car_price_model.joblib")

st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Subtle theme-safe styling for light & dark mode
st.markdown(
    """
    <style>
    .price-card {
        padding: 1.5rem 1.75rem;
        border-radius: 14px;
        background: linear-gradient(135deg, rgba(37, 99, 235, 0.14), rgba(16, 185, 129, 0.10));
        border: 1px solid rgba(37, 99, 235, 0.28);
        margin-bottom: 1rem;
    }
    .price-label {
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        opacity: 0.78;
        margin-bottom: 0.25rem;
    }
    .price-value {
        font-size: 2.8rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0.15rem 0 0.4rem 0;
    }
    .price-range {
        font-size: 1rem;
        opacity: 0.9;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_artifact(path_str: str) -> dict:
    return load_artifact(path_str)


def _format_option(val: str) -> str:
    if val == "":
        return "— Not specified —"
    if len(val) == 2 and val.isalpha():
        return val.upper()
    return val.replace("-", " ").title()


def _safe_index(options: list[str], value: str) -> int:
    try:
        return options.index(value)
    except ValueError:
        return 0


# --------------------------------------------------------------------------
# Guard: Model artifact must exist
# --------------------------------------------------------------------------
if not ARTIFACT_PATH.exists():
    st.title("🚗 Used Car Price Predictor")
    st.warning(
        "No trained model found at `artifacts/car_price_model.joblib` yet."
    )
    st.markdown(
        """
        Train the model first from your terminal in this project folder:

        ```bash
        # Quick test (~20 seconds)
        python car_price_model.py --csv vehicles.csv --nrows 50000

        # Full dataset (~2–5 minutes)
        python car_price_model.py --csv vehicles.csv
        ```
        Then refresh this browser page.
        """
    )
    st.stop()

artifact = get_artifact(str(ARTIFACT_PATH))
categories = artifact["categories"]
meta = artifact.get("ui_metadata", {})
metrics = artifact.get("metrics", {})

# --------------------------------------------------------------------------
# Presets
# --------------------------------------------------------------------------
PRESETS = {
    "2018 Toyota Camry": {
        "manufacturer": "toyota",
        "model": "camry",
        "year": 2018,
        "odometer": 62000,
        "condition": "good",
        "fuel": "gas",
        "transmission": "automatic",
        "drive": "fwd",
        "type": "sedan",
        "cylinders": 4,
        "title_status": "clean",
        "paint_color": "white",
        "state": "ca",
    },
    "2019 Ford F-150": {
        "manufacturer": "ford",
        "model": "f-150",
        "year": 2019,
        "odometer": 54000,
        "condition": "excellent",
        "fuel": "gas",
        "transmission": "automatic",
        "drive": "4wd",
        "type": "pickup",
        "cylinders": 6,
        "title_status": "clean",
        "paint_color": "black",
        "state": "tx",
    },
    "2015 Honda Civic": {
        "manufacturer": "honda",
        "model": "civic",
        "year": 2015,
        "odometer": 95000,
        "condition": "good",
        "fuel": "gas",
        "transmission": "automatic",
        "drive": "fwd",
        "type": "sedan",
        "cylinders": 4,
        "title_status": "clean",
        "paint_color": "silver",
        "state": "fl",
    },
    "2020 Jeep Wrangler": {
        "manufacturer": "jeep",
        "model": "wrangler",
        "year": 2020,
        "odometer": 32000,
        "condition": "like new",
        "fuel": "gas",
        "transmission": "automatic",
        "drive": "4wd",
        "type": "suv",
        "cylinders": 6,
        "title_status": "clean",
        "paint_color": "grey",
        "state": "co",
    },
}

if "selected_preset" not in st.session_state:
    st.session_state["selected_preset"] = PRESETS["2018 Toyota Camry"]

# --------------------------------------------------------------------------
# Sidebar
# --------------------------------------------------------------------------
with st.sidebar:
    st.header("🚗 Car Price AI")
    st.caption("Powered by LightGBM on Craigslist Used Car Listings")

    st.subheader("Quick Presets")
    for preset_name, preset_vals in PRESETS.items():
        if st.button(preset_name, width="stretch"):
            st.session_state["selected_preset"] = preset_vals
            st.rerun()

    st.divider()
    st.subheader("Model Summary")
    if metrics:
        c1, c2 = st.columns(2)
        c1.metric("Test R²", f"{metrics.get('R2', 0):.3f}")
        c2.metric("Median Error", f"${metrics.get('MedianAbsError', 0):,.0f}")
        st.caption(
            f"MAE: **${metrics.get('MAE', 0):,.0f}** | "
            f"Trained on **{metrics.get('train_rows', 0):,}** listings"
        )

# --------------------------------------------------------------------------
# Header & Tabs
# --------------------------------------------------------------------------
st.title("Used Car Price Estimator")
tab_single, tab_batch, tab_insights = st.tabs(
    ["🚗 Single Car Estimator", "📂 Batch CSV Valuation", "📊 Model Insights"]
)

# ==========================================================================
# TAB 1: SINGLE CAR ESTIMATOR
# ==========================================================================
with tab_single:
    defaults = st.session_state["selected_preset"]
    col_inputs, col_results = st.columns([1.05, 1.15], gap="large")

    with col_inputs:
        st.subheader("Vehicle Details")

        # Manufacturer & cascading Model
        make_options = [""] + categories.get("manufacturer", [])
        r1c1, r1c2 = st.columns(2)
        with r1c1:
            manufacturer = st.selectbox(
                "Manufacturer",
                options=make_options,
                index=_safe_index(make_options, defaults.get("manufacturer", "")),
                format_func=_format_option,
            )

        mm_map = meta.get("manufacturer_models", {})
        if manufacturer and manufacturer in mm_map:
            model_options = [""] + mm_map[manufacturer] + ["__custom__"]
        else:
            model_options = [""] + categories.get("model", [])[:300] + ["__custom__"]

        default_model = defaults.get("model", "")
        with r1c2:
            if default_model and default_model not in model_options:
                model_choice = st.selectbox(
                    "Model",
                    options=model_options,
                    index=len(model_options) - 1,
                    format_func=lambda x: "Other / Type custom..." if x == "__custom__" else _format_option(x),
                )
            else:
                model_choice = st.selectbox(
                    "Model",
                    options=model_options,
                    index=_safe_index(model_options, default_model),
                    format_func=lambda x: "Other / Type custom..." if x == "__custom__" else _format_option(x),
                )

        if model_choice == "__custom__":
            model_val = st.text_input(
                "Enter custom model / trim",
                value=default_model if default_model not in model_options else "",
                placeholder="e.g., camry se, civic lx, accord sport",
            )
        else:
            model_val = model_choice

        # Year & Odometer
        r2c1, r2c2 = st.columns(2)
        with r2c1:
            year = st.slider(
                "Model Year",
                min_value=1970,
                max_value=2022,
                value=int(defaults.get("year", 2018)),
            )
        with r2c2:
            odometer = st.number_input(
                "Odometer (miles)",
                min_value=0,
                max_value=500_000,
                value=int(defaults.get("odometer", 65000)),
                step=5000,
            )

        # Core Specs
        r3c1, r3c2, r3c3 = st.columns(3)
        cond_opts = [""] + categories.get("condition", [])
        fuel_opts = [""] + categories.get("fuel", [])
        trans_opts = [""] + categories.get("transmission", [])
        with r3c1:
            condition = st.selectbox(
                "Condition",
                options=cond_opts,
                index=_safe_index(cond_opts, defaults.get("condition", "")),
                format_func=_format_option,
            )
        with r3c2:
            fuel = st.selectbox(
                "Fuel Type",
                options=fuel_opts,
                index=_safe_index(fuel_opts, defaults.get("fuel", "")),
                format_func=_format_option,
            )
        with r3c3:
            transmission = st.selectbox(
                "Transmission",
                options=trans_opts,
                index=_safe_index(trans_opts, defaults.get("transmission", "")),
                format_func=_format_option,
            )

        r4c1, r4c2, r4c3 = st.columns(3)
        drive_opts = [""] + categories.get("drive", [])
        type_opts = [""] + categories.get("type", [])
        title_opts = [""] + categories.get("title_status", [])
        with r4c1:
            drive = st.selectbox(
                "Drivetrain",
                options=drive_opts,
                index=_safe_index(drive_opts, defaults.get("drive", "")),
                format_func=_format_option,
            )
        with r4c2:
            body_type = st.selectbox(
                "Body Type",
                options=type_opts,
                index=_safe_index(type_opts, defaults.get("type", "")),
                format_func=_format_option,
            )
        with r4c3:
            title_status = st.selectbox(
                "Title Status",
                options=title_opts,
                index=_safe_index(title_opts, defaults.get("title_status", "clean")),
                format_func=_format_option,
            )

        r5c1, r5c2, r5c3 = st.columns(3)
        cyl_opts = [None, 3, 4, 5, 6, 8, 10, 12]
        color_opts = [""] + categories.get("paint_color", [])
        size_opts = [""] + categories.get("size", [])
        with r5c1:
            default_cyl = defaults.get("cylinders", 4)
            cyl_idx = cyl_opts.index(default_cyl) if default_cyl in cyl_opts else 0
            cylinders = st.selectbox(
                "Cylinders",
                options=cyl_opts,
                index=cyl_idx,
                format_func=lambda x: "— Not specified —" if x is None else f"{x} Cylinders",
            )
        with r5c2:
            paint_color = st.selectbox(
                "Paint Color",
                options=color_opts,
                index=_safe_index(color_opts, defaults.get("paint_color", "")),
                format_func=_format_option,
            )
        with r5c3:
            size = st.selectbox(
                "Vehicle Size",
                options=size_opts,
                index=0,
                format_func=_format_option,
            )

        # Optional Location & Listing Details
        with st.expander("Location & Listing Details (Optional)", expanded=False):
            loc1, loc2 = st.columns(2)
            state_opts = [""] + categories.get("state", [])
            with loc1:
                state = st.selectbox(
                    "State",
                    options=state_opts,
                    index=_safe_index(state_opts, defaults.get("state", "")),
                    format_func=_format_option,
                )
            sr_map = meta.get("state_regions", {})
            reg_opts = [""] + (sr_map.get(state, []) if state else categories.get("region", []))
            with loc2:
                region = st.selectbox(
                    "Craigslist Region",
                    options=reg_opts,
                    index=0,
                    format_func=_format_option,
                )

            has_vin_toggle = st.checkbox("Listing includes a valid VIN", value=True)
            description = st.text_area(
                "Listing Description (optional)",
                value="",
                placeholder="Paste or type a listing description (detail level affects perceived quality)...",
                height=80,
            )

    # Assemble input payload
    car_input = {
        "manufacturer": manufacturer or None,
        "model": model_val or None,
        "year": year,
        "odometer": odometer,
        "condition": condition or None,
        "fuel": fuel or None,
        "transmission": transmission or None,
        "drive": drive or None,
        "type": body_type or None,
        "title_status": title_status or None,
        "cylinders": f"{cylinders} cylinders" if cylinders is not None else None,
        "paint_color": paint_color or None,
        "size": size or None,
        "state": state or None,
        "region": region or None,
        "vin": "1HGCM82633A004352" if has_vin_toggle else None,
        "description": description if description.strip() else None,
    }

    result = predict_price_with_details(car_input, artifact)

    # ------------------------------------------------------------------
    # Results Panel
    # ------------------------------------------------------------------
    with col_results:
        st.subheader("Estimated Valuation")

        title_str = " ".join(
            p for p in [str(year), _format_option(manufacturer) if manufacturer else "", _format_option(model_val) if model_val else ""]
            if p
        ).strip() or "Selected Vehicle"

        st.markdown(
            f"""
            <div class="price-card">
                <div class="price-label">{title_str}</div>
                <div class="price-value">${result['price']:,.0f}</div>
                <div class="price-range">
                    Fair Market Range (70% interval):
                    <strong>${result['low_price']:,.0f} – ${result['high_price']:,.0f}</strong>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Quick summary metrics
        m1, m2, m3 = st.columns(3)
        est_age = max(2021 - year, 1)
        m1.metric("Market Baseline", f"${result['base_price']:,.0f}")
        m2.metric("Net Adjustment", f"${result['price'] - result['base_price']:+,.0f}")
        m3.metric("Avg. Miles / Year", f"{odometer / est_age:,.0f}")

        # Price Breakdown Chart (TreeSHAP)
        st.markdown("#### Why this price? (Top Price Drivers)")
        top_contrib = pd.DataFrame(result["contributions"][:10])
        top_contrib["direction"] = np.where(
            top_contrib["dollar_impact"] >= 0, "Increases Price", "Decreases Price"
        )

        bar_chart = (
            alt.Chart(top_contrib)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("dollar_impact:Q", title="Estimated Dollar Impact ($)"),
                y=alt.Y("label:N", sort="-x", title=None),
                color=alt.Color(
                    "direction:N",
                    scale=alt.Scale(
                        domain=["Increases Price", "Decreases Price"],
                        range=["#16a34a", "#dc2626"],
                    ),
                    legend=alt.Legend(orient="top", title=None),
                ),
                tooltip=[
                    alt.Tooltip("label:N", title="Feature"),
                    alt.Tooltip("dollar_impact:Q", title="Dollar Impact", format="+$,.0f"),
                    alt.Tooltip("pct_impact:Q", title="Relative Impact", format="+.1f"),
                ],
            )
            .properties(height=280)
        )
        st.altair_chart(bar_chart, width="stretch")

        # Mileage Depreciation Curve
        st.markdown("#### Mileage Impact Simulator")
        mile_grid = sorted(
            set(list(range(10_000, 260_000, 20_000)) + [int(odometer)])
        )
        sim_records = [{**car_input, "odometer": m} for m in mile_grid]
        sim_prices = predict_price(sim_records, artifact)
        sim_df = pd.DataFrame({"odometer": mile_grid, "price": sim_prices})
        current_df = pd.DataFrame({"odometer": [int(odometer)], "price": [result["price"]]})

        line = (
            alt.Chart(sim_df)
            .mark_line(color="#2563eb", strokeWidth=2.5)
            .encode(
                x=alt.X("odometer:Q", title="Odometer (miles)", axis=alt.Axis(format=",")),
                y=alt.Y("price:Q", title="Predicted Price ($)", axis=alt.Axis(format="$,.0f")),
                tooltip=[
                    alt.Tooltip("odometer:Q", title="Miles", format=","),
                    alt.Tooltip("price:Q", title="Est. Price", format="$,.0f"),
                ],
            )
        )
        point = (
            alt.Chart(current_df)
            .mark_point(color="#ef4444", size=140, filled=True)
            .encode(
                x="odometer:Q",
                y="price:Q",
                tooltip=[
                    alt.Tooltip("odometer:Q", title="Your Car Miles", format=","),
                    alt.Tooltip("price:Q", title="Your Car Price", format="$,.0f"),
                ],
            )
        )
        st.altair_chart((line + point).properties(height=230), width="stretch")


# ==========================================================================
# TAB 2: BATCH CSV VALUATION
# ==========================================================================
with tab_batch:
    st.subheader("Bulk Price Prediction from CSV")
    st.write(
        "Upload a CSV file containing any subset of the dataset columns "
        "(`year`, `manufacturer`, `model`, `odometer`, `condition`, `fuel`, `transmission`, `state`, etc.)."
    )

    sample_template = pd.DataFrame([
        {
            "year": 2018, "manufacturer": "toyota", "model": "camry",
            "odometer": 62000, "condition": "good", "fuel": "gas",
            "transmission": "automatic", "drive": "fwd", "state": "ca",
        },
        {
            "year": 2019, "manufacturer": "ford", "model": "f-150",
            "odometer": 48000, "condition": "excellent", "fuel": "gas",
            "transmission": "automatic", "drive": "4wd", "state": "tx",
        },
    ])
    st.download_button(
        "Download Sample CSV Template",
        data=sample_template.to_csv(index=False).encode("utf-8"),
        file_name="sample_cars.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded is not None:
        batch_df = pd.read_csv(uploaded)
        st.write(f"Loaded **{len(batch_df):,}** rows.")
        if st.button("Predict Prices for All Rows", type="primary"):
            with st.spinner("Running batch valuation..."):
                scored = predict_batch_with_ranges(batch_df, artifact)
            st.success("Predictions complete!")
            st.dataframe(scored, width="stretch")
            st.download_button(
                "Download Predictions CSV",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="car_price_predictions.csv",
                mime="text/csv",
            )


# ==========================================================================
# TAB 3: MODEL PERFORMANCE & FEATURE IMPORTANCE
# ==========================================================================
with tab_insights:
    st.subheader("Test-Set Performance")
    if metrics:
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("R² Score", f"{metrics.get('R2', 0):.4f}")
        k2.metric("MAE", f"${metrics.get('MAE', 0):,.0f}")
        k3.metric("Median Abs Error", f"${metrics.get('MedianAbsError', 0):,.0f}")
        k4.metric("MAPE", f"{metrics.get('MAPE_percent', 0):.1f}%")

        k5, k6, k7, k8 = st.columns(4)
        k5.metric("RMSE", f"${metrics.get('RMSE', 0):,.0f}")
        k6.metric("R² (Log Price)", f"{metrics.get('R2_log_price', 0):.4f}")
        k7.metric("Training Rows", f"{metrics.get('train_rows', 0):,}")
        k8.metric("Best Iteration", f"{metrics.get('best_iteration', 0):,}")

    st.divider()
    st.subheader("Global Feature Importance (Gain %)")
    fi_records = artifact.get("feature_importance")
    if fi_records:
        fi_df = pd.DataFrame(fi_records)
    elif Path("artifacts/feature_importance.csv").exists():
        fi_df = pd.read_csv("artifacts/feature_importance.csv")
    else:
        fi_df = pd.DataFrame()

    if not fi_df.empty:
        label_col = "label" if "label" in fi_df.columns else "feature"
        fi_chart = (
            alt.Chart(fi_df.head(15))
            .mark_bar(color="#2563eb", cornerRadiusEnd=4)
            .encode(
                x=alt.X("gain_percent:Q", title="Share of Total Gain (%)"),
                y=alt.Y(f"{label_col}:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip(f"{label_col}:N", title="Feature"),
                    alt.Tooltip("gain_percent:Q", title="Gain %", format=".2f"),
                ],
            )
            .properties(height=420)
        )
        st.altair_chart(fi_chart, width="stretch")

        with st.expander("View Full Feature Importance Table"):
            st.dataframe(fi_df, width="stretch", hide_index=True)