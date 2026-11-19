# 🚗 AutoValue AI — Used Car Price Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://autovalue-ai-hhttxxdmtvynzn4pd3usv6.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-2ea44f)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E?logo=scikitlearn&logoColor=white)

**AutoValue AI** is an end-to-end machine learning web application that estimates fair market values for used cars and trucks from real-world Craigslist listings. Powered by a log-target **LightGBM** regression pipeline and an interactive **Streamlit** dashboard, it goes beyond single-number estimates by providing calibrated **70% fair market price ranges**, per-car **TreeSHAP price explanations**, **mileage depreciation simulations**, and **bulk CSV valuation**.

---

## ✨ Key Features

### 1. Interactive Single-Car Estimator
* **Smart Cascading Dropdowns:** Selecting a **Manufacturer** dynamically filters the **Model** dropdown to valid makes/models (with a custom text input fallback for uncommon trims). Selecting a **State** dynamically filters local **Craigslist Regions** and auto-fills median latitude/longitude coordinates.
* **Quick-Fill Presets:** One-click sidebar presets (*2018 Toyota Camry*, *2019 Ford F-150*, *2015 Honda Civic*, *2020 Jeep Wrangler*) for instant exploration.
* **Calibrated Fair Market Range:** Displays both the point estimate and an out-of-sample **70% Fair Market Range** (derived from validation log-residual quantiles), along with the market baseline, net dollar adjustment, and annual mileage rate.
* **Explainable AI ("Why this price?"):** Native LightGBM **TreeSHAP** (`pred_contrib=True`) visualizes the top 10 features pushing a specific car's valuation up or down in dollars and percentages.
* **Mileage Impact Simulator:** Interactive depreciation curve showing how the selected car's predicted price shifts from 10,000 to 250,000 miles, highlighting the vehicle's current odometer position.

### 2. Batch CSV Valuation
* **Ready-to-Use Template:** Download a sample CSV directly from the UI.
* **Flexible Schema:** Upload any CSV containing any subset of the supported columns—omitted attributes are handled natively as missing values.
* **Instant Export:** Appends `predicted_price`, `fair_range_low`, and `fair_range_high` to every row and exports a downloadable `car_price_predictions.csv`.

### 3. Model Insights Dashboard
* Displays live test-set metrics ($R^2$, log-price $R^2$, MAE, Median Absolute Error, RMSE, MAPE) and global gain-based feature importance directly from the serialized training artifact.

---

## 📊 Model Performance

Evaluated on a strictly held-out test split (**35,552 listings**) after cleaning and VIN-level deduplication (**177,756** cleaned rows from **295,451** raw listings):

| Metric | Test Set Score | Description |
| :--- | :---: | :--- |
| **$R^2$ Score (Dollar Scale)** | **0.8037** | Explains ~80.4% of real-world used car price variance |
| **$R^2$ Score (Log Price)** | **0.8269** | High proportional accuracy across budget and luxury segments |
| **Median Absolute Error** | **$1,315** | 50% of all predictions fall within $1,315 of the listing price |
| **Mean Absolute Error (MAE)** | **$2,507** | Average dollar error across the $500–$100,000 market range |
| **RMSE** | **$4,852** | Root mean squared error on the dollar scale |
| **MAPE** | **25.93%** | Mean absolute percentage error |
| **Best Iteration** | **3,912** | Early stopping on validation L2 loss (`14,221` val rows) |
| **Model Artifact Size** | **21.07 MB** | Compressed `joblib` bundle (model + metadata + quantiles) |

### Top 10 Most Important Features (LightGBM Gain %)

| Rank | Feature | Gain Share (%) | Type |
| :---: | :--- | :---: | :--- |
| 1 | `year` | 21.44% | Numeric |
| 2 | `model_base` | 19.73% | Engineered Categorical |
| 3 | `odometer` | 13.02% | Numeric |
| 4 | `region` | 5.49% | Categorical |
| 5 | `description_length` | 5.46% | Engineered Numeric |
| 6 | `condition` | 4.52% | Categorical |
| 7 | `type` | 4.13% | Categorical |
| 8 | `model` | 3.61% | Categorical |
| 9 | `miles_per_year` | 3.55% | Engineered Numeric |
| 10 | `car_age` | 3.23% | Engineered Numeric |

---

## 🧠 ML Pipeline & Feature Engineering

### Data Cleaning & Anti-Leakage Controls
1. **Range & Outlier Filtering:** Retains listings priced between **$500 and $100,000**, model years **1970–2022**, and odometer readings **$\le$ 500,000 miles**.
2. **Craigslist Noise Removal:** Strips common placeholder prices (`$1,234`, `$12,345`, `$99,999`) and converts suspicious `0–10` mile odometer readings on pre-2021 vehicles to `NaN`.
3. **VIN Deduplication:** Drops exact duplicates and deduplicates cross-posted dealer listings sharing the same valid ($\ge 11$-char) VIN—keeping only the latest listing—to prevent train/test data leakage.
4. **Target Transformation:** Predicts $\log(1 + \text{price})$ to stabilize the heavy right skew of vehicle prices, transforming predictions back via $\exp(\hat{y}) - 1$.
5. **Rare Category Consolidation:** Categories appearing fewer than **30 times** in the training split are treated as unknown (`NaN`), handled natively by LightGBM.

### Model Inputs (25 Features)
* **Categorical (13):** `manufacturer`, `model`, `model_base` *(first token extracted from `model`, e.g., `"f-150 xlt supercrew"` $\rightarrow$ `"f-150"`)*, `condition`, `fuel`, `transmission`, `drive`, `type`, `size`, `title_status`, `paint_color`, `state`, `region`.
* **Numeric & Engineered (12):** `year`, `odometer`, `cylinders` *(extracted via regex from strings like `"8 cylinders"`)*, `lat`, `long`, `posting_year`, `posting_month`, `posting_dayofweek`, `car_age` *(`posting_year - year`, clipped at $\ge 0$)*, `miles_per_year` *(`odometer / max(car_age, 1)`)*, `description_length`, `has_vin`.
* **Dropped Identifiers:** `id`, `url`, `region_url`, `image_url`, and the empty `county` column.

---

## 📁 Project Structure

```text
AutoValue-AI/
├── artifacts/
│   ├── car_price_model.joblib   # Trained LightGBM model, vocabularies, UI metadata & quantiles
│   ├── metrics.json             # Held-out test evaluation metrics
│   └── feature_importance.csv   # Gain-based global feature importance
├── app.py                       # Interactive Streamlit web application
├── car_price_model.py           # Data cleaning, training, inference & TreeSHAP pipeline
├── requirements.txt             # Python dependencies
├── packages.txt                 # System-level Linux dependency (libgomp1 for LightGBM)
├── .gitignore                   # Excludes raw dataset (vehicles.csv) and virtual env
└── README.md                    # Project documentation
```

---

## 🚀 Getting Started Locally

### 1. Clone the Repository
```bash
git clone [https://github.com/](https://github.com/)Dattaraj14/autovalue-ai.git
cd autovalue-ai
```

### 2. Create a Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows PowerShell)
.\.venv\Scripts\Activate.ps1

# Activate (macOS / Linux)
source .venv/bin/activate

# Install Python packages
pip install -r requirements.txt
```

### 3. Launch the Streamlit App
The pre-trained model artifact (`artifacts/car_price_model.joblib`) is included in the repository, so you can launch the UI immediately:
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## 🔄 Retraining the Model from Scratch (Optional)

1. Download **`vehicles.csv`** from the [Kaggle Craigslist Cars & Trucks Dataset](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) and place it in the project root directory.
2. Run the training script:
```bash
# Quick test run on 50,000 rows (~20 seconds)
python car_price_model.py --csv vehicles.csv --nrows 50000

# Full dataset training (~2–5 minutes)
python car_price_model.py --csv vehicles.csv
```
All updated artifacts (`car_price_model.joblib`, `metrics.json`, and `feature_importance.csv`) are saved automatically to `./artifacts/`.

---

## 💻 Using the Model in Python

You can import `car_price_model.py` directly into any Python backend or notebook:

```python
from car_price_model import load_artifact, predict_price, predict_price_with_details

artifact = load_artifact("artifacts/car_price_model.joblib")

sample_car = {
    "year": 2018,
    "manufacturer": "toyota",
    "model": "camry",
    "odometer": 65000,
    "condition": "good",
    "fuel": "gas",
    "transmission": "automatic",
    "drive": "fwd",
    "state": "ca",
}

# Simple point prediction
price = predict_price(sample_car, artifact)
print(f"Estimated Price: ${price:,.0f}")

# Detailed prediction with 70% range & TreeSHAP breakdown
details = predict_price_with_details(sample_car, artifact)
print(
    f"Estimated: ${details['price']:,.0f} "
    f"(Fair Range: ${details['low_price']:,.0f} - ${details['high_price']:,.0f})"
)
```

---

## ☁️ Deploying to Streamlit Community Cloud

1. Push this repository (including `artifacts/car_price_model.joblib`, `requirements.txt`, and `packages.txt`) to GitHub.
2. Visit [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** $\rightarrow$ select `<YOUR-USERNAME>/autovalue-ai`, branch `main`, and main file path `app.py`.
4. Under **Advanced settings**, match the Python version used during training, then click **Deploy**.

> **Note:** `packages.txt` contains `libgomp1`, which is required for LightGBM to run on Streamlit Community Cloud's Linux containers.

---

## 🛠️ Tech Stack

* **Machine Learning:** LightGBM, Scikit-learn, NumPy, Pandas, Joblib
* **Explainability & Uncertainty:** LightGBM Native TreeSHAP, Validation Residual Quantiles
* **Frontend & Visualization:** Streamlit, Altair
* **Dataset:** [Craigslist Cars & Trucks Data](https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data) by Austin Reese (Kaggle)

---

## 🔮 Future Improvements

* Incorporate text embeddings / TF-IDF keywords from vehicle `description` (e.g., `"leather"`, `"sunroof"`, `"one owner"`, `"needs transmission"`).
* Add inflation/market-year adjustment factors to calibrate historical Craigslist prices to current-year market conditions.
* Expose a standalone FastAPI microservice alongside the Streamlit dashboard for third-party integrations.

---

## 👤 Author

**Dattaraj** — Built as an end-to-end applied machine learning & deployment project.
