# 🌾 Crop Yield Predictor

A machine learning web application that predicts crop yield (tonnes/hectare) using four regression models — built with Python, scikit-learn, and Streamlit.

**Author:** Smrutishree Misra

---

 [👉 Click here for Live Demo](https://cropyieldpredictionforcrops.streamlit.app/)


## 📌 Project Overview

This project walks through the complete ML pipeline — from raw agricultural data to a deployed interactive web app:

- **Exploratory Data Analysis (EDA)** — distributions, outliers, correlations
- **Feature Engineering** — ratio features, log transforms, label encoding
- **Model Training** — Linear Regression, Lasso, Ridge, Random Forest
- **Streamlit App** — interactive UI to predict yield from user inputs

---

## 📊 Dataset

- **Source:** Indian crop yield dataset (1997–2020)
- **Rows:** 19,689 → 19,517 after cleaning
- **Features:** Crop, State, Season, Year, Area, Production, Rainfall, Fertilizer, Pesticide
- **Target:** `Yield` (tonnes/hectare)

---

## 🔍 EDA Highlights

| Finding | Detail |
|---|---|
| Target skewness | 12.79 (extremely right-skewed) |
| Coconut removed | Different unit of measurement (nuts, not kg) |
| After log transform | Skewness reduced to 1.94 |
| No missing values | Clean dataset ✅ |
| Outliers (IQR) | ~15.6% of rows |

---

## ⚙️ Feature Engineering

Raw numeric features had near-zero correlation with yield. New features created:

| Feature | Formula | Correlation with log(Yield) |
|---|---|---|
| `yield_per_area` | Production / (Area + 1) | **0.729** 🔥 |
| `log_production` | log(Production + 1) | 0.423 |
| `production_per_rain` | Production / (Rainfall + 1) | 0.255 |
| `log_area` | log(Area + 1) | 0.053 |

**Before feature engineering:** best correlation = 0.28
**After feature engineering:** best correlation = **0.73**

---

## 🤖 Model Results

| Model | R² Score | RMSE |
|---|---|---|
| Linear Regression | 0.9303 | 0.2389 |
| Lasso (scaled) | 0.9294 | 0.2406 |
| Ridge | 0.9303 | 0.2390 |
| **Random Forest** | **0.9904** | **0.0889** |

> All models trained on `log(Yield + 1)` and predictions inverse-transformed with `expm1`.

---

## 🗂️ Project Structure

```
crop_yield_prediction/
│
├── crop_yield.csv          # Raw dataset
│
├── notebooks/
│   └── 01_EDA.ipynb        # EDA + Feature Engineering + Model Training
│
├── app.py                  # Streamlit web application
├── requirements.txt        # Python dependencies
│
├── lr_model.pkl            # Linear Regression model
├── lasso_model.pkl         # Lasso model (with StandardScaler pipeline)
├── ridge_model.pkl         # Ridge model
├── rf_model.pkl            # Random Forest model
├── le_crop.pkl             # Label encoder — Crop
├── le_state.pkl            # Label encoder — State
└── le_season.pkl           # Label encoder — Season
```

---

## 🚀 Run Locally

**1. Clone the repository**
```bash
git clone https://github.com/smruti-123-lang/crop_yield_prediction
cd crop-yield-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. Open in browser**
```
https://cropyieldpredictionforcrops.streamlit.app/
```

---

## 📦 Requirements

```
pandas
numpy
scikit-learn
streamlit
matplotlib
seaborn
```

Or install directly:
```bash
pip install pandas numpy scikit-learn streamlit matplotlib seaborn
```

---

## 🖥️ App Features

- Select **Crop**, **State**, **Season**, **Year** from dropdowns
- Enter **Area**, **Production**, **Rainfall**, **Fertilizer**, **Pesticide**
- Click **Predict Yield** → see all 4 model predictions side by side
- Random Forest result highlighted as the best model

---

## 💡 Key Learnings

- Log-transforming a heavily skewed target dramatically improves model performance
- Feature engineering (ratio features) can boost correlation from 0.28 → 0.73
- Random Forest outperforms linear models significantly on this dataset (R² 0.93 → 0.99)
- Lasso requires feature scaling to converge properly
- Always save `LabelEncoder` objects fitted on training data for consistent inference

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
