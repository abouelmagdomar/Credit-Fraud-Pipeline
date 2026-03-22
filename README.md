# Credit-Fraud-Pipeline

An end-to-end fraud detection pipeline built on Databricks Community Edition. Raw credit card transactions flow through a Bronze → Silver → Gold medallion architecture using PySpark and Delta Lake, a Random Forest model is trained and tracked with MLflow, and results are surfaced on a live public Streamlit dashboard.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks-Community%20Edition-red?logo=databricks&logoColor=white)
![PySpark](https://img.shields.io/badge/PySpark-4.1-orange?logo=apachespark&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta%20Lake-Medallion%20Architecture-00ADD8?logo=delta&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SMOTE%20%2B%20Scaler-orange?logo=scikitlearn&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-Experiment%20Tracking-0194E2?logo=mlflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Community%20Cloud-ff4b4b?logo=streamlit&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Charts-3F4F75?logo=plotly&logoColor=white)

---

## Live dashboard

[View the dashboard on Streamlit Cloud](https://your-streamlit-url-here)

---

## Project overview

This project simulates a production-grade data pipeline as it would exist at a financial institution. The goal is not just to train a fraud detection model, but to build the infrastructure around it,  data ingestion, validation, feature engineering, class balancing, model evaluation, and a public-facing results dashboard.

**Dataset:** [ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) — 284,807 real European cardholder transactions from 2013, of which 492 (0.17%) are fraudulent. Features V1–V28 are PCA-transformed for privacy; only `Time` and `Amount` retain their original form.

---

## Architecture

```
Kaggle CSV
    │
    ▼
┌─────────────────────────────────────────────┐
│  Bronze layer  (01_bronze_ingest.py)        │
│  • Enforce schema                           │
│  • Null / row validation                    │
│  • Write to Delta (ACID + time travel)      │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Silver layer  (02_silver_transform.py)     │
│  • Engineer hour_of_day, is_night           │
│  • Log-scale Amount → amount_log            │
│  • StandardScaler (sklearn)                 │
│  • SMOTE class balancing (1:2 ratio)        │
│  • Write to Delta                           │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Gold layer  (03_gold_ml.py)                │
│  • Train Logistic Regression (baseline)     │
│  • Train Random Forest (100 trees)          │
│  • Evaluate: Precision, Recall, F1,         │
│    ROC-AUC, PR-AUC                          │
│  • Log experiments with MLflow              │
│  • Write scored predictions to Delta        │
│  • Export CSV for dashboard                 │
└─────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────┐
│  Dashboard  (dashboard/app.py)              │
│  • Confusion matrix                         │
│  • ROC curve                                │
│  • Precision-recall curve                   │
│  • Fraud probability distribution           │
│  • Fraud rate by hour of day                │
│  • Live transaction scorer                  │
└─────────────────────────────────────────────┘
```

---

## Model results

| Metric      | Logistic Regression | Random Forest |
|-------------|--------------------:|----------------:|
| ROC-AUC     | 0.9853              | 0.9991          |
| PR-AUC      | 0.9795              | 0.9984          |
| Precision   | 0.9868              | 0.9986          |
| Recall      | 0.8685              | 0.9420          |
| F1          | 0.9239              | 0.9695          |

The Random Forest outperforms Logistic Regression across every metric. The most meaningful improvement is Recall, going from 86.85% to 94.20% means the model catches 7.35% more fraud cases, which in a real banking context translates directly to reduced financial loss and improved customer protection.

**Note on evaluation metrics:** Accuracy is intentionally excluded from this project. With only 0.17% of transactions being fraudulent, a model that predicts "not fraud" on every transaction achieves 99.83% accuracy while catching zero fraud cases. Precision, Recall, F1, and PR-AUC are the honest metrics for this problem.

---

## Class imbalance handling

The raw dataset contains 284,315 legitimate transactions and only 492 fraudulent ones — a 578:1 imbalance. Training directly on this distribution causes models to ignore the minority class entirely.

This project uses **SMOTE (Synthetic Minority Oversampling Technique)** with a 1:2 sampling strategy, generating synthetic fraud cases by interpolating between real ones in feature space. The result is approximately 284,000 legitimate and 142,000 fraud training examples, a balanced enough ratio for the model to learn the fraud signal without distorting its sense of base rates.

---

## Feature engineering

Because V1–V28 are PCA-anonymised and carry no interpretable meaning, all feature engineering is applied to `Time` and `Amount`:

| Feature | Derivation | Rationale |
|---------|-----------|-----------|
| `hour_of_day` | `floor(Time / 3600) % 24` | Fraud rates vary significantly by hour |
| `is_night` | 1 if hour is 22–05, else 0 | Off-hours transactions carry higher risk |
| `amount_log` | `log1p(Amount)` | Compresses right-skewed distribution for model stability |

`Time` and `Amount` are dropped after engineering to avoid redundancy.

---

## Tech stack

| Layer | Technology |
|-------|-----------|
| Compute | Databricks Community Edition (serverless) |
| Language | Python 3, PySpark 4.1 |
| Storage | Delta Lake (Unity Catalog) |
| ML — distributed | PySpark ML (VectorAssembler, RandomForestClassifier, LogisticRegression) |
| ML — in-memory | scikit-learn (StandardScaler, SMOTE via imbalanced-learn) |
| Experiment tracking | MLflow (Databricks managed) |
| Dashboard | Streamlit Cloud |
| Visualisation | Plotly |

---

## Repository structure

```
credit-fraud-pipeline/
├── notebooks/
│   ├── 01_bronze_ingest.py       # Raw CSV → Delta (Bronze)
│   ├── 02_silver_transform.py    # Feature engineering + SMOTE (Silver)
│   └── 03_gold_ml.py             # Model training + scoring (Gold)
├── dashboard/
│   ├── app.py                    # Streamlit dashboard
│   └── scored_predictions.csv   # Model output (exported from Gold)
├── requirements.txt
└── README.md
```

---

## Running the notebooks

1. Upload `creditcard.csv` to `/Volumes/main/credit-fraud-pipeline/data/bronze/` in your Databricks workspace
2. Create a Unity Catalog schema named `credit-fraud-pipeline` under the `main` catalog
3. Run notebooks in order: `01` → `02` → `03`
4. Download `scored_predictions.csv` from `/Volumes/main/credit-fraud-pipeline/data/gold/` and place it in `dashboard/`

## Running the dashboard locally

```bash
pip install -r requirements.txt
streamlit run dashboard/app.py
```

---

## Key concepts demonstrated

- **Medallion architecture:** strict separation of raw, cleaned, and serving layers with Delta tables at each stage
- **Delta Lake time travel:** every table write is versioned; historical states are queryable
- **Class imbalance:** SMOTE oversampling with deliberate 1:2 ratio rather than naive 50/50 balancing
- **Correct evaluation metrics:** PR-AUC and F1 prioritised over accuracy for imbalanced classification
- **MLflow experiment tracking:** both models logged with parameters and metrics for reproducible comparison
- **Serverless-compatible PySpark:** RDD operations replaced with Pandas-native equivalents throughout

