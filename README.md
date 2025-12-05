# Telco Customer Churn Analytics

End-to-end analysis of telecom customer churn using:
- data integration from multiple source tables,
- exploratory data analysis (EDA) and feature engineering,
- predictive modeling,
- and business-oriented retention strategy simulation.

---

## 1. Project Overview

This project follows a typical data science workflow applied to a Telco churn problem:

1. **Data integration** – merge customer status, services, demographics, location and population datasets into a single analytical table.
2. **EDA & feature engineering** – explore churn patterns and create new variables such as engagement, contract value and CLTV per month.
3. **Modeling** – train and evaluate Logistic Regression, Random Forest and XGBoost to predict churn probability.
4. **Threshold tuning & metrics** – optimise decision thresholds with a focus on recall (catching as many churners as possible) and compare accuracy, precision, recall, F1 and ROC-AUC.
5. **Explainability & dashboards** – use SHAP and interactive Plotly dashboards to understand why customers churn.
6. **Business questions & ROI** – analyse tenure windows, CLTV segments, refunds, payment methods and service bundles, and estimate the ROI of retention campaigns.

---

## 2. Repository Structure

```text
telco-customer-churn-analytics/
├── data/
│   └── raw/                  # Original Telco churn Excel files
├── notebooks/                # Jupyter notebooks for EDA, modeling and business analysis
├── 03_Outputs/               # Exported tables, figures and HTML dashboards (created by notebooks)
├── 04_Artifacts/             # Saved models / preprocessing artifacts (created by notebooks)
├── app.py                    # Script to export and load churn model via Hugging Face
├── requirements.txt          # Python dependencies
├── .gitignore                # Files and folders ignored by git
└── README.md                 # This file
