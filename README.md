# Credit Risk Probability Model

**End-to-End ML Project | 10 Academy | Bati Bank – Buy Now, Pay Later**

---

## Project Overview

This project involves the development of a production-grade Credit Risk Scoring System for Bati Bank, enabling them to offer a Buy Now, Pay Later (BNPL) service using alternative eCommerce behavioral data. The dataset includes 95,663 transaction records, and the objective is to:

* Predict the probability of default or credit risk for individual customers.
* Assign interpretable and data-driven credit scores.
* Recommend personalized loan limits and repayment durations.

The implementation is aligned with the **Basel II Capital Accord**, ensuring compliance with global risk management standards. The solution is modular, scalable, and written in an object-oriented Pythonic style, with CI/CD, unit testing, and deployment-ready components using Docker and FastAPI.

---

## Task Descriptions

### Task 1 – Business Understanding and Regulatory Framing

* Reviewed Basel II Capital Accord and its three pillars (Minimum Capital Requirements, Supervisory Review, and Market Discipline).
* Defined project objectives aligned with regulatory compliance, interpretability, and explainability.
* Mapped business goals (credit scoring, loan recommendation) to data science deliverables.
* Chose simple, interpretable models (e.g., Logistic Regression with WoE) as a starting point.

### Task 2 – Exploratory Data Analysis (EDA)

* Performed comprehensive EDA using the `DataExplorer` class and Jupyter notebook (`notebooks/1.0-eda.ipynb`).
* Generated and analyzed:

  * Summary statistics, missing values, outliers
  * Distributions of numerical and categorical features
  * Correlation matrix and visual trends
* Identified critical data patterns:

  * Highly imbalanced target (`FraudResult`) – only 0.2% positive class
  * 40% transactions with negative `Amount`, likely refunds or reversals
  * No missing values across columns
  * Strong right-skew and outliers in `Amount` and `Value`
  * Uninformative feature `CountryCode` (constant = 256)

### Task 3 – Feature Engineering

* Implemented `DataProcessor` class in `src/data_processing.py`.
* Created engineered features such as:

  * `IsNegativeAmount`: binary flag for negative `Amount`
  * Time-based features extracted from `TransactionStartTime`: `Hour`, `Day`, `Month`
  * Log-transformed `Amount` and `Value` for reducing skewness
* Saved processed output to `data/processed/processed_data.csv`

### Task 4 – Proxy Target Variable via RFM Clustering

* Implemented `RFMClustering` class in `src/rfm_clustering.py`
* Computed RFM (Recency, Frequency, Monetary) features for all customers
* Applied KMeans clustering (k=3) to segment customers
* Assigned proxy label `is_high_risk = 1` to least engaged cluster
* Validated cluster assumptions with `FraudResult` indicator

---

## EDA Summary – Key Insights

1. **Highly Imbalanced Target (FraudResult)**
   Only 193 of 95,663 rows (0.2%) have `FraudResult > 0`, indicating high class imbalance.

2. **Negative Amounts Are Common**
   38,189 transactions (\~40%) have negative `Amount`, suggesting the importance of a refund-based flag.

3. **No Missing Data**
   All columns are complete with no nulls, reducing preprocessing complexity.

4. **Outliers and Skewness**
   `Amount` and `Value` fields are right-skewed with extreme outliers; transformation is necessary.

5. **Uninformative Feature**
   `CountryCode` has no variance and will be dropped during feature selection.

---

## Project Structure

```
CREDIT-RISK-MODEL/
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 1.0-eda.ipynb
├── outputs/
│   ├── logs/
│   └── plots/
├── src/
│   ├── config.py
│   ├── data_explorer.py
│   ├── data_processing.py
│   ├── inference.py
│   ├── models.py
│   ├── rfm_clustering.py
│   └── api/
│       ├── main.py
│       └── pydantic_models.py
├── tests/
├── scripts/
│   ├── run_data_processing.py
│   ├── run_models.py
│   └── run_rfm_clustering.py
├── .gitignore
├── .venv/
├── docker-compose.yml
├── LICENSE
├── README.md
├── requirements.txt
└── .pytest_cache/
```

---

## Setup Instructions

```bash
# Clone the repository
git clone https://github.com/emegua19/credit-risk-model.git
cd credit-risk-model

# Set up and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run EDA
jupyter notebook notebooks/1.0-eda.ipynb

# Run data processing
python scripts/run_data_processing.py

# Generate RFM proxy labels
python scripts/run_rfm_clustering.py

# Train models
python scripts/run_models.py

# Deploy with Docker
docker-compose up --build
```

---

## References

* [Basel II: Overview – Investopedia](https://www.investopedia.com/terms/b/baselii.asp)
* [Credit Risk Scoring with Scorecards – TDS](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
* [Proxy Variables – Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/how-to-use-proxy-variables-in-a-regression-model/)
* [SHAP for Explainable ML](https://github.com/slundberg/shap)

---

## Author

**Yitbarek Geletaw**
