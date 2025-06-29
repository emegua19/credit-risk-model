#  Credit Risk Probability Model

*End-to-End ML Project | 10 Academy | Bati Bank – Buy Now, Pay Later*

---

##  Project Overview

This project, part of the **10 Academy KAIM5 - Week 5 Challenge**, delivers a production-ready Credit Risk Scoring System for **Bati Bank**, a leading financial provider. Using alternative **eCommerce behavioral data** (95,663 rows), the system:

✅ Predicts credit risk probabilities
✅ Assigns credit scores
✅ Recommends loan limits and durations

The project is built in full compliance with the **Basel II Capital Accord**, following a modular, scalable, Object-Oriented Programming (OOP) approach.

✔️ **Tasks 1–4** (Business Understanding, EDA, Feature Engineering, Proxy Variable Creation) — ✅ Completed by *June 29, 2025*
✔️ **Tasks 5–6** (Model Training, API Deployment) —  In Progress, scheduled for *July 1, 2025*

---

##  CREDIT-RISK-MODEL Project Structure

```
CREDIT-RISK-MODEL/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions workflow for CI/CD (linting, testing)
├── data/
│   ├── raw/                         # Raw datasets (e.g., transactions.csv)
│   └── processed/                   # Processed data files (e.g., processed_data.csv)
├── notebooks/
│   └── 1.0-eda.ipynb                # Jupyter Notebook for Exploratory Data Analysis
├── outputs/
│   ├── logs/                        # Log files from script executions
│   └── plots/                       # Generated plots and visualizations
├── src/
│   ├── __init__.py                  # Makes src a Python package
│   ├── config.py                    # Configuration settings (paths, hyperparameters)
│   ├── data_explorer.py             # OOP-based class for EDA and visualizations
│   ├── data_processing.py           # OOP-based data processing pipeline
│   ├── inference.py                 # Logic for model inference and predictions
│   ├── models.py                    # Machine learning model definitions
│   ├── rfm_clustering.py            # OOP-based RFM calculation and clustering
│   └── api/
│       ├── __init__.py              # Makes api a Python package
│       ├── main.py                  # FastAPI application for serving predictions
│       └── pydantic_models.py       # Pydantic models for API data validation
├── tests/
│   ├── __init__.py                  # Makes tests a Python package
│   ├── test_data_explorer.py        # Unit tests for DataExplorer
│   ├── test_data_processing.py      # Unit tests for DataProcessor
│   ├── test_rfm_clustering.py       # Unit tests for RFMClustering
│   └── test_models.py               # Unit tests for models
├── scripts/
│   ├── run_data_processing.py       # Script to execute data processing pipeline
│   ├── run_models.py                # Script to execute model training
│   └── run_rfm_clustering.py        # Script to execute RFM clustering
├── .gitignore                       # Files and directories to exclude from version control
├── .venv/                           # Python virtual environment
├── docker-compose.yml               # Docker Compose configuration for API deployment
├── LICENSE                          # Project license (e.g., MIT License)
├── README.md                        # Project documentation and instructions
├── requirements.txt                 # Python dependencies (e.g., pandas, scikit-learn)
└── .pytest_cache/                   # Cache directory for pytest
```

---

##  Setup Instructions

```bash
git clone https://github.com/emegua19/credit-risk-model.git
cd CREDIT-RISK-MODEL

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

##  Run Key Pipelines

```bash
# EDA notebook
jupyter notebook notebooks/1.0-eda.ipynb

# Data Processing
python scripts/run_data_processing.py

# RFM Proxy Label Creation
python scripts/run_rfm_clustering.py

# Model Training (Task 5)
python scripts/run_models.py

# API Deployment (Task 6)
docker-compose up --build
```

---

##  Basel II and the Need for Interpretable, Auditable Models

The Basel II Accord introduces a global standard for risk-sensitive banking regulation through three pillars:

| Pillar       | Description                                                                                                                                                  |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Pillar 1** | Banks must maintain at least 8% capital against risk-weighted assets. Statistical models used for risk must be transparent, fully documented, and validated. |
| **Pillar 2** | Supervisors assess internal models and risk management processes, requiring models to be conservative and stress-tested.                                     |
| **Pillar 3** | Banks must disclose risk management processes and metrics publicly, ensuring accountability and market confidence.                                           |

**Modeling Implications:**
To comply with Basel II, this project prioritizes transparent models like Logistic Regression with Weight of Evidence encoding. All model development, validation, and assumptions are carefully documented to support regulatory review.

---

##  Model Selection Trade-offs

| Feature            | Simple Model (LogReg + WoE)  | Complex Model (Random Forest)        |
| ------------------ | ---------------------------- | ------------------------------------ |
| Interpretability   | High, regulator-friendly     | Lower, requires SHAP for explanation |
| Regulatory Fit     | Easy to document and justify | Requires additional justification    |
| Accuracy           | Moderate baseline            | Higher predictive power              |
| Developer Overhead | Low                          | High, tuning and monitoring required |
| Overfitting Risk   | Low                          | Higher, requires validation          |
| Business Impact    | Conservative decisions       | Optimized approvals, higher risk     |

The project uses both approaches to balance interpretability and performance.

---

##  EDA Summary – Key Insights

* **Highly Imbalanced Target (`FraudResult`)**: Only 0.2% of transactions are flagged as fraud, requiring class weighting or resampling.
* **Significant Negative Amounts**: 40% of transactions have negative amounts, likely refunds or reversals. Captured via `IsNegativeAmount` feature.
* **No Missing Values**: The dataset is clean, requiring no imputation.
* **Severe Skewness and Outliers**: `Amount` and `Value` columns are right-skewed with large outliers; addressed via log transformation.
* **Uninformative Feature**: `CountryCode` is constant and excluded from modeling.

Visual outputs are saved in `outputs/plots/` for further reference.

---

##  Model Training & Evaluation Pipeline

The model training process follows a structured, transparent pipeline designed for reproducibility, regulatory alignment, and model performance tracking.

### ✅ **Data Split Strategy**

* The processed dataset, including the `is_high_risk` proxy target, is split into **training (80%)** and **testing (20%)** sets using stratified sampling to preserve class balance.

### ✅ **Models Used**

Two complementary models are trained and compared:

* **Logistic Regression (with Weight of Evidence encoding)**

  * Interpretable, regulator-friendly baseline model
* **Random Forest Classifier**

  * More complex, non-linear model to potentially improve predictive performance

### ✅ **Hyperparameter Tuning**

* Performed using **GridSearchCV**, optimizing key parameters such as:

  * `C` for Logistic Regression
  * `n_estimators` and `max_depth` for Random Forest
* Cross-validation ensures robust model selection.

### ✅ **Evaluation Metrics**

Model performance is assessed using:

* **Accuracy**
* **Precision**
* **Recall (Sensitivity)**
* **F1 Score**
* **ROC-AUC** (preferred for imbalanced datasets)
* Confusion matrices and classification reports are generated for further insight.

### ✅ **Experiment Tracking with MLflow**

All experiments, including:

* Model hyperparameters
* Evaluation metrics
* Final trained model artifacts
  are logged and tracked using **MLflow**, providing:
* Transparent model comparison
* Reproducible results
* Version control for models

This pipeline is implemented in `src/models.py` and automated via `scripts/run_models.py`.

---

##  Interim Task Progress

| Task   | Status         | Description                                  |
| ------ | -------------- | -------------------------------------------- |
| Task 1 | ✅ Completed    | Basel II understanding and modeling approach |
| Task 2 | ✅ Completed    | EDA notebook, insights, and visualizations   |
| Task 3 | ✅ Completed    | Feature engineering with engineered features |
| Task 4 | ✅ Completed    | RFM clustering and proxy target generation   |
| Task 5 |  In Progress | Model training and MLflow tracking           |
| Task 6 |  Planned      | API deployment with FastAPI and Docker       |

---
