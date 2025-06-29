# Credit Risk Probability Model

End-to-End ML Project | 10 Academy | Bati Bank – Buy Now, Pay Later

---

## Project Overview

This project, part of the 10 Academy KAIM5 - Week 5 Challenge, delivers a production-ready Credit Risk Scoring System for Bati Bank, a leading financial provider. Using alternative eCommerce behavioral data (95,663 rows), the system:

- Predicts credit risk probabilities
- Assigns credit scores
- Recommends loan limits and durations

The project is built in full compliance with the Basel II Capital Accord, following a modular, scalable, Object-Oriented Programming (OOP) approach.

- Tasks 1–4 (Business Understanding, EDA, Feature Engineering, Proxy Variable Creation) - Completed by June 29, 2025
- Tasks 5–6 (Model Training, API Deployment) - In Progress, scheduled for July 1, 2025

---

## CREDIT-RISK-MODEL Project Structure

```
CREDIT-RISK-MODEL/
├── .github/
│   └── workflows/
│       └── ci.yml                    # GitHub Actions workflow for CI/CD (linting, testing)
├── data/
│   ├── raw/                         # Raw datasets (e.g., transactions.csv) as initial input
│   └── processed/                   # Processed data files (e.g., processed_data.csv) post-transformation
├── notebooks/
│   └── 1.0-eda.ipynb                # Jupyter Notebook for Exploratory Data Analysis
├── outputs/
│   ├── logs/                        # Log files from script executions for debugging and tracking
│   └── plots/                       # Generated plots and visualizations from EDA and models
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

## Setup Instructions

```bash
git clone https://github.com/emegua19/credit-risk-model.git
cd CREDIT-RISK-MODEL

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Run Key Pipelines

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

Note: Ensure Docker is installed (check with `docker --version`) and Jupyter Notebook is available (install via `pip install notebook` if needed).

---

## Basel II and the Need for Interpretable, Auditable Models

The Basel II Accord introduces a global standard for risk-sensitive banking regulation through three pillars:

| Pillar       | Description                                                                                                                                                  |
|--------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Pillar 1     | Banks must maintain at least 8% capital against risk-weighted assets. Statistical models used for risk must be transparent, fully documented, and validated. |
| Pillar 2     | Supervisors assess internal models and risk management processes, requiring models to be conservative and stress-tested.                                     |
| Pillar 3     | Banks must disclose risk management processes and metrics publicly, ensuring accountability and market confidence.                                           |

Modeling Implications:  
To comply with Basel II, this project prioritizes transparent models like Logistic Regression with Weight of Evidence encoding. All model development, validation, and assumptions are carefully documented to support regulatory review.

---

## Model Selection Trade-offs

| Feature            | Simple Model (LogReg + WoE)  | Complex Model (Random Forest)        |
|--------------------|------------------------------|--------------------------------------|
| Interpretability   | High, regulator-friendly     | Lower, requires SHAP for explanation |
| Regulatory Fit     | Easy to document and justify | Requires additional justification    |
| Accuracy           | Moderate baseline            | Higher predictive power              |
| Developer Overhead | Low                          | High, tuning and monitoring required |
| Overfitting Risk   | Low                          | Higher, requires validation          |
| Business Impact    | Conservative decisions       | Optimized approvals, higher risk     |

The project uses both approaches to balance interpretability and performance.

---

## EDA Summary – Key Insights and Results

### Data Overview
- Dataset Size: 95,663 transactions with 12 features, including `TransactionID`, `Amount`, `FraudResult`, and `TransactionStartTime`.
- Data Types: Mix of numerical (e.g., `Amount`, `Value`) and categorical (e.g., `Channel`) variables.

### Key Statistical Findings
- Target Distribution (`FraudResult`): Only 193 rows (0.2%) indicate fraud, confirming severe class imbalance.
- Amount Analysis: Mean = 145.32, Median = 89.50, Std = 298.45; 40% of transactions are negative, suggesting refunds or reversals.
- Missing Values: No missing data across all columns, simplifying initial preprocessing.
- Skewness and Outliers: `Amount` and `Value` exhibit right-skewness (skewness > 2.5) with outliers exceeding 3 standard deviations.
- Correlation: Weak correlation (r < 0.1) between `Amount` and `FraudResult`, but negative amounts show a slight positive correlation with fraud (r = 0.15).

### Visualization Results
- Distribution Plots: `Amount` and `Value` show long tails, with log transformation recommended.
- Boxplots: Outliers in `Amount` are prominent, especially in Web channel transactions.
- Correlation Heatmap: Highlights `IsNegativeAmount` as a potential predictor for `FraudResult`.
- Fraud Patterns: Higher fraud incidence (0.5%) in transactions with negative amounts compared to 0.1% in positive amounts.

### Listed Images
The following images are generated and saved in `outputs/plots/`:
- `distribution_amount.png`: Displays the distribution of the `Amount` column.
- `distribution_value.png`: Shows the distribution of the `Value` column.
- `boxplot_amount.png`: Highlights outliers in the `Amount` column across transaction channels.
- `correlation_heatmap.png`: Visualizes correlations between features, including `FraudResult` and `IsNegativeAmount`.
- `fraud_by_amount_type.png`: Compares fraud incidence between positive and negative `Amount` transactions.

### Implications for Modeling
- The imbalance necessitates resampling (e.g., SMOTE) or class weighting.
- Negative amounts and their fraud correlation justify the `IsNegativeAmount` feature.
- Log transformation of `Amount` and `Value` will reduce skewness for better model performance.
- `CountryCode` (constant at 256) is excluded to avoid noise.

These visualizations and interpretations will be further detailed in `1.0-eda.ipynb` with enhanced Markdown to improve comprehensiveness.

---

## Model Training & Evaluation Pipeline

The model training process follows a structured, transparent pipeline designed for reproducibility, regulatory alignment, and model performance tracking.

### Data Split Strategy
- The processed dataset, including the `is_high_risk` proxy target, is split into training (80%) and testing (20%) sets using stratified sampling to preserve class balance.

### Models Used
Two complementary models are trained and compared:
- Logistic Regression (with Weight of Evidence encoding)  
  - Interpretable, regulator-friendly baseline model
- Random Forest Classifier  
  - More complex, non-linear model to potentially improve predictive performance

### Hyperparameter Tuning
- Performed using GridSearchCV, optimizing key parameters such as:  
  - `C` for Logistic Regression  
  - `n_estimators` and `max_depth` for Random Forest  
- Cross-validation ensures robust model selection.

### Evaluation Metrics
Model performance is assessed using:
- Accuracy
- Precision
- Recall (Sensitivity)
- F1 Score
- ROC-AUC (preferred for imbalanced datasets)
- Confusion matrices and classification reports are generated for further insight.

### Experiment Tracking with MLflow
All experiments, including:
- Model hyperparameters
- Evaluation metrics
- Final trained model artifacts
are logged and tracked using MLflow, providing:
- Transparent model comparison
- Reproducible results
- Version control for models

This pipeline is implemented in `src/models.py` and automated via `scripts/run_models.py`.

---

## Interim Task Progress

| Task                     | Status         | Description                                  |
|--------------------------|----------------|----------------------------------------------|
| Task 1 – Business Understanding | Completed    | Basel II understanding and modeling approach |
| Task 2 – Exploratory Data Analysis | Completed    | EDA notebook, insights, and visualizations   |
| Task 3 – Feature Engineering    | Completed    | Feature engineering with engineered features |
| Task 4 – Proxy Target Variable  | Completed    | RFM clustering and proxy target generation   |
| Task 5 – Model Training         | In Progress    | Model training and MLflow tracking           |
| Task 6 – API Deployment         | Planned        | API deployment with FastAPI and Docker       |

---

## References

- [Basel II: Overview – Investopedia](https://www.investopedia.com/terms/b/baselii.asp)
- [Credit Risk Scoring with Scorecards – TDS](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
- [Proxy Variables – Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/how-to-use-proxy-variables-in-a-regression-model/)
- [SHAP for Explainable ML](https://github.com/slundberg/shap)

---

## Author

**Yitbarek Geletaw**  
Analytics Engineer  
10 Academy – Bati Bank
