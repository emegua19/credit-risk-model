# Credit Risk Probability Model  
**End-to-End ML Project | 10 Academy | Bati Bank – Buy Now, Pay Later**

[![CI/CD](https://img.shields.io/github/actions/workflow/status/<your-username>/credit-risk-model/ci.yml?label=CI%2FCD&style=flat-square)](https://github.com/<your-username>/credit-risk-model/actions)
[![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)](./LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?style=flat-square)](https://www.python.org/)

---

## Project Overview

This project is part of the 10 Academy KAIM5 - Week 5 Challenge. It involves building a production-grade Credit Risk Scoring System for Bati Bank, a leading financial provider. The model uses alternative eCommerce behavioral data to:

- Predict risk probability  
- Assign credit scores  
- Recommend loan limits and durations  

All of this is designed in compliance with the Basel II Capital Accord.

---

## Project Structure

```

credit-risk-model/
├── .github/workflows/ci.yml          # GitHub Actions for CI/CD
├── notebooks/1.0-eda.ipynb           # Exploratory data analysis
├── src/
│   ├── config.py                     # Configurations
│   ├── data\_processing.py           # Feature engineering logic
│   ├── inference.py                 # Inference code
│   ├── models.py                    # Training and evaluation logic
│   ├── rfm\_clustering.py            # RFM proxy variable generation
│   └── api/
│       ├── main.py                  # FastAPI backend
│       └── pydantic\_models.py      # Input/output validation
├── tests/                            # Unit tests for robustness
├── docker-compose.yml
├── requirements.txt
└── README.md

````

---

## Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretable, Auditable Models

The Basel II Accord outlines a comprehensive framework for risk-sensitive banking regulation, composed of three interrelated pillars:

| Pillar       | Description |
|--------------|-------------|
| Pillar 1     | Ensures banks hold ≥ 8% capital against risk-weighted assets (RWA). Allows statistical risk models (IRB) requiring full documentation and validation. |
| Pillar 2     | Supervisors evaluate a bank's internal risk models and capital adequacy under stress. Models must be defensible, conservative, and stress-tested. |
| Pillar 3     | Enforces transparency with public disclosure of risk management processes and metrics, creating external pressure to maintain model quality. |

**Modeling Implications:**  
We prioritize transparent models (e.g., Logistic Regression with WoE) and maintain detailed documentation to satisfy internal validation and regulatory audits.

---

### 2. Why a Proxy Variable is Needed — and the Risks

The dataset does not include a direct label for “default.” Instead, we build a proxy target variable:

- Use Recency, Frequency, and Monetary (RFM) features  
- Cluster customers using K-Means (k=3)  
- Label the least engaged group as high-risk (`is_high_risk = 1`)

**Risks Involved:**

- Misclassification: Low-frequency but creditworthy customers may be wrongly denied.
- Bias: Proxy ignores critical variables like income or debt history.
- Endogeneity: Behavior might correlate with untracked confounders, biasing predictions.
- Compliance: Regulatory concerns may arise if proxy logic isn't well-validated.

**Mitigation:** Use clustering diagnostics, domain expert validation, and incorporate more data sources when available.

---

### 3. Model Selection Trade-Offs

| Feature             | Simple Model (Logistic Regression + WoE)        | Complex Model (e.g., Gradient Boosting, XGBoost) |
|---------------------|--------------------------------------------------|---------------------------------------------------|
| Interpretability     | Transparent and regulator-friendly               | Black-box; requires SHAP or LIME for explanation  |
| Regulatory Fit       | Easily documented and auditable                  | Requires additional justification and validation  |
| Accuracy             | May underfit complex relationships               | High predictive power for complex patterns        |
| Developer Overhead   | Simple to train, debug, and deploy               | Resource-intensive; requires tuning and monitoring |
| Overfitting Risk     | Low on small data                                | Higher; needs validation and regularization       |
| Business Impact      | Conservative decisions may miss credit opportunities | Optimized approval rates, but harder to govern    |

**Conclusion:**  
Start with simple models to meet regulatory expectations. Consider complex models only if they significantly improve accuracy and can be explained effectively.

---

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/emegua19/credit-risk-model.git
cd credit-risk-model
````

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run EDA:

```bash
jupyter notebook notebooks/1.0-eda.ipynb
```

4. Train models:

```bash
python src/models.py
```

5. Run the API with Docker:

```bash
docker-compose up --build
```

---

## References

* [Basel II: Overview – Investopedia](https://www.investopedia.com/terms/b/baselii.asp)
* [Credit Risk Scoring with Scorecards – TDS](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
* [Proxy Variables Explained – Analytics Vidhya](https://www.analyticsvidhya.com/blog/2021/06/how-to-use-proxy-variables-in-a-regression-model/)
* [SHAP for Explainable ML](https://github.com/slundberg/shap)

---

## Author

**Yitbarek Geletaw**
Analytics Engineer – Bati Bank