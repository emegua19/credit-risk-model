# Credit Risk Probability Model

## Project Overview
This project, part of the 10 Academy KAIM5 Week 5 program, involves building an end‑to‑end machine learning solution for **Bati Bank** to assess credit risk for a buy‑now‑pay‑later (BNPL) service. Using eCommerce transaction data, the model will:
- Predict credit risk (is_high_risk)
- Assign a credit score from risk probabilities
- Recommend optimal loan amounts and durations  
All while adhering to **Basel II Capital Accord** standards.

---

## Project Structure
```

credit‑risk‑model/
├── .github/
│   └── workflows/
│       └── ci.yml
├── notebooks/
│   └── 1.0‑eda.ipynb
├── src/
│   ├── **init**.py
│   ├── config.py
│   ├── data\_processing.py
│   ├── inference.py
│   ├── models.py
│   ├── rfm\_clustering.py
│   └── api/
│       ├── **init**.py
│       ├── main.py
│       └── pydantic\_models.py
├── tests/
│   ├── **init**.py
│   ├── test\_data\_processing.py
│   ├── test\_rfm\_clustering.py
│   └── test\_models.py
├── .gitignore
├── docker‑compose.yml
├── requirements.txt
└── README.md

````

---

## Credit Scoring Business Understanding

### 1. Influence of Basel II on Model Interpretability and Documentation
The **Basel II Capital Accord** (2004) emphasizes risk‑sensitive capital allocation:

- **Pillar 1 (Minimum Capital Requirements):** Banks must hold ≥ 8 % of risk‑weighted assets against credit risk, often using IRB (Internal Ratings‑Based) models.  
- **Pillar 3 (Market Discipline):** Requires both qualitative (annual) and quantitative (bi‑annual) disclosures of risk management practices.

**Why this matters:**  
Interpretable models (e.g., Logistic Regression with Weight of Evidence) make it clear how each feature influences the risk score, which regulators can audit. Detailed documentation—covering data sources, assumptions, transformation logic, and validation results—is mandatory to satisfy Basel II’s transparency and validation requirements.

### 2. Necessity and Risks of Using a Proxy Variable
#### Why a proxy?
Our dataset lacks a direct “default” label. We construct an **is_high_risk** proxy by:
1. Calculating **RFM** (Recency, Frequency, Monetary) metrics per customer  
2. Clustering with **K‑Means** (3 segments)  
3. Labeling the least engaged cluster (low R, low F, low M) as high‑risk (`is_high_risk = 1`)

#### Potential business and regulatory risks:
- **Misclassification:** Low‑activity but creditworthy customers may be flagged high‑risk (false positives), denying them credit; conversely, risky customers might slip through (false negatives).
- **Incomplete View:** RFM ignores credit history, macroeconomic factors, and payment behavior, possibly biasing risk estimates.
- **Correlation Bias (Endogeneity):** If RFM correlates with unmodeled drivers of default (e.g., seasonal shopping), predictions can be systematically skewed.
- **Regulatory Exposure:** An ineffective proxy could breach Basel II disclosure obligations, leading to penalties.

> **Mitigation:** Rigorously validate the proxy against any available external default data, monitor performance, and incorporate additional data sources when possible.

### 3. Trade‑offs: Simple vs. Complex Models
| Aspect               | **Simple Models**<br/>(Logistic Regression + WoE)       | **Complex Models**<br/>(Gradient Boosting)                            |
|----------------------|---------------------------------------------------------|------------------------------------------------------------------------|
| **Interpretability** | Highly transparent; easy to explain to regulators        | “Black box”; requires SHAP or LIME for feature‑level insights          |
| **Regulatory Fit**   | Straightforward validation and disclosure               | Extra documentation and explainability work needed                     |
| **Predictive Power** | May underfit and miss non‑linear patterns               | Captures complex relationships → higher accuracy                       |
| **Resource Needs**   | Low compute & domain expertise                          | Higher compute, specialized tuning, and ongoing monitoring             |
| **Overfitting Risk** | Lower; robust with smaller datasets                     | Higher; needs careful cross‑validation and regularization              |
| **Business Impact**  | Predictable, stable decisions; potentially more false rejects | Better risk discrimination; may over‑cap and require capital buffers |

> **Recommendation:** In a regulated context, start with a simple, interpretable model for compliance ease. If performance gaps remain material, consider a complex model—paired with strong explainability tools—and re‑evaluate capital requirements accordingly.

---

## Setup Instructions

1. **Clone the repo**  
   ```bash
   git clone <your‑repo‑url>
   cd credit‑risk‑model
````

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```
3. **Run EDA**

   ```bash
   jupyter notebook notebooks/1.0-eda.ipynb
   ```
4. **Train models**

   ```bash
   python src/models.py
   ```
5. **Run API (Docker)**

   ```bash
   docker-compose up --build
   ```

---

## Key References

* [Basel II Capital Accord Overview (Investopedia)](https://www.investopedia.com/terms/b/baselii.asp)
* [Credit Risk Modeling and Scorecards](https://towardsdatascience.com/how-to-develop-a-credit-risk-model-and-scorecard-91335fc01f03)
* [Proxy Variables in Regression Models](https://www.analyticsvidhya.com/blog/2021/06/how-to-use-proxy-variables-in-a-regression-model/)
* [SHAP for Model Explainability](https://github.com/slundberg/shap)

---

```
```
