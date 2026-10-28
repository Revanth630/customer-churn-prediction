# Customer Churn Prediction

An end-to-end churn model written the way a business analyst would approach it: **frame the problem → define requirements → model → turn it into decisions.** It builds on the churn modelling I did at NanoBi Analytics, where proof-of-concept outputs helped win a client. In this version I scoped the business problem, defined the analytical requirements and built the model, and the outputs feed a dashboard so leadership can act on retention signals.

## Business problem
Retention teams need to know *which* customers are at risk and *why*, so they can act before those customers leave.

## Approach
1. **Data**: a synthetic telecom-style dataset (tenure, charges, contract, support tickets, payment method)
2. **Pipeline**: `StandardScaler` + `OneHotEncoder` → class-balanced **Logistic Regression** (Scikit-learn)
3. **Evaluation**: ROC-AUC and a classification report
4. **Explainability**: the model's coefficients rank the churn drivers
5. **Output**: `churn_scores.csv` with a Low/Medium/High **risk band** for each customer, ready for a Power BI dashboard

## Run
```bash
pip install -r requirements.txt
python churn_model.py
```

## Key insights
- Month-to-month contracts and frequent support tickets are the strongest churn signals
- Longer tenure and autopay lower churn risk

> The data is synthetic. It mirrors the churn modelling I've done in industry without exposing any client data.
