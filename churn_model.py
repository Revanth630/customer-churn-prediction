"""Customer churn prediction: synthetic data -> features -> logistic regression -> risk scores."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG = np.random.default_rng(42)


def make_data(n=5000):
    """Generate a telecom-style customer dataset with realistic churn drivers."""
    df = pd.DataFrame({
        "tenure_months": RNG.integers(1, 72, n),
        "monthly_charges": RNG.normal(70, 25, n).clip(20, 150).round(2),
        "support_tickets": RNG.poisson(1.5, n),
        "contract": RNG.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.2]),
        "payment_method": RNG.choice(["UPI", "Card", "Bank transfer", "Cash"], n),
        "has_autopay": RNG.integers(0, 2, n),
    })
    logit = (-1.0 - 0.04 * df.tenure_months + 0.02 * df.monthly_charges
             + 0.35 * df.support_tickets - 0.8 * df.has_autopay
             + df.contract.map({"Month-to-month": 1.2, "One year": 0.0, "Two year": -1.0}))
    df["churn"] = (RNG.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df


def build_model():
    num = ["tenure_months", "monthly_charges", "support_tickets", "has_autopay"]
    cat = ["contract", "payment_method"]
    prep = ColumnTransformer([("num", StandardScaler(), num),
                              ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
    return Pipeline([("prep", prep), ("clf", LogisticRegression(max_iter=1000, class_weight="balanced"))])


def main():
    df = make_data()
    X, y = df.drop(columns="churn"), df.churn
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    model = build_model().fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]
    print(f"Churn rate: {y.mean():.1%}  |  ROC-AUC: {roc_auc_score(y_te, proba):.3f}\n")
    print(classification_report(y_te, model.predict(X_te)))

    names = model.named_steps["prep"].get_feature_names_out()
    coefs = pd.Series(model.named_steps["clf"].coef_[0], index=names).sort_values()
    print("Top churn drivers (log-odds):\n", coefs.round(2).to_string())

    # Business output: risk bands that retention teams can act on
    scored = X_te.assign(churn_prob=proba)
    scored["risk_band"] = pd.cut(proba, [0, 0.3, 0.6, 1], labels=["Low", "Medium", "High"])
    scored.to_csv("churn_scores.csv", index=False)
    print("\nCustomers per risk band:\n", scored.risk_band.value_counts().to_string())


if __name__ == "__main__":
    main()
