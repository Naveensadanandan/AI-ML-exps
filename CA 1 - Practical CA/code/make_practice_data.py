# Creates a synthetic loan-approval CSV (numeric + categorical + missing values + a sensitive attribute)
# so you can practise the full classification pipeline and a fairness check offline.
import numpy as np, pandas as pd

rng = np.random.default_rng(0); n = 2000
df = pd.DataFrame({
    "age": rng.integers(18, 70, n),
    "income": rng.lognormal(10.5, 0.5, n).round(0),
    "loan_amount": rng.lognormal(9.5, 0.6, n).round(0),
    "credit_score": rng.normal(650, 70, n).clip(300, 850).round(0),
    "employment": rng.choice(["salaried", "self_employed", "unemployed"], n, p=[.6, .3, .1]),
    "education": rng.choice(["school", "bachelor", "master"], n, p=[.4, .4, .2]),
    "gender": rng.choice(["F", "M"], n, p=[.45, .55]),
})
logit = (0.012 * (df.credit_score - 650) + 0.8 * np.log(df.income / df.loan_amount)
         - 1.2 * (df.employment == "unemployed") + 0.3 * (df.gender == "M") - 0.6)  # 0.3 = historical bias
df["approved"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
for c in ["income", "credit_score", "employment"]:
    df.loc[rng.random(n) < .05, c] = np.nan
df.to_csv("loan_practice.csv", index=False)
print(df.shape, "approval rate:", df.approved.mean().round(2))
