# Run this ONCE. It creates 3 small practice CSV files:
#   patients.csv  -> classification  (target: heart_disease, 0/1)
#   houses.csv    -> regression      (target: price)
#   customers.csv -> clustering      (no target)
# The data is made up, with some "messy" parts on purpose
# (missing values, duplicates, outliers) so you can practise cleaning it.
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

# ---------- patients.csv ----------
n = 800
p = pd.DataFrame({
    "patient_id": range(1, n + 1),
    "age": rng.integers(25, 80, n),
    "sex": rng.choice(["F", "M"], n, p=[0.4, 0.6]),
    "bmi": rng.normal(27, 4, n).round(1),
    "blood_pressure": rng.normal(130, 15, n).round(0),
    "cholesterol": rng.normal(210, 35, n).round(0),
    "smoker": rng.choice(["yes", "no"], n, p=[0.3, 0.7]),
    "exercise": rng.choice(["low", "medium", "high"], n),
})
score = (0.05 * (p.age - 50) + 0.04 * (p.blood_pressure - 130) + 0.02 * (p.cholesterol - 210)
         + 1.0 * (p.smoker == "yes") - 0.6 * (p.exercise == "high") + 0.1 * (p.bmi - 27) - 0.8)
p["heart_disease"] = (rng.random(n) < 1 / (1 + np.exp(-score))).astype(int)
p.loc[rng.random(n) < 0.06, "bmi"] = np.nan
p.loc[rng.random(n) < 0.05, "cholesterol"] = np.nan
p.loc[rng.random(n) < 0.04, "exercise"] = np.nan
p.loc[rng.choice(n, 5, replace=False), "blood_pressure"] = 300      # data-entry errors (outliers)
p = pd.concat([p, p.sample(15, random_state=1)])                   # 15 duplicate rows
p.to_csv("patients.csv", index=False)

# ---------- houses.csv ----------
n = 600
h = pd.DataFrame({
    "size_sqft": rng.normal(1500, 400, n).clip(500).round(0),
    "bedrooms": rng.integers(1, 6, n),
    "age_years": rng.integers(0, 60, n),
    "distance_city_km": rng.uniform(1, 40, n).round(1),
    "location": rng.choice(["urban", "suburban", "rural"], n),
})
h["price"] = (150 * h.size_sqft + 10000 * h.bedrooms - 1500 * h.age_years - 3000 * h.distance_city_km
              + h.location.map({"urban": 60000, "suburban": 30000, "rural": 0})
              + rng.normal(0, 25000, n)).round(0)
h.loc[rng.random(n) < 0.05, "size_sqft"] = np.nan
h.to_csv("houses.csv", index=False)

# ---------- customers.csv ----------
centers = [(30, 20), (30, 80), (70, 50), (100, 20), (100, 80)]   # (income in k, spending score)
rows = []
for inc, spend in centers:
    for _ in range(60):
        rows.append([round(rng.normal(inc, 8), 1), round(rng.normal(spend, 7), 1), int(rng.integers(18, 70))])
c = pd.DataFrame(rows, columns=["annual_income_k", "spending_score", "age"])
c.to_csv("customers.csv", index=False)

print("Created patients.csv", p.shape, "| houses.csv", h.shape, "| customers.csv", c.shape)
