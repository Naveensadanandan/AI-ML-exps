# ============ 04  REGRESSION: predict a number ============

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %% 1. Load + quick EDA
df = pd.read_csv("houses.csv")
print(df.head())
print(df.isnull().sum())
df["price"].hist()            # is the target skewed?
plt.show()

# %% 2. Preprocess
df["size_sqft"] = df["size_sqft"].fillna(df["size_sqft"].median())
df = pd.get_dummies(df, columns=["location"], drop_first=True, dtype=int)

X = df.drop(columns=["price"])
y = df["price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)   # no stratify

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# %% 3. Train and compare models
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    mae = mean_absolute_error(y_test, pred)                # average error, in target units
    rmse = mean_squared_error(y_test, pred) ** 0.5         # punishes big errors more
    r2 = r2_score(y_test, pred)                            # 1 = perfect, 0 = no better than the mean
    print(f"{name:18}  MAE={mae:,.0f}  RMSE={rmse:,.0f}  R2={r2:.3f}")

# %% 4. Actual vs predicted (points near the red line = good)
pred = models["Linear Regression"].predict(X_test_s)
plt.scatter(y_test, pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r")
plt.xlabel("Actual price"); plt.ylabel("Predicted price")
plt.show()

# %% 5. Which features matter? (linear regression coefficients, on scaled data)
print(pd.Series(models["Linear Regression"].coef_, index=X.columns).sort_values())
