# ============ REGRESSION: predict a number ============

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# %% 1. Load + preprocess
df = pd.read_csv("../../experiments/linear_regression/Housing.csv")   # <-- change
df = df.drop(columns=["Address"])       # text column, not useful
df = df.dropna()

X = df.drop(columns=["Price"])
y = df["Price"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% 2. Model 1: Linear Regression
lin = LinearRegression()
lin.fit(X_train, y_train)
lin_pred = lin.predict(X_test)

# %% 3. Model 2: Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

# %% 4. Evaluate both
for name, pred in [("Linear", lin_pred), ("Random Forest", rf_pred)]:
    print(name)
    print("  MAE :", mean_absolute_error(y_test, pred))          # average error
    print("  RMSE:", mean_squared_error(y_test, pred) ** 0.5)    # punishes big errors
    print("  R2  :", r2_score(y_test, pred))                     # 1 = perfect, 0 = useless

# %% 5. Actual vs predicted (points close to the red line = good)
plt.scatter(y_test, lin_pred, alpha=0.4)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r")
plt.xlabel("Actual"); plt.ylabel("Predicted")
plt.show()

# %% 6. Which features matter? (coefficients of linear regression)
print(pd.Series(lin.coef_, index=X.columns).sort_values())
