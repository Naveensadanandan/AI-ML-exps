# ============ CLASSIFICATION: train, compare, evaluate, fairness ============

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# %% 1. Load + preprocess (same steps as 02_preprocessing.py)
df = pd.read_csv("loan_practice.csv")
df = df.drop_duplicates()
df["income"] = df["income"].fillna(df["income"].median())
df["credit_score"] = df["credit_score"].fillna(df["credit_score"].median())
df["employment"] = df["employment"].fillna(df["employment"].mode()[0])

gender = df["gender"]                 # keep the sensitive column aside for the fairness check
df = df.drop(columns=["gender"])      # do not use it as a feature
df = pd.get_dummies(df, columns=["employment", "education"], drop_first=True)

X = df.drop(columns=["approved"])
y = df["approved"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# %% 2. Model 1: Logistic Regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print("Logistic Regression accuracy:", accuracy_score(y_test, lr_pred))

# %% 3. Model 2: Decision Tree
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
print("Decision Tree accuracy:", accuracy_score(y_test, dt_pred))

# %% 4. Model 3: Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
print("Random Forest accuracy:", accuracy_score(y_test, rf_pred))

# %% 5. Cross-validation (more reliable than one split)
scores = cross_val_score(LogisticRegression(), X_train, y_train, cv=5)
print("CV scores:", scores)
print("Mean CV accuracy:", scores.mean())

# %% 6. Detailed evaluation of the best model
print(classification_report(y_test, lr_pred))   # precision, recall, f1 per class
ConfusionMatrixDisplay.from_predictions(y_test, lr_pred)
plt.show()

# %% 7. Ethics: does the model treat both genders the same?
results = pd.DataFrame({"actual": y_test, "predicted": lr_pred, "gender": gender.loc[y_test.index]})
for g in ["F", "M"]:
    group = results[results["gender"] == g]
    print(g, "accuracy:", round(accuracy_score(group["actual"], group["predicted"]), 3),
          "| approval rate:", round(group["predicted"].mean(), 3))
# Big difference in approval rate = possible bias, even though gender was removed
# (other columns like income can act as a "proxy" for gender)
