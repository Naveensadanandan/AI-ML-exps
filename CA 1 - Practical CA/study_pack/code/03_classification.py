# ============ 03  CLASSIFICATION: train, compare, evaluate, fairness ============

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, ConfusionMatrixDisplay

# %% 1. Load + preprocess (same steps as 02_preprocessing.py)
df = pd.read_csv("patients.csv")
df = df.drop_duplicates().drop(columns=["patient_id"])
df = df[df["blood_pressure"] < 250]
df["bmi"] = df["bmi"].fillna(df["bmi"].median())
df["cholesterol"] = df["cholesterol"].fillna(df["cholesterol"].median())
df["exercise"] = df["exercise"].fillna(df["exercise"].mode()[0])
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})
df = pd.get_dummies(df, columns=["sex", "exercise"], drop_first=True, dtype=int)

X = df.drop(columns=["heart_disease"])
y = df["heart_disease"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

# %% 2. Baseline: always guess the most common class. A real model must beat this.
print("Baseline accuracy:", round(y_test.value_counts(normalize=True).max(), 3))

# %% 3. Train several models and compare
models = {
    "Logistic Regression": LogisticRegression(),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    pred = model.predict(X_test_s)
    print(f"{name:20}  accuracy={accuracy_score(y_test, pred):.3f}  f1={f1_score(y_test, pred):.3f}")

# %% 4. Cross-validation: 5 different splits, more reliable than one split
scores = cross_val_score(LogisticRegression(), X_train_s, y_train, cv=5)
print("CV scores:", scores.round(3), " mean:", scores.mean().round(3))

# %% 5. Detailed evaluation of the best model
best = models["Logistic Regression"]       # <-- pick the best from cell 3
pred = best.predict(X_test_s)
print(classification_report(y_test, pred))
ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.show()

# %% 6. Try a different hyperparameter (simple tuning)
# (trees don't need scaling, so the unscaled X_train is fine here)
for depth in [2, 3, 4, 5, 8, None]:
    tree = DecisionTreeClassifier(max_depth=depth, random_state=42).fit(X_train, y_train)
    print("max_depth", depth, " train", round(tree.score(X_train, y_train), 3),
          " test", round(tree.score(X_test, y_test), 3))
# train much higher than test = overfitting

# %% 7. Explainability: which features push the prediction up or down?
coef = pd.Series(best.coef_[0], index=X.columns).sort_values()
print(coef)            # positive = raises the chance of heart disease

# %% 8. Ethics / fairness: is the model equally good for women and men?
test = X_test.copy()
test["actual"] = y_test
test["pred"] = pred
for name, group in test.groupby("sex_M"):
    label = "Male" if name == 1 else "Female"
    print(label, " n =", len(group),
          " accuracy =", round(accuracy_score(group["actual"], group["pred"]), 3),
          " predicted positive rate =", round(group["pred"].mean(), 3))
# a big gap between groups = possible bias -> mention it in your ethics section
