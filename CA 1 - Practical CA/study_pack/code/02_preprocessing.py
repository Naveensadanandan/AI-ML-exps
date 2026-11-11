# ============ 02  PREPROCESSING + FEATURE SELECTION ============

# %% Load data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("patients.csv")
TARGET = "heart_disease"

# %% 1. Drop duplicates and ID columns (an ID has nothing to learn from)
df = df.drop_duplicates()
df = df.drop(columns=["patient_id"])

# %% 2. Remove impossible values (outliers that are data errors)
df = df[df["blood_pressure"] < 250]

# %% 3. Fill missing values
df["bmi"] = df["bmi"].fillna(df["bmi"].median())                    # numeric -> median
df["cholesterol"] = df["cholesterol"].fillna(df["cholesterol"].median())
df["exercise"] = df["exercise"].fillna(df["exercise"].mode()[0])    # category -> most common
print(df.isnull().sum())          # should be all 0 now
# (if only a few rows are missing you can also just use: df = df.dropna())

# %% 4. Turn categories into numbers
df["smoker"] = df["smoker"].map({"no": 0, "yes": 1})                # 2 values -> 0/1
df = pd.get_dummies(df, columns=["sex", "exercise"], drop_first=True, dtype=int)   # 3+ values -> one-hot
print(df.head())

# %% 5. Feature selection A: correlation with the target
print(df.corr()[TARGET].sort_values(ascending=False))
# keep features with a clear (+ or -) correlation; values near 0 are weak

# %% 6. Split into X / y, then train / test
X = df.drop(columns=[TARGET])
y = df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)    # stratify: same class ratio in both
print(X_train.shape, X_test.shape)

# %% 7. Feature selection B: Random Forest feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importance = pd.Series(rf.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))
importance.sort_values().plot(kind="barh", title="Feature importance")

# %% 8. Keep only the useful features (optional)
keep = ["age", "blood_pressure", "cholesterol", "smoker", "bmi"]    # <-- based on cells 5 and 7
X_train = X_train[keep]
X_test = X_test[keep]

# %% 9. Scale: fit on TRAIN only, then apply to test (avoids data leakage)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# needed for: Logistic/Linear Regression, KNN, SVM, KMeans
# not needed for: Decision Tree, Random Forest
