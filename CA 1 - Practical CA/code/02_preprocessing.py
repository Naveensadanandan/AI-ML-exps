# ============ PREPROCESSING + FEATURE SELECTION ============

# %% Imports and load data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("loan_practice.csv")   # <-- change
target = "approved"                      # <-- change

# %% 1. Remove duplicates and useless columns
df = df.drop_duplicates()
# df = df.drop(columns=["id", "name"])   # IDs / names have no pattern to learn

# %% 2. Fill missing values
# numeric column -> median (not affected by outliers)
df["income"] = df["income"].fillna(df["income"].median())
df["credit_score"] = df["credit_score"].fillna(df["credit_score"].median())
# category column -> most common value (mode)
df["employment"] = df["employment"].fillna(df["employment"].mode()[0])
print(df.isnull().sum())

# %% 3. Encode categories into numbers (one-hot encoding)
df = pd.get_dummies(df, columns=["employment", "education", "gender"], drop_first=True)
print(df.head())

# %% 4. Split into X (features) and y (target), then train / test
X = df.drop(columns=[target])
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# stratify=y keeps the same class ratio in train and test (classification only)

# %% 5. Scale numeric features
# fit on TRAIN only, then apply to test (otherwise test info "leaks" into training)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% 6. Feature selection - method A: correlation with target
print(df.corr(numeric_only=True)[target].sort_values(ascending=False))

# %% 7. Feature selection - method B: SelectKBest (keep the k best features)
selector = SelectKBest(f_classif, k=5)       # regression: use f_regression
selector.fit(X_train, y_train)
print("Selected:", list(X_train.columns[selector.get_support()]))

# %% 8. Feature selection - method C: Random Forest feature importance
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
importance = pd.Series(rf.feature_importances_, index=X_train.columns)
print(importance.sort_values(ascending=False))
