# ===================== 01  EDA TEMPLATE =====================
# For a new dataset: change FILE and TARGET, then run each cell (# %%).

# %% Load data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FILE = "patients.csv"        # <-- change
TARGET = "heart_disease"     # <-- change

df = pd.read_csv(FILE)

# %% 1. First look
print(df.shape)       # (rows, columns)
print(df.head())      # first 5 rows
df.info()             # column names, types, non-null counts

# %% 2. Summary statistics
print(df.describe())                   # numeric: mean, min, max ...
print(df.describe(include="object"))   # text/category: unique values, most common

# %% 3. Missing values and duplicates
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())

# %% 4. Target column
print(df[TARGET].value_counts())    # classification: are the classes balanced?
sns.countplot(x=TARGET, data=df)    # regression: use sns.histplot(df[TARGET])
plt.show()

# %% 5. Distributions of numeric columns
df.hist(figsize=(10, 8))
plt.tight_layout()
plt.show()

# %% 6. Outliers (dots outside the whiskers)
sns.boxplot(data=df[["blood_pressure", "cholesterol"]])   # <-- your numeric columns
plt.show()

# %% 7. Correlation heatmap (numeric columns only)
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm")
plt.show()

# %% 8. A category column vs the target
sns.countplot(x="smoker", hue=TARGET, data=df)   # <-- your category column
plt.show()

# %% 9. A numeric column vs the target
sns.boxplot(x=TARGET, y="age", data=df)          # <-- your numeric column
plt.show()

# %% 10. Write findings in a markdown cell, e.g.
# - 815 rows, 9 columns (6 numeric, 3 categorical)
# - Missing: bmi 52, cholesterol 43, exercise 35  -> will fill
# - 15 duplicate rows -> will drop
# - Target fairly balanced (57% / 43%) -> accuracy is OK, also report F1
# - blood_pressure has values of 300 -> data errors, will remove
# - age and blood_pressure are most correlated with the target
