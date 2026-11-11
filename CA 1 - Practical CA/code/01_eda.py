# ===================== EDA TEMPLATE =====================
# Change the file name and target column, then run cell by cell.

# %% Imports and load data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("loan_practice.csv")   # <-- change
target = "approved"                      # <-- change

# %% 1. First look
print(df.shape)          # (rows, columns)
print(df.head())         # first 5 rows
df.info()                # column types + non-null counts

# %% 2. Summary statistics
print(df.describe())                    # numeric columns
print(df.describe(include="object"))    # text / category columns

# %% 3. Missing values and duplicates
print(df.isnull().sum())
print("Duplicates:", df.duplicated().sum())

# %% 4. Target column
print(df[target].value_counts())   # classification: are classes balanced?
sns.countplot(x=target, data=df)   # regression: use sns.histplot(df[target])
plt.show()

# %% 5. Distribution of numeric columns
df.hist(figsize=(10, 8))
plt.show()

# %% 6. Outliers (dots outside the box)
sns.boxplot(data=df[["age", "credit_score"]])   # <-- pick your numeric columns
plt.show()

# %% 7. Correlation between numeric columns
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

# %% 8. Category column vs target
sns.countplot(x="employment", hue=target, data=df)   # <-- pick a category column
plt.show()

# %% 9. Write your findings (in markdown)
# - Size of data, types of columns
# - Which columns have missing values, how many duplicates
# - Is the target balanced? (classification) / skewed? (regression)
# - Which features are most correlated with the target
# - Any outliers, and what you will do with them
