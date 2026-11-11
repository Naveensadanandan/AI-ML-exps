# ============ 05  CLUSTERING: find groups when there is NO target ============

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# %% 1. Load and scale (KMeans uses distances, so scaling is a must)
df = pd.read_csv("customers.csv")
print(df.describe())
X = StandardScaler().fit_transform(df[["annual_income_k", "spending_score"]])

# %% 2. Elbow method: choose k where the curve bends
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    inertia.append(km.inertia_)
plt.plot(range(1, 11), inertia, "o-")
plt.xlabel("k (number of clusters)"); plt.ylabel("Inertia")
plt.show()

# %% 3. Silhouette score: closer to 1 = better separated clusters
for k in range(2, 9):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    print("k =", k, " silhouette =", round(silhouette_score(X, labels), 3))

# %% 4. Final model with the best k
km = KMeans(n_clusters=5, n_init=10, random_state=42)
df["cluster"] = km.fit_predict(X)
print(df["cluster"].value_counts())

# %% 5. Describe each cluster - this is your "result"
print(df.groupby("cluster").mean().round(1))
# give each cluster a name, e.g. high income + low spending -> "careful savers"

# %% 6. Plot
plt.scatter(df["annual_income_k"], df["spending_score"], c=df["cluster"], cmap="tab10")
plt.xlabel("Annual income (k)"); plt.ylabel("Spending score")
plt.show()
