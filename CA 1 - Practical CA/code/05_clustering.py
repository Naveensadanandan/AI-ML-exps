# ============ CLUSTERING: find groups when there is NO target ============

# %% Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_iris

# %% 1. Load data (numeric columns only) and scale
df = load_iris(as_frame=True).frame.drop(columns="target")   # <-- replace with pd.read_csv(...)
df = df.dropna()
X = StandardScaler().fit_transform(df)    # KMeans uses distance, so scaling is a must

# %% 2. Elbow method: pick k where the curve bends
inertia = []
for k in range(1, 10):
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X)
    inertia.append(km.inertia_)
plt.plot(range(1, 10), inertia, "o-")
plt.xlabel("k"); plt.ylabel("Inertia")
plt.show()

# %% 3. Silhouette score: closer to 1 = better separated clusters
for k in range(2, 7):
    labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(X)
    print(k, round(silhouette_score(X, labels), 3))

# %% 4. Final model
km = KMeans(n_clusters=3, n_init=10, random_state=42)
df["cluster"] = km.fit_predict(X)
print(df["cluster"].value_counts())

# %% 5. Describe each cluster (this is your "result")
print(df.groupby("cluster").mean())

# %% 6. Plot two features coloured by cluster
plt.scatter(df["petal length (cm)"], df["petal width (cm)"], c=df["cluster"])
plt.xlabel("petal length"); plt.ylabel("petal width")
plt.show()
