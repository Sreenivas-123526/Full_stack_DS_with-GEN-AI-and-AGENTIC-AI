import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


# ---------------------------
# 1. Create sample dataset
# ---------------------------

X, y = make_blobs(
    n_samples=300,
    centers=4,
    n_features=5,
    random_state=42
)

print("Original shape:", X.shape)


# ---------------------------
# 2. Standardize the data
# ---------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)


# ---------------------------
# 3. Apply PCA
# Reduce 5 features -> 2 components
# ---------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)


print("After PCA shape:", X_pca.shape)


# ---------------------------
# 4. Check explained variance
# ---------------------------

print(
    "Explained variance ratio:",
    pca.explained_variance_ratio_
)


print(
    "Total variance captured:",
    sum(pca.explained_variance_ratio_)
)


# ---------------------------
# 5. Apply K-Means clustering
# ---------------------------

kmeans = KMeans(
    n_clusters=4,
    random_state=42
)

clusters = kmeans.fit_predict(X_pca)


# ---------------------------
# 6. Plot clusters
# ---------------------------

plt.figure(figsize=(8,6))

plt.scatter(
    X_pca[:,0],
    X_pca[:,1],
    c=clusters,
    s=50
)


plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    marker="X"
)


plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("K-Means Clustering after PCA")

plt.show()