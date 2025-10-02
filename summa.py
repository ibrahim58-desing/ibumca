import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load dataset (first 2 features for 2D visualization)
iris = load_iris()
X = iris.data[:, :2]  

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Visualization
plt.figure(figsize=(8,6))

for i in range(3):
    plt.scatter(X_scaled[labels==i, 0], X_scaled[labels==i, 1], label=f'Cluster {i}')

# Plot centroids
plt.scatter(centroids[:,0], centroids[:,1], s=200, c='black', marker='X', label='Centroids')

plt.xlabel('Sepal Length (scaled)')
plt.ylabel('Sepal Width (scaled)')
plt.title('K-Means Clustering of Iris Dataset')
plt.legend()
plt.show()