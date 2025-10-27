import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Simulated example data for RPM vs Speed (v, r, t) with some noise
np.random.seed(42)
n_samples = 1000
v = np.concatenate([np.random.normal(40, 10, 500), np.random.normal(20, 5, 500)])  # speed (km/h)
r = np.concatenate([np.random.normal(800, 100, 500), np.random.normal(600, 50, 500)])  # RPM
t = np.concatenate([np.random.normal(50, 10, 500), np.random.normal(70, 10, 500)])  # Throttle

# Simulate data points
data = np.vstack([v, r, t]).T

# Constraints (for visualization purposes)
idle_speed_kmh = 3.0  # Idle speed in km/h
gear_bands = [(0.1, 1.0), (0.5, 2.0), (2.0, 3.5), (3.5, 5.0)]  # RPM/Speed slope bands

# 1. KMeans-SMT: Fit with constraints (Simulated as we don't have the actual solver)
def kmeans_smt(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    centroids = kmeans.cluster_centers_
    return centroids

# 2. Projected KMeans: Force centroids into feasible region (simulate projection)
def projected_kmeans(data, n_clusters=3):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    centroids = kmeans.cluster_centers_

    # Simulate projection by pushing centroids to the closest bounds of the feasible region
    for i in range(centroids.shape[0]):
        # If the centroid is outside the gear bands, we move it to the nearest boundary
        for j, (lo, hi) in enumerate(gear_bands):
            if centroids[i, 1] < lo * centroids[i, 0]:
                centroids[i, 1] = lo * centroids[i, 0]  # Project it to the lower bound
            elif centroids[i, 1] > hi * centroids[i, 0]:
                centroids[i, 1] = hi * centroids[i, 0]  # Project it to the upper bound
    return centroids

# Fit the models
centroids_smt = kmeans_smt(data, 3)
centroids_projected = projected_kmeans(data, 3)

# Plotting
plt.figure(figsize=(10, 6))

# Plot the data points
plt.scatter(data[:, 0], data[:, 1], c='gray', alpha=0.5, label="Data Points")

# Plot KMeans-SMT centroids (with constraints)
plt.scatter(centroids_smt[:, 0], centroids_smt[:, 1], c='blue', marker='x', label="KMeans-SMT Centroids", s=100)

# Plot Projected KMeans centroids (forced within feasible region)
plt.scatter(centroids_projected[:, 0], centroids_projected[:, 1], c='red', marker='o', label="Projected KMeans Centroids", s=100)

# Add boundaries for gear bands for visualization
for lo, hi in gear_bands:
    plt.axvline(x=lo, color='green', linestyle='--', alpha=0.5)
    plt.axvline(x=hi, color='green', linestyle='--', alpha=0.5)

plt.xlabel("Speed (km/h)")
plt.ylabel("RPM")
plt.title("Cluster Centroids: KMeans-SMT vs Projected KMeans")
plt.legend()
plt.show()



