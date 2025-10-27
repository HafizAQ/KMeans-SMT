# src/baselines.py
# src/baselines.py --Vanilla KMeans & Projected-Means KMeans (for ablations)
import numpy as np
from sklearn.cluster import KMeans
from typing import List, Tuple

def vanilla_kmeans(X: np.ndarray, k: int, random_state: int = 7):
    km = KMeans(n_clusters=k, n_init="auto", random_state=random_state)
    labels = km.fit_predict(X)
    return km.cluster_centers_.astype(float), labels

def _project_to_bounds(x: np.ndarray, bounds: List[Tuple[float,float]]):
    y = x.copy()
    for j, (lo, hi) in enumerate(bounds):
        y[j] = min(max(y[j], lo), hi)
    return y

def projected_means_kmeans(X: np.ndarray,
                           k: int,
                           bounds: List[Tuple[float,float]],
                           max_iter: int = 100,
                           tol: float = 1e-4,
                           random_state: int = 7):
    """Axis-wise projection of centroids after each mean update (not SMT)."""
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=k, replace=False)
    C = np.array([_project_to_bounds(X[i], bounds) for i in idx], dtype=float)

    for _ in range(max_iter):
        dists = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        labels = np.argmin(dists, axis=1)
        newC = C.copy()
        for j in range(k):
            pts = X[labels == j]
            if len(pts) == 0:
                ridx = rng.integers(0, len(X))
                newC[j] = _project_to_bounds(X[ridx], bounds)
            else:
                newC[j] = _project_to_bounds(pts.mean(axis=0), bounds)
        move = np.linalg.norm(newC - C)
        C = newC
        if move < tol:
            break
    return C.astype(float), labels

def penalty_kmeans(X, k, weight=0.1, random_state=7, max_iter=100):
    """Vanilla KMeans with soft penalty for RPM/speed violations."""
    from sklearn.cluster import KMeans
    km = KMeans(n_clusters=k, random_state=random_state, n_init=10).fit(X)
    C = km.cluster_centers_.copy()
    for i in range(max_iter):
        L = km.predict(X)
        newC = np.zeros_like(C)
        for j in range(k):
            pts = X[L==j]
            if len(pts)==0: continue
            mean = pts.mean(axis=0)
            # soft penalty: if slope < 5 or >400, pull toward bounds
            slope = mean[1]/max(mean[0],1e-6)
            if slope < 5:  mean[1] += weight*(5- slope)*mean[0]
            if slope >400: mean[1] -= weight*(slope-400)*mean[0]
            newC[j] = mean
        if np.linalg.norm(newC-C)<1e-3: break
        C=newC
    return C, km.predict(X)



