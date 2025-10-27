
#src/metrics_plot.py -- more metrics + quick plots
# src/metrics_plot.py
import matplotlib
matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

def compute_metrics(X: np.ndarray, labels: np.ndarray):
    uniq = np.unique(labels)
    if len(uniq) < 2:
        return {"silhouette": np.nan, "dbi": np.nan, "ch": np.nan}
    sil = silhouette_score(X, labels)
    dbi = davies_bouldin_score(X, labels)
    ch  = calinski_harabasz_score(X, labels)
    return {"silhouette": sil, "dbi": dbi, "ch": ch}

def plot_clusters_2d(X, labels, centers, x_idx=0, y_idx=1,
                     feat_names=("speed_kmh","rpm"),
                     constraints=None,
                     gear_bands=None,
                     idle_band=None,
                     idle_speed=3.0,
                     title="Clusters",
                     save_path=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7,5))
    plt.scatter(X[:,x_idx], X[:,y_idx], c=labels, s=6, alpha=0.35)
    plt.scatter(centers[:,x_idx], centers[:,y_idx], c="black",
                marker="X", s=120, edgecolor="white", linewidth=1.2, label="centroids")

    # show idle/gear constraints
    if feat_names == ("speed_kmh","rpm"):
        xs = np.linspace(constraints[0][0], constraints[0][1], 300)
        if gear_bands:
            for (lo, hi) in gear_bands:
                plt.fill_between(xs, lo*xs, hi*xs, color="gray", alpha=0.08)
        if idle_band:
            plt.axvline(idle_speed, color="orange", linestyle="--", alpha=0.7)
            plt.axhspan(idle_band[0], idle_band[1], color="orange", alpha=0.1)

    plt.xlabel(feat_names[0]); plt.ylabel(feat_names[1])
    plt.title(title); plt.legend(loc="best"); plt.tight_layout()
    plt.savefig(save_path, dpi=160); plt.close()


