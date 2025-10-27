# src/extra_plots.py
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import List, Tuple


def save_gear_bands_overlay(out_path: str, bands: List[Tuple[float,float]], vmax_ratio: float = 400.0):
    """Legend card image showing the learned gear ratio bands (rpm/kmh)."""
    plt.figure(figsize=(5,2.6))
    y = 1.0
    for i,(lo,hi) in enumerate(bands,1):
        plt.plot([lo,hi],[y,y], linewidth=6)
        plt.text(hi+3,y, f"Gear {i}: [{lo:.1f}, {hi:.1f}] rpm/kmh", va="center")
        y -= 0.15
    plt.xlim(0, vmax_ratio); plt.ylim(0,1.1)
    plt.yticks([]); plt.xlabel("rpm per km/h (slope)")
    plt.title("Learned Gear Ratio Bands")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def save_speed_rpm_density(X, bands, out_path: str, idle_speed: float = 3.0):
    """Speed–RPM density with idle band and gear bands overlay."""
    v, r = X[:,0], X[:,1]
    plt.figure(figsize=(6,4))
    plt.hist2d(v, r, bins=200, cmap="viridis", norm=matplotlib.colors.LogNorm())
    # idle band
    plt.axvline(idle_speed, color="white", linestyle="--", linewidth=1.5, alpha=0.9, label="idle speed")
    # gear bands: draw central slope lines
    xs = np.linspace(idle_speed+0.1, v.max(), 200)
    for i,(lo,hi) in enumerate(bands,1):
        mid = 0.5*(lo+hi)
        plt.plot(xs, mid*xs, linewidth=1.2, alpha=0.9, label=(f"gear {i}" if i<=5 else None))
    plt.xlabel("Speed (km/h)"); plt.ylabel("RPM")
    plt.title("Speed–RPM Density with Gear Manifolds")
    plt.legend(loc="upper left", fontsize=8)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def save_slope_histogram(X, bands, out_path: str, idle_speed: float = 3.0):
    """Histogram of rpm/speed slopes with band shading."""
    v = X[:,0]; r = X[:,1]
    mask = v > idle_speed
    s = (r[mask] / np.maximum(v[mask], 1e-6)).clip(0, 500)
    plt.figure(figsize=(6,4))
    plt.hist(s, bins=200, alpha=0.8)
    for (lo,hi) in bands:
        plt.axvspan(lo, hi, alpha=0.15)
    plt.xlabel("rpm per km/h"); plt.ylabel("count")
    plt.title("Slope (RPM / Speed) Distribution with Bands")
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()

def save_violation_bars(agg_smt, agg_van, agg_proj, out_path: str):
    """Stacked bars of violation counts per method."""
    keys_bounds = ["v_low","v_high","r_low","r_high","t_low","t_high"]
    methods = ["Vanilla","Projected","KMeans–SMT"]
    vals = [
        [agg_van[k] for k in keys_bounds] + [agg_van["idle_rule"], agg_van["gear_rule"]],
        [agg_proj[k] for k in keys_bounds] + [agg_proj["idle_rule"], agg_proj["gear_rule"]],
        [agg_smt[k] for k in keys_bounds] + [agg_smt["idle_rule"], agg_smt["gear_rule"]],
    ]
    labels = keys_bounds + ["idle_rule","gear_rule"]
    idx = np.arange(len(methods))
    plt.figure(figsize=(7,3.5))
    bottom = np.zeros(len(methods))
    colors = [None]*len(labels)
    for i,label in enumerate(labels):
        col = None
        bar = [vals[j][i] for j in range(len(methods))]
        plt.bar(idx, bar, bottom=bottom, label=label)
        bottom += np.array(bar)
    plt.xticks(idx, methods); plt.ylabel("violations (count)")
    plt.title("Constraint Violations per Method")
    plt.legend(fontsize=8, ncol=3)
    plt.tight_layout(); plt.savefig(out_path, dpi=160); plt.close()


def save_centroid_table_image(centers, names, save_path,
                              idle_band=None, gear_bands=None, idle_speed=3.0):
    """
    Save a centroid table image with constraint interpretation columns.
    Adds: 'Regime', 'Constraint OK?' columns beside centroid values.
    """
    df = pd.DataFrame(centers, columns=names)
    regimes, checks = [], []

    for _, row in df.iterrows():
        v, r, t = row["speed_kmh"], row["rpm"], row["throttle_pct"]
        if v <= idle_speed:
            regimes.append("IDLE")
            ok = (idle_band is not None) and (idle_band[0] <= r <= idle_band[1])
            checks.append("✓" if ok else "✗")
        else:
            slope = r / max(v, 1e-6)
            idx = None
            if gear_bands:
                for i, (lo, hi) in enumerate(gear_bands):
                    if lo <= slope <= hi:
                        idx = i + 1
                        break
            regimes.append(f"GEAR#{idx or '?'}")
            checks.append("✓" if idx else "✗")

    df["Regime"] = regimes
    df["Constraint OK?"] = checks

    fig, ax = plt.subplots(figsize=(7, 0.5 * len(df) + 1))
    ax.axis("off")
    tbl = ax.table(cellText=df.round(2).values,
                   colLabels=df.columns,
                   loc="center",
                   cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.1, 1.2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_centroid_trajectories(traj_list, names, out_path: str):
    """Plot per-iteration centroid trajectories (speed,rpm only) for interpretability."""
    # traj_list: list of lists, where each list contains the centroids' trajectory over iterations.
    plt.figure(figsize=(6,4))

    # Ensure that `traj_list` contains the proper structure (list of lists or arrays)
    for i, tr in enumerate(traj_list):
        if isinstance(tr, list) and len(tr) > 0:
            # Convert tr into a numpy array to access by index
            tr = np.array(tr)
            plt.plot(tr[:, 0], tr[:, 1], marker=".", linewidth=1, alpha=0.8, label=f"C{i}")
    
    plt.xlabel(names[0])
    plt.ylabel(names[1])
    plt.title("Centroid Trajectories (Speed vs RPM)")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()





