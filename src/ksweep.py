# src/ksweep.py
import os, json, time
import numpy as np
import pandas as pd
from typing import Iterable
from datasets import load_folder, make_X, learn_idle_band_and_gear_bands
from kmeans_smt import KMeansSMT
from baselines import vanilla_kmeans, projected_means_kmeans
from metrics_plot import compute_metrics, plot_clusters_2d
from audit import audit_centers

def _metrics_row(X, labels):
    m = compute_metrics(X, labels)
    return dict(silhouette=m["silhouette"], dbi=m["dbi"], ch=m["ch"])

def run_sweep(data_dir: str,
              ks: Iterable[int] = range(3, 11),
              max_rows: int = 40000,
              out_dir: str = "outputs",
              include_baselines: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    df = load_folder(data_dir)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=7).reset_index(drop=True)

    learned = learn_idle_band_and_gear_bands(df, n_gears=5)
    idle_lo, idle_hi = learned["idle_rpm"]
    gear_bands = learned["gear_bands"]
    X = make_X(df)

    bounds = [(0.0,130.0), (0.0,6000.0), (0.0,100.0)]
    idle_speed = 3.0
    safety_margin = (2.0, 50.0, 2.0)

    rows = []
    for k in ks:
        rec_base = {"k": k}

        smt = KMeansSMT(
            n_clusters=k, domain_bounds=bounds,
            idle_rpm_band=(idle_lo,idle_hi), gear_bands=gear_bands,
            idle_speed_kmh=idle_speed, safety_margin=safety_margin,
            random_state=7, z3_timeout_ms=400, verbose=False,
            prefer_l1=True, strict_eps=1e-3, damping_eta=0.65, max_band_trials=1
        )
        t0 = time.time(); smt.fit(X); t1 = time.time()
        smt_m = _metrics_row(X, smt.labels_)
        smt_a = audit_centers(smt.centers_, bounds, idle_speed, (idle_lo,idle_hi), gear_bands)
        rows.append({**rec_base, "method":"KMeans-SMT",
                     "silhouette":smt_m["silhouette"], "dbi":smt_m["dbi"], "ch":smt_m["ch"],
                     "time_s": t1-t0,
                     "viol_idle": smt_a["aggregate"]["idle_rule"],
                     "viol_gear": smt_a["aggregate"]["gear_rule"],
                     "viol_bounds": sum(smt_a["aggregate"][k] for k in ["v_low","v_high","r_low","r_high","t_low","t_high"]),
                     })

        if include_baselines:
            t0 = time.time(); vC, vL = vanilla_kmeans(X, k, 7); t1 = time.time()
            vm = _metrics_row(X, vL)
            va = audit_centers(vC, bounds, idle_speed, (idle_lo,idle_hi), gear_bands)
            rows.append({**rec_base, "method":"Vanilla",
                         "silhouette":vm["silhouette"], "dbi":vm["dbi"], "ch":vm["ch"],
                         "time_s": t1-t0,
                         "viol_idle": va["aggregate"]["idle_rule"],
                         "viol_gear": va["aggregate"]["gear_rule"],
                         "viol_bounds": sum(va["aggregate"][k] for k in ["v_low","v_high","r_low","r_high","t_low","t_high"]),
                         })

            t0 = time.time(); pC, pL = projected_means_kmeans(X, k, bounds, random_state=7); t1 = time.time()
            pm = _metrics_row(X, pL)
            pa = audit_centers(pC, bounds, idle_speed, (idle_lo,idle_hi), gear_bands)
            rows.append({**rec_base, "method":"Projected",
                         "silhouette":pm["silhouette"], "dbi":pm["dbi"], "ch":pm["ch"],
                         "time_s": t1-t0,
                         "viol_idle": pa["aggregate"]["idle_rule"],
                         "viol_gear": pa["aggregate"]["gear_rule"],
                         "viol_bounds": sum(pa["aggregate"][k] for k in ["v_low","v_high","r_low","r_high","t_low","t_high"]),
                         })

        plot_base = os.path.join(out_dir, f"k{k}_smt")
        plot_clusters_2d(X, smt.labels_, smt.centers_, 0, 1,
                         feat_names=("speed_kmh","rpm"),
                         constraints=bounds,
                         title=f"KMeans–SMT (k={k}) Speed–RPM",
                         save_path=plot_base+"_speed_rpm.png")
        plot_clusters_2d(X, smt.labels_, smt.centers_, 0, 2,
                         feat_names=("speed_kmh","throttle_pct"),
                         constraints=bounds,
                         title=f"KMeans–SMT (k={k}) Speed–Throttle",
                         save_path=plot_base+"_speed_throttle.png")

    csv_path = os.path.join(out_dir, "ksweep_summary.csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    with open(os.path.join(out_dir, "ksweep_summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    print(f"[INFO] Saved sweep summary:\n - {csv_path}")


