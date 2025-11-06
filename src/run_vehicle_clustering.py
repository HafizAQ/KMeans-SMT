# src/run_vehicle_clustering.py
# End-to-end experiment driver for:
#   "KMeans–SMT: Constrained Clustering for Intelligent Vehicle Diagnostics Data"
#
# What this script does:
#   • Loads OBD-II data and (optionally) downsamples (configurable via MAX_ROWS env).
#   • Learns idle RPM band + gear slope bands from data.
#   • Fits four methods: SMT, Vanilla, Projected, and (optional) Penalty KMeans.
#   • Audits every method against physical constraints; prints + plots comparisons.
#   • Produces intuitive visuals: cluster overlays, centroid tables, trajectories (SMT),
#     density, gear-band card, slope histogram, violation bars.
#   • Runs a k-sweep (elbow + silhouette) for model selection transparency.


# src/run_vehicle_clustering.py
# End-to-end experiment driver for:
#   "KMeans–SMT: Constrained Clustering for Intelligent Vehicle Diagnostics Data"

import os
import time
import numpy as np
import warnings

from datasets import load_folder, make_X, learn_idle_band_and_gear_bands
from kmeans_smt import KMeansSMT
from baselines import vanilla_kmeans, projected_means_kmeans
try:
    from baselines import penalty_kmeans
    _HAS_PENALTY = True
except Exception:
    _HAS_PENALTY = False

from audit import audit_centers
from metrics_plot import compute_metrics, plot_clusters_2d
from extra_plots import (
    save_speed_rpm_density,
    save_slope_histogram,
    save_violation_bars,
    save_centroid_table_image,
    save_gear_bands_overlay,
    save_centroid_trajectories,
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans

def ksweep_vanilla_elbow_silhouette(X, k_values, out_dir):
    inertias, sils = [], []
    print("[INFO] Running vanilla k-sweep for elbow/silhouette:", k_values, flush=True)
    for k in k_values:
        km = KMeans(n_clusters=k, random_state=7, n_init=10)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        try:
            sil = silhouette_score(X, labels)
        except Exception:
            sil = np.nan
        sils.append(sil)
        print(f"  k={k}: inertia={km.inertia_:.1f} | silhouette={sil if np.isfinite(sil) else float('nan'):.3f}",
              flush=True)

    elbow_path = os.path.join(out_dir, "elbow_vanilla.png")
    plt.figure(figsize=(6,4))
    plt.plot(k_values, inertias, marker="o")
    plt.title("Elbow (Vanilla KMeans)")
    plt.xlabel("k"); plt.ylabel("Inertia (SSE)")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(elbow_path, dpi=160); plt.close()

    sil_path = os.path.join(out_dir, "silhouette_vanilla.png")
    plt.figure(figsize=(6,4))
    plt.plot(k_values, sils, marker="o")
    plt.title("Silhouette (Vanilla KMeans)")
    plt.xlabel("k"); plt.ylabel("Mean silhouette")
    plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(sil_path, dpi=160); plt.close()

    print(f"[INFO] Saved model-selection plots:\n - {elbow_path}\n - {sil_path}", flush=True)

def fit_and_report_method(name, X, k, out_dir, domain_bounds, idle_speed, idle_band, gear_bands,
                          fit_fn, centers_labels=None, plot_speed_throttle=True, save_table=True):
    t0 = time.time()
    if centers_labels is None:
        C, L = fit_fn(X, k)
    else:
        C, L = centers_labels
    t1 = time.time()

    metrics = compute_metrics(X, L)
    audit = audit_centers(
        C, domain_bounds,
        idle_speed_kmh=idle_speed,
        idle_band=idle_band,
        gear_bands=gear_bands
    )

    p_sp_rpm = os.path.join(out_dir, f"{name}_clusters_speed_rpm.png")
    plot_clusters_2d(
        X, L, C,
        x_idx=0, y_idx=1,
        feat_names=("speed_kmh", "rpm"),
        constraints=domain_bounds,
        gear_bands=gear_bands,
        idle_band=idle_band,
        idle_speed=idle_speed,
        title=f"{name} (Speed vs RPM)",
        save_path=p_sp_rpm
    )

    p_sp_th = None
    if plot_speed_throttle:
        p_sp_th = os.path.join(out_dir, f"{name}_clusters_speed_throttle.png")
        plot_clusters_2d(
            X, L, C,
            x_idx=0, y_idx=2,
            feat_names=("speed_kmh", "throttle_pct"),
            constraints=domain_bounds,
            gear_bands=gear_bands,
            idle_band=idle_band,
            idle_speed=idle_speed,
            title=f"{name} (Speed vs Throttle)",
            save_path=p_sp_th
        )

    p_table = None
    if save_table:
        p_table = os.path.join(out_dir, f"{name}_centers_table.png")
        save_centroid_table_image(
            C,
            ["speed_kmh", "rpm", "throttle_pct"],
            p_table,
            idle_band=idle_band,
            gear_bands=gear_bands,
            idle_speed=idle_speed
        )

    print(f"[INFO] {name:>10s} | Sil: {metrics['silhouette']:.3f} | DBI: {metrics['dbi']:.3f} | CH: {metrics['ch']:.1f} | fit={t1-t0:.2f}s")
    print(f"[INFO] {name:>10s} violations: {audit['aggregate']}")
    if p_sp_rpm: print(f"[INFO] {name:>10s} plot: {p_sp_rpm}")
    if p_sp_th:  print(f"[INFO] {name:>10s} plot: {p_sp_th}")
    if p_table:  print(f"[INFO] {name:>10s} table: {p_table}")

    return C, L, metrics, audit

def main():
    warnings.filterwarnings("ignore")

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    out_dir  = os.path.join(os.path.dirname(__file__), "..", "outputs")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading CSVs from: {os.path.abspath(data_dir)}", flush=True)
    df = load_folder(data_dir)
    print(f"[INFO] Loaded rows: {len(df):,}", flush=True)

    MAX_ROWS = int(os.environ.get("MAX_ROWS", "80000"))
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=7).reset_index(drop=True)
        print(f"[INFO] Downsampled to {len(df):,} rows.", flush=True)

    learned = learn_idle_band_and_gear_bands(df, n_gears=5)
    idle_lo, idle_hi = learned["idle_rpm"]
    gear_bands = learned["gear_bands"]
    idle_speed = 3.0
    print(f"[INFO] Idle RPM band: [{idle_lo:.0f}, {idle_hi:.0f}]")
    print(f"[INFO] Gear bands (rpm/kmh): {['[%.1f, %.1f]'%(a,b) for a,b in gear_bands]}")

    X = make_X(df)
    print(f"[INFO] Feature matrix shape: {X.shape}", flush=True)

    #Consistent with preprocessing cap/lim
    domain_bounds = [
        (0.0, 130.0),  # speed_kmh 
        (0.0, 6000.0), # rpm 
        (0.0, 100.0),  # throttle %
    ]

    k_values = list(range(int(os.environ.get("K_MIN","4")), int(os.environ.get("K_MAX","9")) + 1))
    ksweep_vanilla_elbow_silhouette(X, k_values, out_dir)

    k = int(os.environ.get("K", "6"))
    print(f"[INFO] Using k={k}", flush=True)

    z3_timeout_ms = int(os.environ.get("Z3_TIMEOUT_MS", "1500"))
    safety_margin = (2.0, 50.0, 2.0)
    print(f"[INFO] k={k}, Z3 timeout={z3_timeout_ms} ms", flush=True)

    smt_kwargs = dict(
        n_clusters=k,
        domain_bounds=domain_bounds,
        idle_rpm_band=(idle_lo, idle_hi),
        gear_bands=gear_bands,
        idle_speed_kmh=idle_speed,
        safety_margin=safety_margin,
        random_state=7,
        z3_timeout_ms=z3_timeout_ms,
        verbose=True
    )
    if "extra_constraints" in KMeansSMT.__init__.__code__.co_varnames:
        smt_kwargs["extra_constraints"] = bool(int(os.environ.get("SMT_EXTRA_CONSTRAINTS","1")))

    print("[INFO] Fitting KMeans–SMT...", flush=True)
    t_smt0 = time.time()
    smt = KMeansSMT(**smt_kwargs).fit(X)
    t_smt1 = time.time()
    print(f"[INFO] SMT fit time: {t_smt1 - t_smt0:.2f}s", flush=True)

    smtC, smtL, smt_metrics, smt_audit = fit_and_report_method(
        "SMT_KMeans",
        X, k, out_dir,
        domain_bounds=domain_bounds,
        idle_speed=idle_speed,
        idle_band=(idle_lo, idle_hi),
        gear_bands=gear_bands,
        fit_fn=None,
        centers_labels=(smt.centers_, smt.labels_),
        plot_speed_throttle=True,
        save_table=True
    )

    try:
        if hasattr(smt, "_traj_"):
            p_traj = os.path.join(out_dir, "smt_centroid_trajectories.png")
            save_centroid_trajectories(smt._traj_, ["speed_kmh","rpm","throttle_pct"], p_traj)
            print(f"[INFO] SMT trajectories plot: {p_traj}", flush=True)
    except Exception as e:
        print(f"[WARN] Could not save centroid trajectories: {e}")

    vC, vL, v_metrics, v_audit = fit_and_report_method(
        "Vanilla_KMeans",
        X, k, out_dir,
        domain_bounds=domain_bounds,
        idle_speed=idle_speed,
        idle_band=(idle_lo, idle_hi),
        gear_bands=gear_bands,
        fit_fn=lambda X_, k_: vanilla_kmeans(X_, k_, random_state=7),
        plot_speed_throttle=True,
        save_table=True
    )

    pC, pL, p_metrics, p_audit = fit_and_report_method(
        "Projected_KMeans",
        X, k, out_dir,
        domain_bounds=domain_bounds,
        idle_speed=idle_speed,
        idle_band=(idle_lo, idle_hi),
        gear_bands=gear_bands,
        fit_fn=lambda X_, k_: projected_means_kmeans(X_, k_, domain_bounds, random_state=7),
        plot_speed_throttle=True,
        save_table=True
    )

    if _HAS_PENALTY:
        penC, penL, pen_metrics, pen_audit = fit_and_report_method(
            "Penalty_KMeans",
            X, k, out_dir,
            domain_bounds=domain_bounds,
            idle_speed=idle_speed,
            idle_band=(idle_lo, idle_hi),
            gear_bands=gear_bands,
            fit_fn=lambda X_, k_: penalty_kmeans(
                X_, k_, weight=float(os.environ.get("PENALTY_W","0.1")), random_state=7),
            plot_speed_throttle=True,
            save_table=True
        )
    else:
        pen_metrics, pen_audit = None, None
        print("[INFO] Penalty_KMeans baseline not found in baselines.py — skipping.", flush=True)

    # Shared contextual visuals
    save_speed_rpm_density(X, gear_bands, os.path.join(out_dir, "smt_density_speed_rpm.png"), idle_speed=idle_speed)
    save_gear_bands_overlay(os.path.join(out_dir, "smt_gear_bands_card.png"), gear_bands)
    save_slope_histogram(X, gear_bands, os.path.join(out_dir, "smt_slope_hist.png"), idle_speed=idle_speed)

    # # Violations comparison (3-way, as defined in extra_plots.save_violation_bars)
    # save_violation_bars(
    #     smt_audit["aggregate"],
    #     v_audit["aggregate"],
    #     p_audit["aggregate"],
    #     os.path.join(out_dir, "violations_by_method.png")
    # )
    # print(f"[INFO] Saved violation comparison → {os.path.join(out_dir, 'violations_by_method.png')}", flush=True)

    # # Optional: a second figure swapping in the Penalty baseline (no API change)
    # if pen_audit is not None:
    #     save_violation_bars(
    #         smt_audit["aggregate"],
    #         v_audit["aggregate"],
    #         pen_audit["aggregate"],
    #         os.path.join(out_dir, "violations_by_method_with_penalty.png")
    #     )
    #     print(f"[INFO] Saved violation comparison (Penalty) → "
    #           f"{os.path.join(out_dir, 'violations_by_method_with_penalty.png')}", flush=True)

    # === Unified violations comparison for all available methods ===
    # Ensures consistency with center-table audits by reusing the same audit dicts.

    def _safe_aggregate(audit_dict):
        # audit_centers(...) returns a dict; we use its "aggregate" mapping of rule -> count
        return (audit_dict or {}).get("aggregate", {}) if isinstance(audit_dict, dict) else {}

    # Collect methods that actually ran
    methods = []
    aggregates = []

    # Vanilla
    methods.append("Vanilla")
    aggregates.append(_safe_aggregate(v_audit))

    # Projected
    methods.append("Projected")
    aggregates.append(_safe_aggregate(p_audit))

    # Penalty (only if present)
    if pen_audit is not None:
        methods.append("Penalty")
        aggregates.append(_safe_aggregate(p_audit))

    # SMT
    methods.append("SMT")
    aggregates.append(_safe_aggregate(smt_audit))

    # Build the union of all rule names to keep bar stacks aligned across methods
    all_rules = set()
    for agg in aggregates:
        all_rules.update(agg.keys())
    # Optional: stable order
    rule_order = sorted(all_rules)

    # Build stacked bar data: one bar per method, stacks per rule
    import matplotlib.pyplot as plt
    import numpy as np

    counts_by_rule = {rule: [] for rule in rule_order}
    for agg in aggregates:
        for rule in rule_order:
            counts_by_rule[rule].append(int(agg.get(rule, 0)))

    x = np.arange(len(methods))
    width = 0.6

    fig, ax = plt.subplots(figsize=(8.5, 4.75))
    bottom = np.zeros(len(methods), dtype=float)

    # Plot stacks in rule order so legend is stable
    for rule in rule_order:
        vals = np.array(counts_by_rule[rule], dtype=float)
        ax.bar(x, vals, width, bottom=bottom, label=rule)
        bottom += vals

    # Labels & style
    ax.set_ylabel("Violation count")
    ax.set_title("Constraint violations by method (consistent with center tables)")
    ax.set_xticks(x, methods)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="Rule", ncols=min(3, len(rule_order)), fontsize=9, title_fontsize=9)

    # Tight + save
    plt.tight_layout()
    viol_path = os.path.join(out_dir, "violations_all_methods.png")
    plt.savefig(viol_path, dpi=160)
    plt.close()

    print(f"[INFO] Saved unified violation comparison → {viol_path}", flush=True)


    # Console summary
    def _fmt(m): return f"Sil={m['silhouette']:.3f} | DBI={m['dbi']:.3f} | CH={m['ch']:.1f}"
    print("\n=== SUMMARY ===")
    print(f"SMT_KMeans       : {_fmt(smt_metrics)} | Viol: {smt_audit['aggregate']}")
    print(f"Vanilla_KMeans   : {_fmt(v_metrics)} | Viol: {v_audit['aggregate']}")
    print(f"Projected_KMeans : {_fmt(p_metrics)} | Viol: {p_audit['aggregate']}")
    if pen_metrics is not None:
        print(f"Penalty_KMeans   : {_fmt(pen_metrics)} | Viol: {pen_audit['aggregate']}")
    print("================\n")

    print("[INFO] Done. Inspect:")
    print(f"  - Center tables:   *_{'centers_table.png'}")
    print(f"  - Cluster plots:   *_{'clusters_speed_rpm.png'} and *_{'clusters_speed_throttle.png'}")
    print(f"  - SMT trajectories smt_centroid_trajectories.png (if available)")
    print(f"  - Violation bars:  violations_by_method.png (+ violations_by_method_with_penalty.png if present)")
    print(f"  - Model selection: elbow_vanilla.png & silhouette_vanilla.png")

if __name__ == "__main__":
    main()




