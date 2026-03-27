# ###########################################################################################################################
#           KMeans–SMT: Physics-Constrained Clustering with Symbolic Reasoning for Intelligent Vehicle Diagnostics          #
# ###########################################################################################################################

    This repository contains the code and artifacts for “ KMeans–SMT: Physics-Constrained Clustering with Symbolic Reasoning for Intelligent Vehicle Diagnostics” 
    We enforce simple, physically-motivated rules during centroid updates using an SMT solver (Z3). The result is zero-violation cluster centroids that remain competitive on standard clustering metrics.

### What this repo does ###

    1) Loads raw OBD-II logs from data/ and normalizes columns into (speed_kmh, rpm, throttle_pct).

    2) Learns constraints from data: an idle RPM band and gear-ratio (rpm/speed) bands.

    3) Fits KMeans–SMT (our method) and baselines (Vanilla, Projected, Penalty).

    4) Produces paper-ready plots including elbow, silhouette, cluster overlays, violation bars, centroid tables, density maps, slope histograms, and SMT centroid trajectories.

    5) Prints a summary with internal metrics and violation counts.

#### Repository Structure

    src/
        __init__.py
        audit.py                    # constraint checks + audit summaries
        baselines.py                # vanilla kmeans, projected-means, penalty-kmeans
        datasets.py                 # robust CSV loader + idle/gear band learning
        extra_plots.py              # density, slope, trajectories, violation bars, tables
        kmeans_smt.py               # our SMT-constrained KMeans
        ksweep.py                   # optional sweep runner (k across methods)
        metrics_plot.py             # metrics (sil/DBI/CH) + 2D plot helper
        run_vehicle_clustering.py   # main end-to-end script
        data/
        *.csv                       # your OBD-II logs (not included)
        outputs/
        (generated figures/tables are saved here)


#### Setup 

    python -m venv .venv
    source .venv/bin/activate    # (Windows: .venv\Scripts\activate)
    pip install -r requirements.txt
    # or minimally:
    pip install numpy pandas scikit-learn matplotlib z3-solver

    Python: 3.1+ recommended
    OS: Works on Windows/Linux/macOS


#### Data (https://radar.kit.edu/radar/en/dataset/bCtGxdTklQlfQcAq#)
    Place your CSVs in data/. The loader auto-detects common header variants:

        Speed: vehicle_speed, veh_speed, vss, …

        RPM: engine_rpm, engine_speed, …

        Throttle: prefers absolute throttle position; falls back to accelerator pedal if needed.

        Values are cleaned and clipped to plausible ranges:

        speed: [0, 130] km/h, rpm: [0, 6000], throttle: [0, 100]%.


#### Run the experiment 

    # from repo root
        python src/run_vehicle_clustering.py
    #Environment knobs (Optional)
        # model selection sweep for Vanilla KMeans
        set K_MIN=4 & set K_MAX=9           # Windows (PowerShell: $env:K_MIN="4")
        # experiment size
        set MAX_ROWS=80000
        # target clusters
        set K=6
        # Z3 budget (ms per optimize call)
        set Z3_TIMEOUT_MS=1000

#### Outputs (paper figures)

    Created under outputs/:

        Model selection:
            elbow_vanilla.png, silhouette_vanilla.png

        Method visuals (for each of SMT/Vanilla/Projected/Penalty):
            *_clusters_speed_rpm.png, *_clusters_speed_throttle.png, *_centers_table.png

        Physics context:
            smt_density_speed_rpm.png, smt_slope_hist.png, smt_gear_bands_card.png

        Optimization trace:
            smt_centroid_trajectories.png

        Safety comparison:
            violations_all_methods, violations_by_method_with_penalty.png (if penalty run)

#### Reproducing the figures in the paper

    Model selection: elbow_vanilla.png, silhouette_vanilla.png

    Context: smt_density_speed_rpm.png, smt_slope_hist.png, smt_gear_bands_card.png

    Method comparisons:
    SMT_KMeans_clusters_*, Vanilla_KMeans_clusters_*, Projected_KMeans_clusters_*, Penalty_KMeans_clusters_*
    *_centers_table.png

    Trajectories: smt_centroid_trajectories.png

    Violations bars: violations_by_method*.png

### Citation (placeholder)
    Quddus, H.A.; Jesser, A. KMeans–SMT: Physics-Constrained Clustering with Symbolic Reasoning for Intelligent Vehicle Diagnostics. International Symposium on Intelligent Technology for Future Transportation. LOCATION OF CONFERENCE, United KingdomDATE OF CONFERENCE; pp. 113–126..

#### License 
    MIT License of choice is added (at the top).

#### Acknowledgements 
    Z3 SMT solver (Microsoft Research) and scikit-learn.
    

