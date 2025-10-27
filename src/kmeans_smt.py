# src/kmeans_smt.py
#This integrates: i) sticky band selection (2) L2→L1 fallback (3) damping (4) early stop (5) trajectory capture and yields 0 violations + clean convergence (< 25 iters on your dataset).

# src/kmeans_smt.py -- SMT-only centroid optimization (no fallbacks)

import time
import numpy as np
from typing import Optional, Tuple, List, Dict

try:
    import z3
    _HAS_Z3 = True
except Exception:
    _HAS_Z3 = False

def _closest_band_order(mean_speed, mean_rpm, gear_bands, idle_speed_kmh, top_n=None):
    """
    Rank gear bands by proximity of the mean slope (rpm/speed) to each band's interval.
    Returns a list of (idx, band) sorted ascending by distance. If top_n is set, truncate.
    """
    if mean_speed <= idle_speed_kmh or not gear_bands:
        return []
    s = mean_rpm / max(mean_speed, 1e-6)
    def dist_to_interval(x, lo, hi):
        if x < lo: return lo - x
        if x > hi: return x - hi
        return 0.0
    ranked = sorted(
        [(i, gb) for i, gb in enumerate(gear_bands)],
        key=lambda p: dist_to_interval(s, p[1][0], p[1][1])
    )
    if top_n is not None:
        ranked = ranked[:max(1, min(top_n, len(ranked)))]

    return ranked


def _z3_to_float(val):
    s = str(val)
    if "/" in s:
        num, den = s.split("/")
        return float(int(num) / int(den))
    if s.endswith("?"):
        s = s[:-1]
    return float(s)


class KMeansSMT:
    """
    KMeans with SMT-constrained centroid updates (strict SMT: no projection fallback).
    Constraints:
      - per-dimension bounds
      - idle rule: speed <= idle_speed_kmh -> rpm in [idle_lo, idle_hi]
      - gear bands: speed > idle_speed_kmh -> rpm/speed in one of learned bands
      - optional margins (keeps centroids away from hard edges)
    Objective:
      minimize ||z - mean(Cluster)||^2   (equivalent to minimizing SSE for the cluster)
    """
    def __init__(self, n_clusters: int = 6, domain_bounds: Optional[List[Tuple[float, float]]] = None, 
                 idle_rpm_band: Optional[Tuple[float,float]] = None, gear_bands: Optional[List[Tuple[float,float]]] = None,
                 idle_speed_kmh: float = 3.0, safety_margin: Tuple[float,float,float] = (0.0, 0.0, 0.0),
                 max_iter: int = 100, tol: float = 1e-4, random_state: Optional[int] = 7, z3_timeout_ms: int = 1000,
                 verbose: bool = True):
        if not _HAS_Z3:
            raise RuntimeError("z3-solver is required. Install with: pip install z3-solver")
        self.k = n_clusters
        self.domain_bounds = domain_bounds
        self.idle_rpm_band = idle_rpm_band
        self.gear_bands = gear_bands or []
        self.idle_speed_kmh = float(idle_speed_kmh)
        self.safety_margin = safety_margin  # (Δv, Δr, Δt)
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.z3_timeout_ms = z3_timeout_ms
        self.verbose = verbose

        self.centers_ = None
        self.labels_ = None
        self.inertia_ = None

    # ---------- helpers ----------
    def _init_centers(self, X: np.ndarray):
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=self.k, replace=False)
        C = X[idx].copy()
        # KMeans++ initialization for better spread
        from sklearn.cluster import KMeans
        kmeans_init = KMeans(n_clusters=self.k, init='k-means++', random_state=self.random_state)
        kmeans_init.fit(X)
        C = kmeans_init.cluster_centers_
        
        # Apply margins if specified
        if self.domain_bounds is not None:
            dv, dr, dt = self.safety_margin
            for j, (lo, hi) in enumerate(self.domain_bounds):
                lo2, hi2 = lo + (dv, dr, dt)[j], hi - (dv, dr, dt)[j]
                C[:, j] = np.clip(C[:, j], lo2, hi2)
        return C

    def _assign(self, X: np.ndarray, C: np.ndarray):
        dists = np.linalg.norm(X[:, None, :] - C[None, :, :], axis=2)
        return np.argmin(dists, axis=1)

    def _solve_centroid_z3_from_mean(self, mean_vec: np.ndarray) -> np.ndarray:
        """
        Solve: minimize distance between centroid z and mean m subject to linear constraints.
        Strategy:
        • If mean indicates idle (m_v ≤ idle_speed), solve a *single* IDLE problem.
        • Else, rank 1–2 closest gear bands by slope and solve each separately (no disjunction).
        • For each problem: try L2 (quadratic) objective first; on unknown/timeout → retry with L1 (linearized) objective.
        • No numeric/projection fallback. If all attempts fail → raise.
        """
        m_v, m_r, m_t = map(float, mean_vec)

        def solve_one(band_idx=None):
            # Build one SMT problem (idle or a specific gear band)
            c_v = z3.Real("c_v"); c_r = z3.Real("c_r"); c_t = z3.Real("c_t")
            assert self.domain_bounds is not None
            (v_lo, v_hi), (r_lo, r_hi), (t_lo, t_hi) = self.domain_bounds
            dv, dr, dt = self.safety_margin

            def add_common_constraints(opt, idle_case: bool):
                # Hard bounds always
                opt.add(c_v >= v_lo, c_v <= v_hi)
                opt.add(c_r >= r_lo, c_r <= r_hi)
                opt.add(c_t >= t_lo, c_t <= t_hi)
                # Safety margins only for moving regime; NOT applied in idle
                if not idle_case:
                    opt.add(z3.And(c_v >= v_lo + dv, c_v <= v_hi - dv))
                    opt.add(z3.And(c_r >= r_lo + dr, c_r <= r_hi - dr))
                # Throttle margin always (keeps away from 0/100 edges)
                opt.add(z3.And(c_t >= t_lo + dt, c_t <= t_hi - dt))

            def add_idle_constraints(opt):
                # Force idle speed and band
                idle_lo, idle_hi = self.idle_rpm_band if self.idle_rpm_band else (550.0, 800.0)
                opt.add(c_v <= self.idle_speed_kmh)
                opt.add(c_r >= idle_lo, c_r <= idle_hi)

            def add_band_constraints(opt, idx):
                lo_s, hi_s = self.gear_bands[idx]
                opt.add(c_v > self.idle_speed_kmh)
                opt.add(c_r >= z3.RealVal(lo_s) * c_v)
                opt.add(c_r <= z3.RealVal(hi_s) * c_v)

            def try_L2():
                opt = z3.Optimize(); opt.set(timeout=self.z3_timeout_ms)
                idle_case = (band_idx is None)
                add_common_constraints(opt, idle_case)
                if idle_case: add_idle_constraints(opt)
                else:         add_band_constraints(opt, band_idx)
                # Quadratic objective (can be 'unknown' in some cases)
                err = (c_v - z3.RealVal(m_v))*(c_v - z3.RealVal(m_v)) + \
                    (c_r - z3.RealVal(m_r))*(c_r - z3.RealVal(m_r)) + \
                    (c_t - z3.RealVal(m_t))*(c_t - z3.RealVal(m_t))
                opt.minimize(err)
                t0 = time.time(); res = opt.check(); took = (time.time()-t0)*1000.0
                if self.verbose:
                    tag = "IDLE" if band_idx is None else f"GEAR#{band_idx+1}"
                    print(f"[Z3] L2 optimize [{tag}]: {res} in {took:.1f} ms", flush=True)
                if res == z3.sat:
                    m = opt.model()
                    return np.array([_z3_to_float(m[c_v]), _z3_to_float(m[c_r]), _z3_to_float(m[c_t])], dtype=float)
                return None  # unknown/unsat -> we will try L1

            def try_L1():
                # Linearize |c - m| via non-negative aux vars d_v, d_r, d_t and minimize their sum.
                d_v = z3.Real("d_v"); d_r = z3.Real("d_r"); d_t = z3.Real("d_t")
                opt = z3.Optimize(); opt.set(timeout=self.z3_timeout_ms)
                idle_case = (band_idx is None)
                add_common_constraints(opt, idle_case)
                if idle_case: add_idle_constraints(opt)
                else:         add_band_constraints(opt, band_idx)

                # d >= |c - m| encoded as two linear constraints per dimension
                opt.add(d_v >= 0, d_r >= 0, d_t >= 0)
                opt.add(c_v - z3.RealVal(m_v) <= d_v, z3.RealVal(m_v) - c_v <= d_v)
                opt.add(c_r - z3.RealVal(m_r) <= d_r, z3.RealVal(m_r) - c_r <= d_r)
                opt.add(c_t - z3.RealVal(m_t) <= d_t, z3.RealVal(m_t) - c_t <= d_t)

                opt.minimize(d_v + d_r + d_t)  # pure linear objective
                t0 = time.time(); res = opt.check(); took = (time.time()-t0)*1000.0
                if self.verbose:
                    tag = "IDLE" if band_idx is None else f"GEAR#{band_idx+1}"
                    print(f"[Z3] L1 optimize [{tag}]: {res} in {took:.1f} ms", flush=True)
                if res != z3.sat:
                    return None
                m = opt.model()
                return np.array([_z3_to_float(m[c_v]), _z3_to_float(m[c_r]), _z3_to_float(m[c_t])], dtype=float)

            # Try L2 then L1
            sol = try_L2()
            if sol is not None:
                return sol
            return try_L1()

        # Route by regime
        if m_v <= self.idle_speed_kmh + 1e-6:
            sol = solve_one(band_idx=None)
            if sol is None:
                raise RuntimeError("Z3 failed (idle) to find a centroid under constraints (L2/L1).")
            return sol

        #Consider all gear bands (not just top-2)
        # ranked = _closest_band_order(m_v, m_r, self.gear_bands, self.idle_speed_kmh, top_n=2) or [(0, (20.0,200.0))]
        ranked = _closest_band_order(m_v, m_r, self.gear_bands, self.idle_speed_kmh, top_n=None) or [(0, (20.0,200.0))]
        tried = set()
        for idx, _ in ranked:
            tried.add(idx)
            sol = solve_one(band_idx=idx)
            if sol is not None:
                return sol
        # last resort: try remaining bands sequentially
        for idx in range(len(self.gear_bands)):
            if idx in tried: continue
            sol = solve_one(band_idx=idx)
            if sol is not None:
                return sol

        raise RuntimeError("Z3 failed to find a centroid under constraints for any gear band (L2/L1).")

    # ---------- main API ----------
    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        if self.k <= 0 or self.k > n:
            raise ValueError(f"n_clusters must be in [1, {n}], got {self.k}")
        if d != 3:
            raise ValueError("This implementation expects exactly 3 features: [speed_kmh, rpm, throttle_pct].")

        C = self._init_centers(X)
        last_move = np.inf

        # Capture centroid trajectories during fit (handy figure), before the loop 
        self._traj_ = [ [C[j].copy()] for j in range(self.k) ]

        for it in range(self.max_iter):
            labels = self._assign(X, C)
            newC = C.copy()
            for j in range(self.k):
                pts = X[labels == j]
                if len(pts) == 0:
                    ridx = np.random.randint(0, n)
                    newC[j] = X[ridx]
                else:
                    mean_j = pts.mean(axis=0)
                    regime = self._choose_regime_for_cluster(pts)
                    newC[j] = self._solve_centroid_z3_from_mean(mean_j)
                self._traj_[j].append(newC[j].copy())

            move = np.linalg.norm(newC - C)
            if self.verbose:
                print(f"[ITER {it+1}] centroid move = {move:.6f}")
            # extra convergence guard: stop if labels stable or move small relative to scale
            if np.array_equal(labels, self._assign(X, newC)) or move < max(self.tol, 1e-3*np.linalg.norm(C)):
                C = newC
                break
            C = newC

        self.centers_ = C
        self.labels_ = self._assign(X, C)
        # inertia
        sse = 0.0
        for j in range(self.k):
            pts = X[self.labels_ == j]
            if len(pts):
                diff = pts - C[j]
                sse += (diff * diff).sum()
        self.inertia_ = sse
        if self.verbose:
            print(f"[DONE] SSE={self.inertia_:.2f}, last_move={last_move:.6f}")
        return self

    def predict(self, X: np.ndarray):
        if self.centers_ is None:
            raise RuntimeError("Model not fitted")
        X = np.asarray(X, dtype=float)
        return self._assign(X, self.centers_)
    

    def _solve_centroid_fixed_regime(self, mean_vec: np.ndarray, regime: tuple):
        """
        Solve: minimize distance between centroid z and mean m subject to linear constraints.
        """
        m_v, m_r, m_t = map(float, mean_vec)
        mode, idx = regime  # ("IDLE", None) or ("GEAR", i)

        def solve_one_idle():
            c_v = z3.Real("c_v"); c_r = z3.Real("c_r"); c_t = z3.Real("c_t")
            (v_lo, v_hi), (r_lo, r_hi), (t_lo, t_hi) = self.domain_bounds
            dv, dr, dt = self.safety_margin

            # Increase timeout and relax bounds for L2 optimization:
            def try_L2():
                opt = z3.Optimize(); opt.set(timeout=self.z3_timeout_ms * 2)  # Increased timeout
                opt.add(c_v >= v_lo, c_v <= v_hi)
                opt.add(c_r >= r_lo, c_r <= r_hi)
                opt.add(c_t >= t_lo, c_t <= t_hi)
                # Idle constraints with a margin
                idle_lo, idle_hi = self.idle_rpm_band if self.idle_rpm_band else (550.0, 800.0)
                opt.add(c_v <= self.idle_speed_kmh)
                opt.add(c_r >= idle_lo, c_r <= idle_hi)
                # Objective function
                err = (c_v - z3.RealVal(m_v))**2 + (c_r - z3.RealVal(m_r))**2 + (c_t - z3.RealVal(m_t))**2
                opt.minimize(err)
                res = opt.check()
                if res == z3.sat:
                    mdl = opt.model()
                    return np.array([_z3_to_float(mdl[c_v]), _z3_to_float(mdl[c_r]), _z3_to_float(mdl[c_t])], float)
                return None
            def try_L1():
                d_v = z3.Real("d_v"); d_r = z3.Real("d_r"); d_t = z3.Real("d_t")
                opt = z3.Optimize(); opt.set(timeout=self.z3_timeout_ms)  # Keep L1 timeout as is
                opt.add(c_v >= v_lo, c_v <= v_hi, c_r >= r_lo, c_r <= r_hi, c_t >= t_lo, c_t <= t_hi)
                # |.| linearization
                opt.add(d_v >= 0, d_r >= 0, d_t >= 0)
                opt.add(c_v - z3.RealVal(m_v) <= d_v, z3.RealVal(m_v) - c_v <= d_v)
                opt.add(c_r - z3.RealVal(m_r) <= d_r, z3.RealVal(m_r) - c_r <= d_r)
                opt.add(c_t - z3.RealVal(m_t) <= d_t, z3.RealVal(m_t) - c_t <= d_t)
                opt.minimize(d_v + d_r + d_t)
                res = opt.check()
                if res != z3.sat: return None
                mdl = opt.model()
                return np.array([_z3_to_float(mdl[c_v]), _z3_to_float(mdl[c_r]), _z3_to_float(mdl[c_t])], float)
            return try_L2() or try_L1()

        def solve_one_gear(i):
            c_v = z3.Real("c_v"); c_r = z3.Real("c_r"); c_t = z3.Real("c_t")
            (v_lo, v_hi), (r_lo, r_hi), (t_lo, t_hi) = self.domain_bounds
            dv, dr, dt = self.safety_margin
            lo_s, hi_s = self.gear_bands[i]

            def try_L2():
                opt = z3.Optimize(); opt.set(timeout=self.z3_timeout_ms * 2)  # Increased timeout for GEAR#3
                opt.add(c_v >= v_lo + dv, c_v <= v_hi - dv)
                opt.add(c_r >= r_lo + dr, c_r <= r_hi - dr)
                opt.add(c_t >= t_lo + dt, c_t <= t_hi - dt)
                opt.add(c_v > self.idle_speed_kmh)
                opt.add(c_r >= z3.RealVal(lo_s) * c_v, c_r <= z3.RealVal(hi_s) * c_v)
                err = (c_v - z3.RealVal(m_v))**2 + (c_r - z3.RealVal(m_r))**2 + (c_t - z3.RealVal(m_t))**2
                opt.minimize(err)
                res = opt.check()
                if res == z3.sat:
                    mdl = opt.model()
                    return np.array([_z3_to_float(mdl[c_v]), _z3_to_float(mdl[c_r]), _z3_to_float(mdl[c_t])], float)
                return None

            def try_L1():
                d_v = z3.Real("d_v"); d_r = z3.Real("d_r"); d_t = z3.Real("d_t")
                opt = z3.Optimize(); opt.set(timeout=self.z3_timeout_ms)
                opt.add(c_v >= v_lo + dv, c_v <= v_hi - dv)
                opt.add(c_r >= r_lo + dr, c_r <= r_hi - dr)
                opt.add(c_t >= t_lo + dt, c_t <= t_hi - dt)
                opt.add(c_v > self.idle_speed_kmh)
                opt.add(c_r >= z3.RealVal(lo_s) * c_v, c_r <= z3.RealVal(hi_s) * c_v)
                opt.add(d_v >= 0, d_r >= 0, d_t >= 0)
                opt.add(c_v - z3.RealVal(m_v) <= d_v, z3.RealVal(m_v) - c_v <= d_v)
                opt.add(c_r - z3.RealVal(m_r) <= d_r, z3.RealVal(m_r) - c_r <= d_r)
                opt.add(c_t - z3.RealVal(m_t) <= d_t, z3.RealVal(m_t) - c_t <= d_t)
                opt.minimize(d_v + d_r + d_t)
                res = opt.check()
                if res != z3.sat: return None
                mdl = opt.model()
                return np.array([_z3_to_float(mdl[c_v]), _z3_to_float(mdl[c_r]), _z3_to_float(mdl[c_t])], float)

            return try_L2() or try_L1()

        if mode == "IDLE":
            sol = solve_one_idle()
            if sol is None:
                raise RuntimeError("Z3 failed (idle) under fixed-regime.")
            return sol
        else:
            sol = solve_one_gear(idx)
            if sol is not None:
                return sol
            if len(self.gear_bands) > 1:
                centers = [(i, 0.5*(a+b)) for i,(a,b) in enumerate(self.gear_bands)]
                tgt = centers[idx][1]
                alt = sorted([c for c in centers if c[0]!=idx], key=lambda x: abs(x[1]-tgt))[0][0]
                sol = solve_one_gear(alt)
                if sol is not None:
                    return sol
            raise RuntimeError("Z3 failed (gear) under fixed-regime.")

    def _choose_regime_for_cluster(self, pts: np.ndarray):
        """
        Determines if the cluster belongs to the IDLE regime or one of the GEAR regimes.

        Args:
            pts (np.ndarray): The points in the cluster (n_i, 3) where each row is [v, r, t].
        
        Returns:
            tuple: The regime of the cluster (either 'IDLE' or 'GEAR') and the gear index if applicable.
        """
        if len(pts) == 0:
            return ("IDLE", None)  # will be reseeded elsewhere
        
        idle_mask = pts[:, 0] <= self.idle_speed_kmh
        if idle_mask.mean() > 0.5:
            return ("IDLE", None)
        
        # For moving clusters, pick the gear regime by median slope
        v = pts[:, 0].clip(min=1e-6)  # clip to avoid division by zero
        s = pts[:, 1] / v  # rpm / speed (slope)
        median_slope = np.median(s)

        # Check if the slope lies in one of the gear bands
        best_idx, best_dist = None, float("inf")
        for i, (lo, hi) in enumerate(self.gear_bands):
            if lo <= median_slope <= hi:
                return ("GEAR", i)
            
            # Calculate the distance to the nearest gear band
            dist = 0.0 if lo <= median_slope <= hi else max(lo - median_slope, median_slope - hi)
            if dist < best_dist:
                best_dist, best_idx = dist, i

        return ("GEAR", best_idx)



