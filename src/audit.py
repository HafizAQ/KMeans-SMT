# src/audit.py
import numpy as np
from typing import Dict, List, Tuple

def check_idle_rule(center, idle_speed_kmh: float, idle_band: Tuple[float,float]) -> int:
    v, r, _ = center
    if v <= idle_speed_kmh:
        lo, hi = idle_band
        return 0 if (lo <= r <= hi) else 1
    return 0

def check_gear_rule(center,
                    idle_speed_kmh: float,
                    gear_bands: List[Tuple[float,float]]) -> int:
    v, r, _ = center
    if v <= idle_speed_kmh:
        return 0
    if v <= 0:
        return 1
    slope = r / v
    ok = any((lo <= slope <= hi) for (lo, hi) in gear_bands)
    return 0 if ok else 1

def check_bounds(center, bounds) -> Dict[str, int]:
    v, r, t = center
    (v_lo, v_hi), (r_lo, r_hi), (t_lo, t_hi) = bounds
    return {
        "v_low":  int(v < v_lo),
        "v_high": int(v > v_hi),
        "r_low":  int(r < r_lo),
        "r_high": int(r > r_hi),
        "t_low":  int(t < t_lo),
        "t_high": int(t > t_hi),
    }

#src/audit.py -- constraint violation audit 

def audit_centers(centers: np.ndarray,
                  bounds,
                  idle_speed_kmh: float,
                  idle_band: Tuple[float,float],
                  gear_bands: List[Tuple[float,float]]) -> Dict:
    """
    Returns aggregate counts and per-center flags for all constraints.
    """
    agg = dict(v_low=0,v_high=0,r_low=0,r_high=0,t_low=0,t_high=0,
               idle_rule=0, gear_rule=0)
    per_center = []
    for i, c in enumerate(centers):
        b = check_bounds(c, bounds)
        idle_v = check_idle_rule(c, idle_speed_kmh, idle_band)
        gear_v = check_gear_rule(c, idle_speed_kmh, gear_bands)
        row = dict(center=i, **b, idle_rule=idle_v, gear_rule=gear_v)
        per_center.append(row)
        for k in agg:
            agg[k] += row[k]
    return {"aggregate": agg, "per_center": per_center}

def distance_to_bounds(center, bounds):
    """Signed distances to each bound (positive inside, negative outside)."""
    v, r, t = center
    (v_lo, v_hi), (r_lo, r_hi), (t_lo, t_hi) = bounds
    return dict(
        v_lo=v - v_lo, v_hi=v_hi - v,
        r_lo=r - r_lo, r_hi=r_hi - r,
        t_lo=t - t_lo, t_hi=t_hi - t
    )



