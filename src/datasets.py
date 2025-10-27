#srce/datasets.py --- robust loader + constraint discovery (idle/gear bands)
# src/datasets.py
import os, glob
import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Optional, Dict

# ---------- Header normalization ----------
def normalize_header(name: str) -> str:
    s = re.sub(r"\[.*?\]|\(.*?\)", "", str(name))
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_")
    return s.lower()

def build_normalized_map(cols) -> dict:
    return {normalize_header(c): c for c in cols}

SPEED_PATTERNS = [
    r"^(vehicle_)?speed(_sensor)?$", r"^speed$", r"^veh_speed$", r"^vss$",
    r"^vehicle_speed_sensor$", r"^vehicle_speed$"
]
RPM_PATTERNS = [
    r"^(engine_)?rpm$", r"^rpm$", r"^engine_speed$", r"^eng_rpm$"
]
THROTTLE_PATTERNS = [
    r"^(absolute_)?throttle(_position)?$", r"^throttle$", r"^throttle_position$",
    r"^throttle_pct$", r"^throttle_position_absolute$",
]

def _find_by_patterns(norm_map: dict, patterns: List[str]) -> Optional[str]:
    for pat in patterns:
        rx = re.compile(pat)
        for norm in norm_map.keys():
            if rx.match(norm):
                return norm_map[norm]
    return None

def _read_csv_robust(path: str) -> pd.DataFrame:
    # Most OBD logs are UTF-8 + comma; try ; fallback
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.read_csv(path, sep=";")

# ---------- Public API ----------
def load_folder(folder: str) -> pd.DataFrame:
    """Load all CSVs, extract speed_kmh, rpm, throttle_pct, clean & concatenate."""
    paths = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"No CSVs found in {folder}")

    frames, misses = [], []
    for p in paths:
        try:
            df = _read_csv_robust(p)
        except Exception as e:
            misses.append((p, f"read_error: {e}"))
            continue

        norm_map = build_normalized_map(df.columns)
        speed_col = _find_by_patterns(norm_map, SPEED_PATTERNS)
        rpm_col   = _find_by_patterns(norm_map, RPM_PATTERNS)

        # Prefer Absolute Throttle Position if present
        throttle_col = _find_by_patterns(norm_map, THROTTLE_PATTERNS)
        if not throttle_col:
            # fallback to accelerator pedal if available
            for cand in ["accelerator_pedal_position_d", "accelerator_pedal_position_e"]:
                raw = build_normalized_map(df.columns).get(cand)
                if raw:
                    throttle_col = raw
                    break

        if not (speed_col and rpm_col and throttle_col):
            misses.append((p, {"found": {"speed": speed_col, "rpm": rpm_col, "throttle": throttle_col},
                               "headers": list(df.columns)}))
            continue

        sub = df[[speed_col, rpm_col, throttle_col]].copy()
        sub.columns = ["speed_kmh", "rpm", "throttle_pct"]
        # Clean
        sub = sub.replace([np.inf, -np.inf], np.nan).dropna()
        # Clip to plausible physicals (generous)
        sub["speed_kmh"]    = sub["speed_kmh"].clip(0, 220)
        sub["rpm"]          = sub["rpm"].clip(0, 8000)
        sub["throttle_pct"] = sub["throttle_pct"].clip(0, 100)
        if len(sub):
            frames.append(sub)

    if not frames:
        msg = "No valid CSV with required columns."
        if misses:
            msg += f"\nDiagnostics (first 3): {misses[:3]}"
        raise RuntimeError(msg)

    data = pd.concat(frames, ignore_index=True)
    return data

def learn_idle_band_and_gear_bands(df: pd.DataFrame,
                                   min_speed_idle: float = 3.0,
                                   n_gears: int = 5) -> Dict:
    """
    Learn idle rpm band and gear ratio bands from data:
    - idle band from rpm values at very low speed
    - gear bands from ratio (rpm/speed) at speed > min_speed_idle
    Returns:
      {
        "idle_rpm": (lo, hi),
        "gear_bands": [(slope_lo, slope_hi), ...]  # length n_gears, sorted high->low
      }
    """
    out = {}

    # --- Idle band ---
    idle = df[df["speed_kmh"] <= min_speed_idle]["rpm"]
    if len(idle) >= 50:
        q1, q3 = np.percentile(idle, [25, 75])
        iqr = q3 - q1
        lo = max(400.0, q1 - 1.5*iqr)   # be conservative
        hi = min(1200.0, q3 + 1.5*iqr)
    else:
        lo, hi = 550.0, 800.0
    out["idle_rpm"] = (float(lo), float(hi))

    # --- Gear bands ---
    moving = df[(df["speed_kmh"] > min_speed_idle) & (df["rpm"] > 0)]
    if len(moving) < 200:
        # fall back to typical passenger car bands (rpm per km/h)
        base = [(100,140),(60,75),(40,50),(30,37),(25,30)]
        out["gear_bands"] = base[:n_gears]
        return out

    ratio = (moving["rpm"] / moving["speed_kmh"]).clip(5, 400)  # avoid div noise
    ratio = ratio[np.isfinite(ratio)]
    # Quantile partition into n_gears groups, widen each by +/-10%
    qs = np.quantile(ratio, np.linspace(0,1,n_gears+1))
    bands = []
    for i in range(n_gears):
        a, b = float(qs[i]), float(qs[i+1])
        if a == b:
            b = a * 1.05 + 1.0
        widen = 0.10
        lo = max(5.0, a*(1-widen))
        hi = min(500.0, b*(1+widen))
        if lo > hi:
            lo, hi = min(lo, hi), max(lo, hi)
        bands.append((lo, hi))
    # Sort high->low slope (1st gear high slope, last gear low slope)
    bands = sorted(bands, key=lambda x: -x[0])
    out["gear_bands"] = bands
    return out

def make_X(df: pd.DataFrame) -> np.ndarray:
    return df[["speed_kmh","rpm","throttle_pct"]].to_numpy(dtype=float)



