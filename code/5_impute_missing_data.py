# ─────────────────────────────────────────────────────────────────────────────
# 5_impute_missing_data.py
# Gap-filling pipeline — NUS Campus Meteorological Network (2025)
#
# METHODOLOGY (for paper)
# ───────────────────────
# Two-stage spatial imputation:
#
#   Stage 1 — Linear interpolation for contiguous gaps up to a
#     variable-specific limit (LINEAR_GAP_H) where valid observations exist on
#     both sides.  Wind direction is interpolated via the shortest circular arc
#     to avoid 359°→1° artefacts.  The limits are the longest gap lengths for
#     which linear interpolation was at least as accurate as the Stage 2 model
#     in the head-to-head test below (v1.1.0; v1.0.0 used 6 h for all).
#
#   Stage 2 — Gradient-boosted spatial regression (XGBoost) for all
#     remaining gaps.  One model is trained per meteorological variable
#     across all 40 stations.  Features at each missing timestep:
#       · Same-timestep readings of all 5 numeric variables from the
#         K_SPATIAL geographically nearest stations.
#       · Wind direction from those neighbours, encoded as (sin, cos)
#         to preserve circular continuity.
#       · Target-station latitude and longitude.
#       · Cyclic hour-of-day and day-of-year encodings:
#         sin/cos(2π·h/24),  sin/cos(2π·doy/365).
#     XGBoost handles missing neighbour values natively via its built-in
#     sparsity-aware split-finding algorithm; no imputation of features
#     is required.  Wind direction is predicted as (sin, cos) components
#     and reconstructed via atan2.
#
# VALIDATION — Leave-One-Station-Out (LOSO) cross-validation
#   Five complete stations are withheld from training entirely.  The
#   Stage 2 model for each variable is retrained on the remaining 35
#   stations and applied to synthetic contiguous gaps of lengths
#   {6, 24, 72, 168, 720} h excised from the held-out records.
#   N_BLOCKS independent non-overlapping gaps are sampled per
#   (station × block-length) combination.  The held-out station's
#   anomaly baseline (its mean) is computed with the gap excluded, as it
#   would be for a real gap.  Metrics reported:
#     MAE  — mean absolute error (physical units)
#     RMSE — root mean square error
#     MBE  — mean bias error (sign indicates over/under prediction)
#     R²   — coefficient of determination
#   Wind direction uses mean circular error (degrees).
#
# STAGE 1 VALIDATION — linear interpolation vs Stage 2 on identical gaps
#   At the same held-out stations, N_BLOCKS_SHORT synthetic gaps of
#   {1, 2, 3, 6, 12, 24} h are filled by both linear interpolation and the
#   LOSO Stage 2 model.  Output: stage1_validation*.csv.  This sets the
#   LINEAR_GAP_H limits and gives the expected error of flag = 1 values.
#
# FLAGS
#   0 = original observation, 1 = Stage 1 linear interpolation,
#   2 = Stage 2 XGBoost spatial regression, 3 = not measured / not imputed.
#
# NOT IMPUTED (flag 3)
#   Station-variables with less than MIN_VALID_FRACTION of the year observed
#   are not reconstructed, because the station's own baseline is unknown:
#   global irradiance at WS38 (no pyranometer) and at WS32 and WS39
#   (pyranometers non-functional, removed by 0_quality_control.py).
#   Faulty periods (e.g. WS17 pressure, Jan–Apr) are removed by
#   0_quality_control.py and then treated as ordinary gaps.
# ─────────────────────────────────────────────────────────────────────────────

import sys, os, glob, time, warnings, io
import numpy as np

# Force UTF-8 output on Windows consoles
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import pandas as pd
from scipy.spatial.distance import cdist
from xgboost import XGBRegressor
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1 limits (h): longest gap filled by linear interpolation, per variable.
# 0 = never interpolated (always Stage 2).  See STAGE 1 VALIDATION above.
LINEAR_GAP_H = {
    "AirTemp Ave (C)":       1,
    "RelHum Ave (%)":        1,
    "AtmPress Ave (hPa)":    0,
    "GlobalRad Ave (W/m2)":  3,
    "WindSpeed Ave (m/s)":   12,
    "WindDir Ave (degrees)": 12,
}
K_SPATIAL          = 8        # nearest-neighbour stations for spatial features
EVAL_STATIONS      = ["WS01", "WS08", "WS12", "WS24", "WS28"]   # held-out for LOSO
BLOCK_LENGTHS      = [6, 24, 72, 168, 720]                       # synthetic gap lengths (h)
N_BLOCKS           = 5        # non-overlapping synthetic gaps per (station × length)
SHORT_BLOCKS       = [1, 2, 3, 6, 12, 24]                        # Stage 1 validation (h)
N_BLOCKS_SHORT     = 40       # gaps per (station × length) for Stage 1 validation
MIN_VALID_FRACTION = 0.10     # below this, a station-variable is not imputed (flag 3)
RANDOM_SEED        = 42

rng       = np.random.default_rng(RANDOM_SEED)       # LOSO (Table 7)
rng_short = np.random.default_rng(RANDOM_SEED + 1)   # Stage 1 validation

ROOT     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data", "raw")
OUT_DIR  = os.path.join(ROOT, "data", "imputed")
RESULT_DIR = os.path.join(ROOT, "data_quality_assessment")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

VARS_NUMERIC = [
    "AirTemp Ave (C)",
    "RelHum Ave (%)",
    "WindSpeed Ave (m/s)",
    "AtmPress Ave (hPa)",
    "GlobalRad Ave (W/m2)",
]
VAR_WD   = "WindDir Ave (degrees)"
ALL_VARS = VARS_NUMERIC + [VAR_WD]

FLAG_COL = {v: v.split()[0] + "_flag" for v in ALL_VARS}   # e.g. AirTemp_flag

BOUNDS = {
    "AirTemp Ave (C)":      (15.0,  45.0),
    "RelHum Ave (%)":       (10.0, 100.0),
    "WindSpeed Ave (m/s)":  ( 0.0,  25.0),
    "AtmPress Ave (hPa)":   (990.0,1020.0),
    "GlobalRad Ave (W/m2)": ( 0.0, 1500.0),
}

XGB_PARAMS = dict(
    n_estimators=400, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    tree_method="hist", random_state=RANDOM_SEED, n_jobs=-1, verbosity=0,
)

# ═══════════════════════════════════════════════════════════════════════════════
# 1 — Load data
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("NUS Campus Meteorological Network — Gap-Filling Pipeline")
print("=" * 70)

FILES    = sorted(glob.glob(os.path.join(DATA_DIR, "NUS_CAMPUS_WS*.csv")))
FULL_IDX = pd.date_range("2025-01-01", periods=8760, freq="h")

raw  = {}   # {station: DataFrame on full hourly index}
meta = {}   # {station: {lat, lon}}

print(f"\nLoading {len(FILES)} station files …")
for fpath in FILES:
    ws = os.path.basename(fpath).split("_")[2]
    df = (pd.read_csv(fpath, parse_dates=["Datetime"])
            .set_index("Datetime")
            .reindex(FULL_IDX))
    lat_s = df["Latitude"].dropna()
    lon_s = df["Longitude"].dropna()
    if len(lat_s) == 0:
        continue
    meta[ws] = {"lat": float(lat_s.iloc[0]), "lon": float(lon_s.iloc[0])}
    raw[ws] = df

stations = sorted(raw.keys())
print(f"Loaded {len(stations)} stations | {FULL_IDX[0].date()} – {FULL_IDX[-1].date()}")

# ── Pivot tables: variable → (timestep × station) ─────────────────────────
pivots = {v: pd.DataFrame({ws: raw[ws][v] for ws in stations}, index=FULL_IDX)
          for v in ALL_VARS}

wd_rad     = np.deg2rad(pivots[VAR_WD])
wd_sin_piv = np.sin(wd_rad)
wd_cos_piv = np.cos(wd_rad)

# ── Station distances and K nearest neighbours ─────────────────────────────
coords = np.array([[meta[ws]["lat"], meta[ws]["lon"]] for ws in stations])
dist_m = cdist(coords, coords)
np.fill_diagonal(dist_m, np.inf)
neighbors = {
    ws: [stations[j] for j in np.argsort(dist_m[i])[:K_SPATIAL]]
    for i, ws in enumerate(stations)
}

# ── Cyclic time features ───────────────────────────────────────────────────
_h = FULL_IDX.hour.values
_d = FULL_IDX.dayofyear.values
time_feats = pd.DataFrame({
    "sin_h":   np.sin(2 * np.pi * _h / 24),
    "cos_h":   np.cos(2 * np.pi * _h / 24),
    "sin_doy": np.sin(2 * np.pi * _d / 365),
    "cos_doy": np.cos(2 * np.pi * _d / 365),
}, index=FULL_IDX)

# ═══════════════════════════════════════════════════════════════════════════════
# 2 — Gap summary (before imputation)
# ═══════════════════════════════════════════════════════════════════════════════
def find_gap_runs(arr):
    """Return list of (start_pos, length) for each contiguous NaN run."""
    nan_mask = np.isnan(np.asarray(arr, float))
    if not nan_mask.any():
        return []
    padded = np.concatenate([[False], nan_mask, [False]])
    changes = np.diff(padded.astype(int))
    starts = np.where(changes == 1)[0]
    ends   = np.where(changes == -1)[0]
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]

print("\n--- Pre-imputation gap count (missing values per station/variable) ---")
hdr = f"{'Stn':<6}" + "".join(f"  {v.split()[0]:>9}" for v in ALL_VARS)
print(hdr)
for ws in stations:
    row = f"{ws:<6}"
    for v in ALL_VARS:
        n = int(pivots[v][ws].isna().sum())
        row += f"  {n:>9,}"
    print(row)

# ═══════════════════════════════════════════════════════════════════════════════
# 3 — Stage 1: Linear interpolation (gaps ≤ LINEAR_GAP_H, bounded on both sides)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"Stage 1 — Linear interpolation (variable-specific limits: {LINEAR_GAP_H})")

s1_pivots = {v: pivots[v].copy() for v in ALL_VARS}
s1_counts = {v: 0 for v in ALL_VARS}

for ws in stations:
    for v in ALL_VARS:
        col = s1_pivots[v][ws].copy()
        arr = col.values.astype(float)
        for start, length in find_gap_runs(arr):
            if length > LINEAR_GAP_H[v]:
                continue
            if start == 0 or start + length >= len(arr):
                continue
            v_lo = arr[start - 1]
            v_hi = arr[start + length]
            if np.isnan(v_lo) or np.isnan(v_hi):
                continue
            for k in range(length):
                frac = (k + 1) / (length + 1)
                if v == VAR_WD:
                    diff = ((v_hi - v_lo + 180) % 360) - 180
                    arr[start + k] = (v_lo + frac * diff) % 360
                else:
                    arr[start + k] = v_lo + frac * (v_hi - v_lo)
                s1_counts[v] += 1
        s1_pivots[v][ws] = arr

for v in ALL_VARS:
    if s1_counts[v]:
        print(f"  {v:<35}  {s1_counts[v]:>5,} values interpolated")

total_s1 = sum(s1_counts.values())
print(f"  Total Stage 1 fills: {total_s1:,}")

# Update sin/cos pivots after Stage 1
_wd_s1 = np.deg2rad(s1_pivots[VAR_WD])
wd_sin_s1 = np.sin(_wd_s1)
wd_cos_s1 = np.cos(_wd_s1)

# ── Station climatological means (anomaly-based imputation) ───────────────
# Training targets and neighbour features are expressed as deviations from
# each station's observed mean.  This removes systematic inter-station
# offsets (sensor calibration, elevation, microclimate exposure) so the
# model predicts relative change, not absolute level.
# For stations with no valid data (e.g. WS38 GlobalRad), mean is estimated
# from the K nearest neighbours.

station_means = {}
for ws in stations:
    station_means[ws] = {}
    for v in VARS_NUMERIC:
        valid_vals = pivots[v][ws].dropna()
        station_means[ws][v] = float(valid_vals.mean()) if len(valid_vals) >= 50 else np.nan

# Fill missing means from neighbour average
for ws in stations:
    for v in VARS_NUMERIC:
        if not np.isnan(station_means[ws][v]):
            continue
        nbr_means = [station_means[n][v] for n in neighbors[ws]
                     if not np.isnan(station_means[n][v])]
        station_means[ws][v] = float(np.mean(nbr_means)) if nbr_means else 0.0

# ═══════════════════════════════════════════════════════════════════════════════
# Feature matrix builder (shared by LOSO and Stage 2)
# ═══════════════════════════════════════════════════════════════════════════════
def make_features(target_ws, feat_piv, feat_wd_sin, feat_wd_cos, idx=None):
    """
    Features use relative neighbour naming (NN1…NN8) for a consistent schema.
    Numeric neighbour values are expressed as anomalies (value − station mean)
    to match the anomaly-based training targets.
    Wind direction sin/cos remain absolute (circular — anomaly undefined).
    """
    nn    = neighbors[target_ws]
    parts = []

    for k, n in enumerate(nn, 1):
        for v in VARS_NUMERIC:
            anom = (feat_piv[v][n] - station_means[n][v]).rename(f"NN{k}_{v.split()[0]}")
            parts.append(anom)
        parts.append(feat_wd_sin[n].rename(f"NN{k}_wd_sin"))
        parts.append(feat_wd_cos[n].rename(f"NN{k}_wd_cos"))

    parts.append(pd.Series(meta[target_ws]["lat"], index=FULL_IDX, name="lat"))
    parts.append(pd.Series(meta[target_ws]["lon"], index=FULL_IDX, name="lon"))
    parts.append(time_feats)

    X = pd.concat(parts, axis=1)
    return X.loc[idx] if idx is not None else X

# ═══════════════════════════════════════════════════════════════════════════════
# Metric helpers
# ═══════════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) < 2:
        return dict(n=0, MAE=np.nan, RMSE=np.nan, MBE=np.nan, R2=np.nan)
    mae  = float(np.mean(np.abs(yt - yp)))
    rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
    mbe  = float(np.mean(yp - yt))
    sst  = float(np.sum((yt - yt.mean()) ** 2))
    r2   = float(1 - np.sum((yt - yp) ** 2) / sst) if sst > 0 else np.nan
    return dict(n=int(len(yt)), MAE=mae, RMSE=rmse, MBE=mbe, R2=r2)

def circ_mae_deg(y_true, y_pred):
    diff = np.abs(((np.asarray(y_true) - np.asarray(y_pred) + 180) % 360) - 180)
    return float(np.mean(diff))

# ═══════════════════════════════════════════════════════════════════════════════
# 4 — LOSO Cross-Validation (Table 7) and Stage 1 validation
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("LOSO Cross-Validation")
print(f"  Held-out stations : {EVAL_STATIONS}")
print(f"  Training stations : {len(stations) - len(EVAL_STATIONS)}")
print(f"  Block lengths     : {BLOCK_LENGTHS} h  ({N_BLOCKS} gaps each)")
print(f"  Stage 1 test      : {SHORT_BLOCKS} h  ({N_BLOCKS_SHORT} gaps each)")

loso_train_stations = [ws for ws in stations if ws not in EVAL_STATIONS]
eval_records = []
s1_records   = []


def sample_gaps(truth, bl, n, generator):
    """Non-overlapping start positions of fully observed windows of length bl,
    with at least one valid value on each side (needed for interpolation)."""
    ok = np.isfinite(truth).astype(int)
    cumv = np.concatenate([[0], np.cumsum(ok)])
    window_valid = cumv[bl:] - cumv[:-bl]
    possible = [i for i in np.where(window_valid == bl)[0]
                if i >= 5 and i + bl <= len(truth) - 5
                and ok[i - 1] and ok[i + bl]]
    chosen = []
    for _ in range(n * 20):
        if len(chosen) >= n or not possible:
            break
        st = int(generator.choice(possible))
        if any(max(st, a) < min(st + bl, b) for a, b in chosen):
            continue
        chosen.append((st, st + bl))
    return [a for a, _ in chosen]


def predict_gap(v, models, X_gap, base):
    if v == VAR_WD:
        m_sin, m_cos = models
        return np.degrees(np.arctan2(m_sin.predict(X_gap), m_cos.predict(X_gap))) % 360
    return models.predict(X_gap) + base


for v in ALL_VARS:
    is_wd = (v == VAR_WD)
    t0 = time.time()
    print(f"\n  [{v}]")

    # ── Build training data from 35 non-held-out stations ─────────────────
    Xs, ys_sin, ys_cos, ys_num = [], [], [], []

    for ws in loso_train_stations:
        valid = ~pivots[v][ws].isna()   # train on original (uninterpolated) targets
        if valid.sum() < 100:
            continue
        X_ws = make_features(ws, s1_pivots, wd_sin_s1, wd_cos_s1)
        Xs.append(X_ws[valid])
        if is_wd:
            ys_sin.append(wd_sin_piv[ws][valid])
            ys_cos.append(wd_cos_piv[ws][valid])
        else:
            ys_num.append(pivots[v][ws][valid] - station_means[ws][v])

    X_tr = pd.concat(Xs).reset_index(drop=True)

    if is_wd:
        y_sin_tr = pd.concat(ys_sin).reset_index(drop=True)
        y_cos_tr = pd.concat(ys_cos).reset_index(drop=True)
        m_sin = XGBRegressor(**XGB_PARAMS); m_sin.fit(X_tr, y_sin_tr)
        m_cos = XGBRegressor(**XGB_PARAMS); m_cos.fit(X_tr, y_cos_tr)
        models = (m_sin, m_cos)
    else:
        y_tr = pd.concat(ys_num).reset_index(drop=True)
        m = XGBRegressor(**XGB_PARAMS); m.fit(X_tr, y_tr)
        models = m

    print(f"    Trained on {len(X_tr):,} samples ({time.time() - t0:.1f}s)")

    # ── Evaluate synthetic gaps in each held-out station ──────────────────
    for ws in EVAL_STATIONS:
        truth = np.asarray(pivots[v][ws].values, dtype=float)
        if np.isfinite(truth).sum() < max(BLOCK_LENGTHS) + 20:
            continue
        X_ws = make_features(ws, s1_pivots, wd_sin_s1, wd_cos_s1)

        def baseline(start, bl):
            # station mean with the synthetic gap removed (as for a real gap)
            keep = np.ones(len(truth), bool); keep[start:start + bl] = False
            return float(np.nanmean(truth[keep]))

        # Table 7: long-gap LOSO metrics
        for bl in BLOCK_LENGTHS:
            starts = sample_gaps(truth, bl, N_BLOCKS, rng)
            maes, rmses, mbes, r2s = [], [], [], []
            for start in starts:
                gap_ts = FULL_IDX[start:start + bl]
                y_true = truth[start:start + bl]
                y_pred = predict_gap(v, models, X_ws.loc[gap_ts], baseline(start, bl))
                if is_wd:
                    maes.append(circ_mae_deg(y_true, y_pred))
                else:
                    met = compute_metrics(y_true, y_pred)
                    maes.append(met["MAE"]); rmses.append(met["RMSE"])
                    mbes.append(met["MBE"]); r2s.append(met["R2"])
            if not maes:
                continue
            rec = {
                "Variable": v, "Station": ws, "BlockLength_h": bl,
                "N_blocks": len(maes),
                "MAE":  round(float(np.nanmean(maes)),  4),
                "RMSE": round(float(np.nanmean(rmses)), 4) if rmses else np.nan,
                "MBE":  round(float(np.nanmean(mbes)),  4) if mbes  else np.nan,
                "R2":   round(float(np.nanmean(r2s)),   4) if r2s   else np.nan,
            }
            eval_records.append(rec)
            r2_str   = f"{rec['R2']:.4f}" if not np.isnan(rec['R2']) else "  n/a "
            rmse_str = f"{rec['RMSE']:.4f}" if not np.isnan(rec['RMSE']) else "  n/a "
            mbe_str  = f"{rec['MBE']:+.4f}" if not np.isnan(rec['MBE']) else "  n/a "
            print(f"    {ws} | {bl:4d}h | MAE={rec['MAE']:.4f}  "
                  f"RMSE={rmse_str}  MBE={mbe_str}  R²={r2_str}")

        # Stage 1 validation: linear interpolation vs Stage 2 on identical gaps
        for bl in SHORT_BLOCKS:
            for start in sample_gaps(truth, bl, N_BLOCKS_SHORT, rng_short):
                gap_ts = FULL_IDX[start:start + bl]
                y_true = truth[start:start + bl]
                v_lo, v_hi = truth[start - 1], truth[start + bl]
                frac = np.arange(1, bl + 1) / (bl + 1)
                y_xgb = predict_gap(v, models, X_ws.loc[gap_ts], baseline(start, bl))
                if is_wd:
                    diff = ((v_hi - v_lo + 180) % 360) - 180
                    y_lin = (v_lo + frac * diff) % 360
                    err_lin = np.abs(((y_lin - y_true + 180) % 360) - 180)
                    err_xgb = np.abs(((y_xgb - y_true + 180) % 360) - 180)
                else:
                    y_lin = v_lo + frac * (v_hi - v_lo)
                    err_lin = np.abs(y_lin - y_true)
                    err_xgb = np.abs(y_xgb - y_true)
                s1_records.append({
                    "Variable": v, "Station": ws, "GapLength_h": bl, "Start": str(FULL_IDX[start]),
                    "Linear_MAE": float(err_lin.mean()), "Linear_MSE": float((err_lin ** 2).mean()),
                    "XGBoost_MAE": float(err_xgb.mean()), "XGBoost_MSE": float((err_xgb ** 2).mean()),
                })

eval_df = pd.DataFrame(eval_records)
s1_df   = pd.DataFrame(s1_records)

# ═══════════════════════════════════════════════════════════════════════════════
# 5 — Train final models on all 40 stations
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("Stage 2 — Training final models on all stations")

final_models = {}

for v in ALL_VARS:
    is_wd = (v == VAR_WD)
    t0 = time.time()
    Xs, ys_sin, ys_cos, ys_num = [], [], [], []

    for ws in stations:
        valid = ~pivots[v][ws].isna()
        if valid.sum() < 50:
            continue
        X_ws = make_features(ws, s1_pivots, wd_sin_s1, wd_cos_s1)
        Xs.append(X_ws[valid])
        if is_wd:
            ys_sin.append(wd_sin_piv[ws][valid])
            ys_cos.append(wd_cos_piv[ws][valid])
        else:
            ys_num.append(pivots[v][ws][valid] - station_means[ws][v])

    X_tr = pd.concat(Xs).reset_index(drop=True)

    if is_wd:
        y_sin_tr = pd.concat(ys_sin).reset_index(drop=True)
        y_cos_tr = pd.concat(ys_cos).reset_index(drop=True)
        m_sin = XGBRegressor(**XGB_PARAMS); m_sin.fit(X_tr, y_sin_tr)
        m_cos = XGBRegressor(**XGB_PARAMS); m_cos.fit(X_tr, y_cos_tr)
        final_models[v] = (m_sin, m_cos)
    else:
        y_tr = pd.concat(ys_num).reset_index(drop=True)
        m = XGBRegressor(**XGB_PARAMS); m.fit(X_tr, y_tr)
        final_models[v] = m

    print(f"  {v:<35}  {len(X_tr):>8,} samples  ({time.time() - t0:.1f}s)")

# ═══════════════════════════════════════════════════════════════════════════════
# 6 — Apply imputation to actual gaps
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("Applying imputation to actual gaps …")

final_pivots = {v: s1_pivots[v].copy() for v in ALL_VARS}

# Flags: 0 = observed, 1 = Stage 1, 2 = Stage 2, 3 = not measured / not imputed
flag_pivots = {}
for v in ALL_VARS:
    flags = pd.DataFrame(0, index=FULL_IDX, columns=stations, dtype=np.int8)
    for ws in stations:
        # Stage 1 filled: was NaN in original, now valid in s1
        s1_filled = s1_pivots[v][ws].notna() & pivots[v][ws].isna()
        flags.loc[s1_filled, ws] = 1
    flag_pivots[v] = flags

s2_counts = {v: 0 for v in ALL_VARS}
s3_counts = {v: 0 for v in ALL_VARS}
not_imputed = []          # (station, variable) left missing, flag 3

for ws in stations:
    X_ws = make_features(ws, s1_pivots, wd_sin_s1, wd_cos_s1)

    for v in ALL_VARS:
        is_wd = (v == VAR_WD)

        still_missing = s1_pivots[v][ws].isna()
        if not still_missing.any():
            continue

        # Too little data to anchor the station: leave missing, flag 3
        if pivots[v][ws].notna().mean() < MIN_VALID_FRACTION:
            flag_pivots[v].loc[still_missing, ws] = 3
            s3_counts[v] += int(still_missing.sum())
            not_imputed.append((ws, v))
            continue

        gap_ts = FULL_IDX[still_missing]
        X_gap  = X_ws.loc[gap_ts]

        if is_wd:
            m_sin, m_cos = final_models[v]
            ps   = m_sin.predict(X_gap)
            pc   = m_cos.predict(X_gap)
            pred = np.degrees(np.arctan2(ps, pc)) % 360
        else:
            pred = final_models[v].predict(X_gap) + station_means[ws][v]
            lo, hi = BOUNDS[v]
            pred = np.clip(pred, lo, hi)

        final_pivots[v].loc[gap_ts, ws] = pred
        flag_pivots[v].loc[gap_ts, ws]  = 2   # 2 = Stage 2 XGBoost
        s2_counts[v] += len(gap_ts)

for v in ALL_VARS:
    if s2_counts[v]:
        print(f"  {v:<35}  {s2_counts[v]:>6,} values imputed (Stage 2)")

# ═══════════════════════════════════════════════════════════════════════════════
# 7 — Physical plausibility checks
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("Physical plausibility checks on Stage 2 imputed values …")

plaus_records = []
any_warning = False

for ws in stations:
    for v in VARS_NUMERIC:
        s2_mask    = flag_pivots[v][ws] == 2
        imp_vals   = final_pivots[v][ws][s2_mask]
        if len(imp_vals) == 0:
            continue
        lo, hi = BOUNDS[v]
        n_oor  = int(((imp_vals < lo) | (imp_vals > hi)).sum())
        if n_oor > 0:
            print(f"  WARNING: {ws} {v}: {n_oor} out-of-range values (clipped)")
            any_warning = True
        plaus_records.append({
            "Station": ws, "Variable": v,
            "N_stage2": len(imp_vals),
            "Min":      round(float(imp_vals.min()),  3),
            "Max":      round(float(imp_vals.max()),  3),
            "Mean":     round(float(imp_vals.mean()), 3),
            "N_out_of_range": n_oor,
        })

if not any_warning:
    print("  All imputed values within physical bounds.")

# ═══════════════════════════════════════════════════════════════════════════════
# 8 — Save imputed datasets
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print(f"Saving imputed CSVs to {OUT_DIR}/ …")

for ws in stations:
    df_out = raw[ws].copy()
    # Restore metadata that became NaN on rows added during reindex
    df_out["Latitude"]  = meta[ws]["lat"]
    df_out["Longitude"] = meta[ws]["lon"]
    for col in df_out.columns:
        if col not in ALL_VARS + ["Latitude", "Longitude"]:
            df_out[col] = df_out[col].ffill().bfill()

    for v in ALL_VARS:
        df_out[v] = final_pivots[v][ws]
        # Flag: 0=original, 1=Stage1 linear interp, 2=Stage2 XGBoost, 3=not imputed
        df_out[FLAG_COL[v]] = flag_pivots[v][ws].values

    df_out.index.name = "Datetime"
    out_path = os.path.join(OUT_DIR, f"NUS_CAMPUS_{ws}_2025_Hourly_imputed.csv")
    df_out.to_csv(out_path)

print(f"  Saved {len(stations)} files.")
print(f"  Flag column values: 0=original, 1=Stage1 linear, 2=Stage2 XGBoost, 3=not imputed")

# ── Save evaluation and plausibility tables ────────────────────────────────
eval_path  = os.path.join(RESULT_DIR, "imputation_evaluation.csv")
plaus_path = os.path.join(RESULT_DIR, "imputation_plausibility.csv")
s1_path    = os.path.join(RESULT_DIR, "stage1_validation.csv")
eval_df.to_csv(eval_path, index=False)
pd.DataFrame(plaus_records).to_csv(plaus_path, index=False)
s1_df.to_csv(s1_path, index=False)

# Summaries used in the paper
eval_sum = (eval_df.groupby(["Variable", "BlockLength_h"])[["MAE", "RMSE", "MBE", "R2"]]
            .mean().round(4).reset_index())
eval_sum.to_csv(os.path.join(RESULT_DIR, "imputation_evaluation_summary.csv"), index=False)
g = s1_df.groupby(["Variable", "GapLength_h"])
s1_sum = pd.DataFrame({
    "N_gaps": g.size(),
    "Linear_MAE": g["Linear_MAE"].mean(), "Linear_RMSE": np.sqrt(g["Linear_MSE"].mean()),
    "XGBoost_MAE": g["XGBoost_MAE"].mean(), "XGBoost_RMSE": np.sqrt(g["XGBoost_MSE"].mean()),
}).round(4).reset_index()
s1_sum.to_csv(os.path.join(RESULT_DIR, "stage1_validation_summary.csv"), index=False)
counts = pd.DataFrame({
    "Variable": ALL_VARS,
    "Stage1_linear": [s1_counts[v] for v in ALL_VARS],
    "Stage2_xgboost": [s2_counts[v] for v in ALL_VARS],
    "Not_imputed_flag3": [s3_counts[v] for v in ALL_VARS],
})
counts.to_csv(os.path.join(RESULT_DIR, "imputation_counts.csv"), index=False)

# ═══════════════════════════════════════════════════════════════════════════════
# 9 — Report
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'=' * 70}")
print("EVALUATION SUMMARY — LOSO Cross-Validation")
print(f"(Mean across {len(EVAL_STATIONS)} held-out stations: {EVAL_STATIONS})")
print(f"{'=' * 70}")

VAR_UNITS = {
    "AirTemp Ave (C)":       "°C",
    "RelHum Ave (%)":        "%",
    "WindSpeed Ave (m/s)":   "m/s",
    "AtmPress Ave (hPa)":    "hPa",
    "GlobalRad Ave (W/m2)":  "W/m²",
    "WindDir Ave (degrees)": "°  (circular MAE)",
}

if len(eval_df):
    summary = (eval_df
               .groupby(["Variable", "BlockLength_h"])[["MAE", "RMSE", "MBE", "R2"]]
               .mean()
               .round(4))

    for v in ALL_VARS:
        if v not in summary.index.get_level_values(0):
            continue
        print(f"\n  {v}  [{VAR_UNITS[v]}]")
        print(f"  {'Block_h':>8}  {'MAE':>8}  {'RMSE':>8}  {'MBE':>8}  {'R²':>7}")
        sub = summary.loc[v]
        for bl in BLOCK_LENGTHS:
            if bl not in sub.index:
                continue
            r = sub.loc[bl]
            r2s   = f"{r['R2']:.4f}"   if not np.isnan(r['R2'])   else "   n/a"
            rmss  = f"{r['RMSE']:.4f}" if not np.isnan(r['RMSE']) else "   n/a"
            mbs   = f"{r['MBE']:+.4f}" if not np.isnan(r['MBE']) else "   n/a"
            print(f"  {bl:>8}  {r['MAE']:>8.4f}  {rmss:>8}  {mbs:>8}  {r2s:>7}")

print(f"\n{'=' * 70}")
print("POST-IMPUTATION COMPLETENESS")
print(f"{'=' * 70}")
print(f"{'Stn':<6}" + "".join(f"  {v.split()[0]:>9}" for v in ALL_VARS))

for ws in stations:
    row = f"{ws:<6}"
    for v in ALL_VARS:
        pct = final_pivots[v][ws].notna().sum() / 8760 * 100
        row += f"  {pct:>8.2f}%"
    print(row)

total_s1, total_s2 = sum(s1_counts.values()), sum(s2_counts.values())
n_values = len(stations) * 8760 * len(ALL_VARS)
print(f"\n{'=' * 70}")
print("IMPUTATION VOLUME SUMMARY")
print(f"{'=' * 70}")
print(f"  Stage 1 (linear interp)  : {total_s1:>8,} values")
print(f"  Stage 2 (XGBoost spatial): {total_s2:>8,} values")
print(f"  Total filled             : {total_s1 + total_s2:>8,} values "
      f"({(total_s1 + total_s2) / n_values * 100:.2f}% of {n_values:,} hourly values)")
print(f"  Not imputed (flag 3)     : {sum(s3_counts.values()):>8,} values  {not_imputed}")
print(f"\nOutputs:")
print(f"  Imputed CSVs : {OUT_DIR}/")
print(f"  LOSO metrics : {eval_path}")
print(f"  Stage 1 test : {s1_path}")
print(f"  Plausibility : {plaus_path}")
print(f"\nPipeline complete.")
