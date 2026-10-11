# ─────────────────────────────────────────────────────────────────────────────
# 0_quality_control.py
# Rule-based quality control of the hourly station files (added in v1.1.0)
#
# Faulty values are set to missing (NaN) in data/raw/ and every removal is
# written to data_quality_assessment/qc_log.csv (station, variable, period,
# number of values, rule, reason). The unmasked sensor values are preserved in
# release v1.0.0 of this repository (Zenodo 10.5281/zenodo.20477761).
#
# Rules
#   G1  Global irradiance with no diurnal cycle (median daytime minus median
#       night-time < 50 W m-2): pyranometer non-functional, whole record removed.
#   G2  Station-months in which the 95th percentile of night-time irradiance
#       exceeds 100 W m-2: channel corrupted for that month, month removed.
#   G3  Remaining night-time irradiance > 20 W m-2 (sun below the horizon
#       throughout the hour and the adjacent hours): value removed.
#   G4  Hourly irradiance > 1200 W m-2 (above any plausible hourly mean at
#       1.3 N; the highest value at stations without faults is ~1090 W m-2).
#   P1  Atmospheric pressure outside 950-1050 hPa.
#   W1  Stale wind records: wind speed and direction both unchanged for three
#       or more consecutive hours (logger repeating the last reading).
#   M1  Manually confirmed sensor malfunction (listed in MANUAL below).
#
# The script is idempotent: re-running it on already-cleaned files removes
# nothing further. Run it once, before 1_data_quality_check.py.
# ─────────────────────────────────────────────────────────────────────────────
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "data_quality_assessment")
os.makedirs(OUT_DIR, exist_ok=True)

RAD, PRES = "GlobalRad Ave (W/m2)", "AtmPress Ave (hPa)"
WS, WD = "WindSpeed Ave (m/s)", "WindDir Ave (degrees)"

NIGHT_MAX = 20.0          # W m-2, G3
MONTH_P95_MAX = 100.0     # W m-2, G2
RAD_MAX = 1200.0          # W m-2, G4
DIURNAL_MIN = 50.0        # W m-2, G1
PRES_RANGE = (950.0, 1050.0)
STALE_HOURS = 3

# Confirmed malfunctions found by inspection (M1)
MANUAL = [
    ("WS17", [WS, WD], "2025-12-20 12:00", "2025-12-26 08:00",
     "Anemometer malfunction: repeated stale records and implausible hourly "
     "means up to 17 m/s at a sheltered ground-level station"),
]


def solar_elevation(ts_local, lat, lon):
    """Solar elevation (deg) for SGT (UTC+8) timestamps; NOAA approximation."""
    t = ts_local - pd.Timedelta(hours=8)
    doy = t.dayofyear.values
    hr = t.hour.values + t.minute.values / 60.0
    g = 2 * np.pi / 365 * (doy - 1 + (hr - 12) / 24)
    eqt = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                    - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))
    dec = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g)
           - 0.006758 * np.cos(2 * g) + 0.000907 * np.sin(2 * g)
           - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    ha = np.radians((hr * 60 + eqt + 4 * lon) / 4 - 180)
    la = np.radians(lat)
    cz = np.sin(la) * np.sin(dec) + np.cos(la) * np.cos(dec) * np.cos(ha)
    return np.degrees(np.arcsin(np.clip(cz, -1, 1)))


def night_mask(ts, lat, lon):
    """True where the sun is below the horizon from t-1h to t+1h, so the
    result does not depend on whether hourly means are labelled at the start
    or the end of the averaging interval."""
    el = np.vstack([solar_elevation(ts + pd.Timedelta(minutes=m), lat, lon)
                    for m in range(-60, 61, 15)])
    return el.max(axis=0) < 0


def runs_of(mask):
    """Length of the run of True values each position belongs to (0 if False)."""
    grp = (mask != mask.shift()).cumsum()
    return mask.groupby(grp).transform("size").where(mask, 0)


log = []


def remove(df, ws, var, sel, rule, reason):
    sel = sel & df[var].notna()
    n = int(sel.sum())
    if n == 0:
        return
    t = df.loc[sel, "Datetime"]
    log.append(dict(Station=ws, Variable=var, Start=t.min(), End=t.max(),
                    N_values=n, Rule=rule, Reason=reason))
    df.loc[sel, var] = np.nan


files = sorted(glob.glob(os.path.join(RAW_DIR, "NUS_CAMPUS_WS*_2025_Hourly.csv")))
print(f"Quality control of {len(files)} station files")

for fpath in files:
    ws = os.path.basename(fpath).split("_")[2]
    df = pd.read_csv(fpath, parse_dates=["Datetime"])
    if "Elevation" in df.columns:
        df = df.rename(columns={"Elevation": "Elevation (m)"})
    lat, lon = float(df["Latitude"].iloc[0]), float(df["Longitude"].iloc[0])
    ts = pd.DatetimeIndex(df["Datetime"])
    night = pd.Series(night_mask(ts, lat, lon), index=df.index)
    month = df["Datetime"].dt.month

    # ── Global irradiance ────────────────────────────────────────────────────
    g = df[RAD]
    if g.notna().sum() > 0:
        day = ~night & df["Datetime"].dt.hour.between(10, 15)
        amp = g[day].median() - g[night].median()
        if not (amp >= DIURNAL_MIN):
            remove(df, ws, RAD, pd.Series(True, index=df.index), "G1",
                   f"No diurnal cycle (day-night median difference {amp:.0f} W m-2); "
                   "pyranometer non-functional")
        else:
            p95 = g[night].groupby(month[night]).quantile(0.95)
            for m, v in p95.items():
                if v > MONTH_P95_MAX:
                    remove(df, ws, RAD, month == m, "G2",
                           f"Night-time irradiance 95th percentile {v:.0f} W m-2 "
                           "in this month; channel corrupted")
            remove(df, ws, RAD, night & (df[RAD] > NIGHT_MAX), "G3",
                   f"Night-time irradiance above {NIGHT_MAX:.0f} W m-2")
            remove(df, ws, RAD, df[RAD] > RAD_MAX, "G4",
                   f"Hourly irradiance above {RAD_MAX:.0f} W m-2")

    # ── Atmospheric pressure ─────────────────────────────────────────────────
    lo, hi = PRES_RANGE
    remove(df, ws, PRES, (df[PRES] < lo) | (df[PRES] > hi), "P1",
           f"Pressure outside {lo:.0f}-{hi:.0f} hPa (sensor fault)")

    # ── Wind: stale repeated records ─────────────────────────────────────────
    same = (df[WS].diff() == 0) & (df[WD].diff() == 0) & df[WS].notna()
    run = runs_of(same)
    stale = run >= STALE_HOURS - 1
    stale = stale | stale.shift(-1, fill_value=False)   # include first record of the run
    for v in (WS, WD):
        remove(df, ws, v, stale, "W1",
               f"Wind speed and direction unchanged for >= {STALE_HOURS} h (stale record)")

    # ── Manual flags ─────────────────────────────────────────────────────────
    for st, vars_, start, end, reason in MANUAL:
        if st != ws:
            continue
        sel = df["Datetime"].between(pd.Timestamp(start), pd.Timestamp(end))
        for v in vars_:
            remove(df, ws, v, sel, "M1", reason)

    df.to_csv(fpath, index=False)

qc = pd.DataFrame(log, columns=["Station", "Variable", "Start", "End",
                                "N_values", "Rule", "Reason"])
log_path = os.path.join(OUT_DIR, "qc_log.csv")
if os.path.exists(log_path) and qc.empty:
    print("No further values removed (files already quality-controlled); "
          "existing qc_log.csv kept.")
else:
    qc.to_csv(log_path, index=False)
    print(qc.groupby(["Station", "Variable", "Rule"])["N_values"].sum().to_string())
    print(f"\nTotal values removed: {qc['N_values'].sum():,}")
    print(f"Log written to {log_path}")
