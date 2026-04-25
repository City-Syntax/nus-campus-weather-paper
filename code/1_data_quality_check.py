import pandas as pd
import glob
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EXPECTED_HOURS = 8760  # 2025 is not a leap year
DATA_COLS = [
    "WindSpeed Ave (m/s)",
    "WindDir Ave (degrees)",
    "AirTemp Ave (C)",
    "RelHum Ave (%)",
    "AtmPress Ave (hPa)",
    "GlobalRad Ave (W/m2)",
]
FILES = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "NUS_CAMPUS_WS*.csv")))

records = []

for fpath in FILES:
    ws = os.path.basename(fpath).split("_")[2]  # e.g. WS01
    df = pd.read_csv(fpath, parse_dates=["Datetime"])

    n_rows = len(df)
    row_completeness = n_rows / EXPECTED_HOURS * 100

    # Duplicate timestamps
    n_dupes = df["Datetime"].duplicated().sum()

    # Check for full hourly coverage gaps
    if n_rows > 0:
        full_range = pd.date_range(df["Datetime"].min(), df["Datetime"].max(), freq="h")
        n_missing_ts = len(full_range) - df["Datetime"].nunique()
    else:
        n_missing_ts = EXPECTED_HOURS

    rec = {
        "Station": ws,
        "Total_Rows": n_rows,
        "Row_Completeness_%": round(row_completeness, 2),
        "Missing_Timestamps": n_missing_ts,
        "Duplicate_Timestamps": n_dupes,
    }

    for col in DATA_COLS:
        if col in df.columns:
            n_null = df[col].isna().sum()
            pct_valid = round((n_rows - n_null) / EXPECTED_HOURS * 100, 2)
            short = col.split(" ")[0]  # WindSpeed, WindDir, AirTemp, RelHum, AtmPress, GlobalRad
            rec[f"{short}_Missing"] = n_null
            rec[f"{short}_Valid_%"] = pct_valid

    records.append(rec)

summary = pd.DataFrame(records)
summary.to_csv(os.path.join(ROOT, "results", "data_quality_summary.csv"), index=False)

# --- Print report ---
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

print("=" * 80)
print("DATA QUALITY SUMMARY — NUS CAMPUS 40 WEATHER STATIONS 2025")
print(f"Expected hours per station: {EXPECTED_HOURS}")
print("=" * 80)

# Row completeness
print("\n--- ROW COMPLETENESS (per station) ---")
row_cols = ["Station", "Total_Rows", "Row_Completeness_%", "Missing_Timestamps", "Duplicate_Timestamps"]
print(summary[row_cols].to_string(index=False))

# Variable completeness
print("\n--- VARIABLE COMPLETENESS (% valid of 8760 expected) ---")
var_cols = ["Station"] + [c for c in summary.columns if c.endswith("_Valid_%")]
print(summary[var_cols].to_string(index=False))

# Network-level summary
print("\n--- NETWORK-LEVEL SUMMARY ---")
print(f"Stations with 100% row completeness : {(summary['Row_Completeness_%'] == 100).sum()} / {len(summary)}")
print(f"Stations with any missing timestamps : {(summary['Missing_Timestamps'] > 0).sum()} / {len(summary)}")
print(f"Stations with duplicate timestamps  : {(summary['Duplicate_Timestamps'] > 0).sum()} / {len(summary)}")
print()
for col in DATA_COLS:
    short = col.split(" ")[0]
    vcol = f"{short}_Valid_%"
    if vcol in summary.columns:
        mean_v = summary[vcol].mean()
        min_v  = summary[vcol].min()
        min_ws = summary.loc[summary[vcol].idxmin(), "Station"]
        print(f"  {col:<30}  mean valid: {mean_v:.2f}%   min: {min_v:.2f}% ({min_ws})")

print("\nFull summary saved to: data_quality_summary.csv")
