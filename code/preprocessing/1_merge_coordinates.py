import pandas as pd
import glob
import os
import re

locations = pd.read_csv("D:/2025Hourly/BEAM_WS_Locations.csv", encoding="latin-1")
locations.columns = locations.columns.str.strip()
loc_lookup = locations.set_index("Procurement number")

station_files = sorted(glob.glob("D:/2025Hourly/WS*.csv"))
print(f"Found {len(station_files)} station files\n")

errors = []
for fpath in station_files:
    fname = os.path.basename(fpath)
    df = pd.read_csv(fpath)

    proc_id_col = df["Procurement ID"].iloc[0]
    match = re.search(r"NUS_WS(\d+)", proc_id_col)
    if not match:
        errors.append(f"Could not parse Procurement ID in {fname}")
        continue
    proc_num = int(match.group(1))

    if proc_num not in loc_lookup.index:
        errors.append(f"Procurement number {proc_num} not in locations file ({fname})")
        continue

    row = loc_lookup.loc[proc_num]
    lat = row["Lat"]
    lon = row["Lon"]

    # Insert Latitude and Longitude after Datetime (position 1 and 2)
    df.insert(1, "Latitude", lat)
    df.insert(2, "Longitude", lon)

    df.to_csv(fpath, index=False)
    print(f"OK  {fname}  ->  Lat={lat}, Lon={lon}")

print("\n--- Summary ---")
if errors:
    print(f"{len(errors)} error(s):")
    for e in errors:
        print(f"  ERROR: {e}")
else:
    print(f"All {len(station_files)} files updated successfully.")
