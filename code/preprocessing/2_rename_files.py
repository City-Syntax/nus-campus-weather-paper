import os
import shutil
import glob

station_files = sorted(glob.glob("D:/2025Hourly/WS*.csv"))
print(f"Found {len(station_files)} station files\n")

for fpath in station_files:
    fname = os.path.basename(fpath)
    # Extract station number e.g. WS01 from WS01(...)
    ws_num = fname[:4]  # "WS01"
    new_name = f"NUS_CAMPUS_{ws_num}_2025_Hourly.csv"
    new_path = os.path.join("D:/2025Hourly", new_name)
    shutil.copy2(fpath, new_path)
    print(f"{fname}  ->  {new_name}")

print(f"\nDone. {len(station_files)} renamed copies created. Originals untouched.")
