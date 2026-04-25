# 🌦️ NUS Campus Meteorological Network — Hourly Dataset (2025)

Hourly meteorological observations from a **40-station automatic weather station (AWS) network** deployed across the National University of Singapore (NUS) main campus for the full calendar year 2025 — to our knowledge, the densest publicly documented campus-scale meteorological network in the world.

---

## 📋 At a Glance

| Item | Detail |
|---|---|
| Stations | 40 (WS01–WS40) |
| Coverage | 1 January – 31 December 2025 (8,760 hourly steps) |
| Location | NUS main campus, Singapore (1.290°–1.308°N, 103.770°–103.783°E) |
| Variables | Air temperature, relative humidity, atmospheric pressure, global solar irradiance, wind speed, wind direction; rainfall at WS02, WS16, WS35 |
| Temporal resolution | 1 hour |
| Raw completeness | 99.77% network mean; 34/40 stations at 100% |
| Gap-filling | Two-stage: linear interpolation + XGBoost spatial regression |
| Formats | CSV (raw + gap-filled with flags); companion Python package |

---

## 📁 Repository Structure

```
nus-campus-weather-paper/
│
├── data/
│   ├── raw/                             # Original sensor readings (40 CSV files)
│   │   └── NUS_CAMPUS_WS##_2025_Hourly.csv
│   └── imputed/                         # Gap-filled dataset (40 CSV files + flag columns)
│       └── NUS_CAMPUS_WS##_2025_Hourly_imputed.csv
│
├── figures/                             # All manuscript figures
│   ├── Fig4_BubbleMap_Completeness.png
│   ├── Fig5_Heatmap_MonthlyCompleteness.png
│   ├── Fig6a_WeatherRadials_Network.png
│   ├── Fig6b_Climatology_DiurnalSeasonal.png
│   ├── Fig7_NetworkTimeSeries.png
│   ├── Fig8a–f_*_AllStations.png        # Per-variable radials, all 40 stations
│   ├── Fig9_SpatialVariability.png
│   └── Fig10_Imputation_Examples.{png,pdf}
│
├── results/
│   ├── data_quality_summary.csv         # Per-station completeness statistics
│   ├── imputation_evaluation.csv        # LOSO cross-validation metrics
│   └── imputation_plausibility.csv      # Post-imputation physical bounds check
│
├── code/
│   ├── 1_data_quality_check.py          # Compute completeness → results/
│   ├── 2_plot_data_completeness.py      # Fig 4, Fig 5
│   ├── 3_plot_weather_data.py           # Fig 6b, Fig 7, Fig 9
│   ├── 4_plot_weather_radials.py        # Fig 6a, Fig 8a–f
│   ├── 5_impute_missing_data.py         # Gap-filling pipeline + LOSO validation
│   ├── 6_plot_imputation_examples.py    # Fig 10
│   └── preprocessing/                   # Historical only — not needed for re-use
│       ├── 1_merge_coordinates.py
│       ├── 2_rename_files.py
│       └── 3_build_ws_photo_slide.py
│
├── .gitignore
├── LICENSE
└── README.md
```

---

## 📂 Data Files

### Raw CSV — column structure

| Column | Units | Description |
|---|---|---|
| `Datetime` | SGT (UTC+8) | Hourly timestamp, ISO 8601 |
| `Latitude` | °N WGS84 | Station latitude (constant within file) |
| `Longitude` | °E WGS84 | Station longitude (constant within file) |
| `AirTemp Ave (C)` | °C | Hourly mean air temperature at 2 m |
| `RelHum Ave (%)` | % | Hourly mean relative humidity at 2 m |
| `AtmPress Ave (hPa)` | hPa | Hourly mean atmospheric pressure |
| `GlobalRad Ave (W/m2)` | W m⁻² | Hourly mean global horizontal irradiance |
| `WindSpeed Ave (m/s)` | m s⁻¹ | Hourly mean wind speed at 2 m |
| `WindDir Ave (degrees)` | ° | Hourly mean wind direction (0–360°) |
| `Rain Tot (mm)` | mm | Hourly rainfall total *(WS02, WS16, WS35 only)* |

Missing values are `NaN` (empty cells in CSV).

### Imputed CSV — flag columns

Each variable gains a companion `_flag` column (e.g. `AirTemp_flag`):

| Flag | Meaning |
|---|---|
| `0` | Original sensor observation |
| `1` | Stage 1 — linear interpolation (gap ≤ 6 h) |
| `2` | Stage 2 — XGBoost spatial regression |

> To use only original data, filter all flag columns == 0.

### ⚠️ Known instrument issues

- **WS17 atmospheric pressure** — sensor fault (~811 hPa, ~200 hPa below ambient). `AtmPress Ave (hPa)` is `NaN` in both raw and imputed files. All other WS17 variables are valid.
- **WS38 global irradiance** — complete pyranometer failure for the full year (0% valid). The imputed file contains a spatially-reconstructed estimate from neighbouring stations; treat with caution.
- **Wind speed / direction** — LOSO R² is negative at intra-campus scale (building wake effects decorrelate neighbouring stations). Use only flag = 0 observations for wind analyses.

---

## 🔧 Gap-Filling Methodology

### Stage 1 — Linear interpolation
Gaps of ≤ 6 consecutive hours with valid bounding observations on both sides. Wind direction interpolated via shortest circular arc. **3,302 values filled (3.3% of all fills).**

### Stage 2 — XGBoost spatial regression
All remaining gaps. One model per variable, trained across all 40 stations simultaneously. Targets are expressed as anomalies from each station's annual mean, removing calibration and microclimate offsets between stations. Features: same-timestep readings from the 8 nearest stations, station coordinates, cyclic hour/month encodings. **95,967 values filled (96.7% of all fills).**

### Leave-one-station-out validation

Five stations held out (WS01, WS08, WS12, WS24, WS28), evaluated on synthetic gaps of 24–720 h:

| Variable | MAE | R² |
|---|---|---|
| Air temperature | ~0.40–0.44 °C | ~0.92–0.94 |
| Relative humidity | ~1.6–1.8 % | ~0.91–0.95 |
| Atmospheric pressure | ~0.05–0.06 hPa | ~0.993–0.998 |
| Global solar irradiance | ~95–104 W m⁻² | ~0.63–0.68 |
| Wind speed | ~0.29–0.34 m s⁻¹ | negative |

---

## 🐍 Python Package

A companion package `nus_campus_weather` is available for direct programmatic access to the dataset.

**Installation (once published on PyPI):**
```bash
pip install nus-campus-weather
```

**Quick start:**
```python
import nus_campus_weather as ncw

ncw.set_data_dir("/path/to/nus-campus-weather-paper")

df       = ncw.load_station("WS01")                          # single station, raw
df_imp   = ncw.load_station("WS01", imputed=True)            # single station, gap-filled
wide_df  = ncw.load_all(variable="AirTemp Ave (C)")          # all 40 stations, wide (8760×40)
long_df  = ncw.load_all(imputed=True)                        # all stations, long form
meta     = ncw.station_metadata()                            # lat/lon per station
```

Or set the data directory via environment variable:
```bash
export NUS_WEATHER_DATA="/path/to/nus-campus-weather-paper"
```

---

## ▶️ Reproducing the Analysis

Run the scripts in order from the repository root. Each script resolves paths relative to the repo — no configuration needed.

```bash
python code/1_data_quality_check.py        # → results/data_quality_summary.csv
python code/2_plot_data_completeness.py    # → figures/Fig4, Fig5
python code/3_plot_weather_data.py         # → figures/Fig6b, Fig7, Fig9
python code/4_plot_weather_radials.py      # → figures/Fig6a, Fig8a–f
python code/5_impute_missing_data.py       # → data/imputed/, results/imputation_*.csv
python code/6_plot_imputation_examples.py  # → figures/Fig10
```

**Dependencies:** Python ≥ 3.9

```bash
pip install pandas numpy matplotlib seaborn scipy xgboost contextily
```

---

## 📄 Citation

Citation will be added upon publication.

---

## 📜 Licence

Data: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Code: [MIT](https://opensource.org/licenses/MIT)
