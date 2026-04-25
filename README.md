# NUS Campus Meteorological Network — Hourly Dataset (2025)

Hourly meteorological observations from a 40-station automatic weather station (AWS) network deployed across the National University of Singapore (NUS) main campus for the full calendar year 2025.

---

## Overview

| Item | Detail |
|---|---|
| Stations | 40 (WS01–WS40) |
| Coverage | 1 January 2025 – 31 December 2025 (8,760 hourly steps) |
| Location | NUS main campus, Singapore (1.290°–1.308°N, 103.770°–103.783°E) |
| Variables | Air temperature, relative humidity, atmospheric pressure, global solar irradiance, wind speed, wind direction; rainfall at WS02, WS16, WS35 |
| Temporal resolution | 1 hour |
| Raw completeness | 99.77% (network mean); 34/40 stations at 100% |
| Formats | CSV (raw and gap-filled); Python package |

---

## Repository Structure

```
NUS_CAMPUS_ScientificData/
│
├── data/
│   ├── raw/                          # Original sensor readings (40 CSV files)
│   │   └── NUS_CAMPUS_WS##_2025_Hourly.csv
│   └── imputed/                      # Gap-filled dataset (40 CSV files + flag columns)
│       └── NUS_CAMPUS_WS##_2025_Hourly_imputed.csv
│
├── figures/                          # All manuscript figures (PNG)
│   ├── Fig1_BubbleMap_Completeness.png
│   ├── Fig2_Heatmap_MonthlyCompleteness.png
│   ├── Fig3_Climatology_DiurnalSeasonal.png
│   ├── Fig4_NetworkTimeSeries.png
│   ├── Fig5_SpatialVariability.png
│   ├── Fig6a_WeatherRadials_Network.png
│   ├── Fig6b_*_AllStations.png
│   └── Fig7_Imputation_Examples.png / .pdf
│
├── results/
│   ├── data_quality_summary.csv      # Per-station completeness statistics
│   ├── imputation_evaluation.csv     # LOSO cross-validation metrics
│   └── imputation_plausibility.csv   # Post-imputation physical bounds check
│
├── pre-processing/                   # Historical pre-processing scripts (not needed for re-use)
│   ├── 1_merge_coordinates.py        # Merged GPS coordinates into raw CSVs
│   └── 2_rename_files.py             # Standardised filenames
│
├── 1_data_quality_check.py           # Compute completeness statistics
├── 2_plot_data_completeness.py       # Figures 1 and 2
├── 3_plot_weather_data.py            # Figures 3, 4, and 5
├── 4_plot_weather_radials.py         # Figures 6a and 6b
├── 5_impute_missing_data.py          # Gap-filling pipeline + LOSO validation
├── 6_plot_imputation_examples.py     # Figure 7
├── build_manuscript.py               # Generates NUS_Campus_Weather_Manuscript.docx
│
├── NUS_Campus_Weather_Manuscript.docx
└── README.md
```

---

## Data Files

### Column structure (raw CSV)

| Column | Units | Description |
|---|---|---|
| `Datetime` | SGT (UTC+8) | Hourly timestamp, ISO 8601 format |
| `Latitude` | °N (WGS84) | Station latitude, constant within file |
| `Longitude` | °E (WGS84) | Station longitude, constant within file |
| `WindSpeed Ave (m/s)` | m s⁻¹ | Hourly mean wind speed at 2 m |
| `WindDir Ave (degrees)` | ° | Hourly mean wind direction (0–360°) |
| `AirTemp Ave (C)` | °C | Hourly mean air temperature at 2 m |
| `RelHum Ave (%)` | % | Hourly mean relative humidity at 2 m |
| `AtmPress Ave (hPa)` | hPa | Hourly mean atmospheric pressure |
| `GlobalRad Ave (W/m2)` | W m⁻² | Hourly mean global horizontal solar irradiance |
| `Rain Tot (mm)` | mm | Hourly rainfall total (WS02, WS16, WS35 only) |

Missing values are represented as empty fields (NaN).

### Imputed CSV — additional columns

Each variable has a companion flag column (e.g. `AirTemp_flag`):

| Flag value | Meaning |
|---|---|
| `0` | Original sensor observation |
| `1` | Stage 1 — linear interpolation (gap ≤ 6 h) |
| `2` | Stage 2 — XGBoost spatial regression |

To work only with original observations, filter rows where all flag columns equal `0`.

### Known instrument issues

- **WS17 atmospheric pressure**: sensor fault (recorded ~811 hPa throughout the year, ~200 hPa below expected). The `AtmPress Ave (hPa)` column at WS17 is `NaN` in both raw and imputed files. All other WS17 variables are valid.
- **WS38 global radiation**: complete pyranometer failure for the full year (0% valid). The imputed file provides a spatially-reconstructed estimate from neighbouring stations; treat with caution.

---

## Gap-Filling Methodology

Missing values were filled in two stages:

**Stage 1 — Linear interpolation**  
Gaps of ≤ 6 consecutive hours bounded by valid observations on both sides. Wind direction interpolated via shortest circular arc. Filled 3,302 values (3.3% of all fills).

**Stage 2 — XGBoost spatial regression (anomaly-based)**  
All remaining gaps. One XGBoost model per variable, trained on all available stations simultaneously. Targets expressed as deviations from each station's annual mean to remove systematic inter-station offsets (sensor calibration, elevation, microclimate). Features include simultaneous readings from the 8 nearest stations, station coordinates, and cyclic time encodings. Filled 95,967 values (96.7% of all fills).

**Leave-one-station-out validation** (5 held-out stations: WS01, WS08, WS12, WS24, WS28):

| Variable | MAE (24–720 h gaps) | R² (24–720 h gaps) |
|---|---|---|
| Air temperature | ~0.40–0.44 °C | ~0.92–0.94 |
| Relative humidity | ~1.6–1.8 % | ~0.91–0.95 |
| Atmospheric pressure | ~0.05–0.06 hPa | ~0.993–0.998 |
| Global solar irradiance | ~95–104 W m⁻² | ~0.63–0.68 |
| Wind speed | ~0.29–0.34 m s⁻¹ | negative (spatially decorrelated) |

Wind speed and wind direction cannot be reliably imputed at intra-campus scales due to building wake effects. Use only flag = 0 observations for wind analyses.

---

## Python Package

The companion `nus_campus_weather` package provides direct programmatic access to the dataset.

**Installation**

From the package directory:
```bash
pip install -e /path/to/nus_campus_weather
```
Or from PyPI once published:
```bash
pip install nus-campus-weather
```

**Quick start**

```python
import nus_campus_weather as ncw

ncw.set_data_dir("/path/to/NUS_CAMPUS_ScientificData")

# Load a single station (raw)
df = ncw.load_station("WS01")

# Load a single station (gap-filled)
df = ncw.load_station("WS01", imputed=True)

# Load all stations — wide pivot (8760 × 40)
air_temp = ncw.load_all(variable="AirTemp Ave (C)", imputed=True)

# Load all stations — long form
long_df = ncw.load_all(imputed=True)

# Station coordinates
meta = ncw.station_metadata()   # DataFrame with lat, lon indexed by station ID
```

**Plot functions**

```python
ncw.plot_completeness()          # Bubble map + monthly completeness heatmap
ncw.plot_climatology()           # Diurnal × seasonal climatology
ncw.plot_radials()               # Annual radial climatology (all 6 variables)
ncw.plot_imputation_examples()   # Representative gap-fill examples
```

Alternatively, set the data directory via environment variable:
```bash
export NUS_WEATHER_DATA="/path/to/NUS_CAMPUS_ScientificData"
```

---

## Reproducing the Analysis

Run the numbered scripts in order from the repository root:

```bash
python 1_data_quality_check.py        # → results/data_quality_summary.csv
python 2_plot_data_completeness.py    # → figures/Fig1, Fig2
python 3_plot_weather_data.py         # → figures/Fig3, Fig4, Fig5
python 4_plot_weather_radials.py      # → figures/Fig6a, Fig6b
python 5_impute_missing_data.py       # → data/imputed/, results/imputation_*.csv
python 6_plot_imputation_examples.py  # → figures/Fig7
```

**Dependencies**: Python ≥ 3.9, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`, `xgboost`, `contextily` (for map backgrounds)

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scipy xgboost contextily
```

---

## Citation

If you use this dataset, please cite:

> [Authors]. A High-Density Hourly Meteorological Dataset from a 40-Station Campus Weather Network at the National University of Singapore (2025). *Scientific Data* [year]. DOI: [to be assigned]

---

## Licence

Dataset: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)  
Code: [MIT](https://opensource.org/licenses/MIT)
