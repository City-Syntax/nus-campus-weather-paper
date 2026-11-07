# 🌦️ NUS Campus Meteorological Network — Hourly Dataset (2025)

Hourly meteorological observations from a **40-station automatic weather station (AWS) network** deployed across the National University of Singapore (NUS) main campus for the full calendar year 2025. To our knowledge, it is among the densest openly available, research-grade urban meteorological networks reported to date.

> ⚠️ **Wind data:** gap-filled wind speed and direction (flag 1–2) have little or no skill at this scale, because neighbouring stations only 100–200 m apart experience very different local wind. **Use only original observations (flag = 0) for wind analyses** — e.g. `ncw.load_station("WS01", imputed=True, observed_only=True)`.

---

## 📋 At a Glance

| Item | Detail |
|---|---|
| Stations | 40 (WS01–WS40): 32 ground-level, 8 rooftop |
| Coverage | 1 January – 31 December 2025 (8,760 hourly steps) |
| Location | NUS main campus, Singapore (≈ 1.30°N, 103.78°E) |
| Density | 23.5 stations/km² across ~170 hectares |
| Variables | Air temperature, relative humidity, atmospheric pressure, global horizontal irradiance, wind speed, wind direction; rainfall at WS02, WS16, WS35 |
| Temporal resolution | 1 hour |
| Timestamp completeness | 99.77% network mean; 34/40 stations at 100% |
| Variable completeness | 93.70% network mean over all station–variable pairs, after quality control (see `data_quality_assessment/`) |
| Quality control | Rule-based; faulty values removed, rainfall scale corrected, and air temperatures above 37.0 °C flagged, all logged in `data_quality_assessment/qc_log.csv` |
| Gap-filling | Two-stage, per variable: linear interpolation for short gaps + XGBoost spatial regression for longer gaps; every value flagged |
| Formats | CSV (quality-controlled observations + gap-filled with flags); companion Python package |

---

## 📁 Repository Structure

```
nus-campus-weather-paper/
│
├── data/
│   ├── raw/                             # Quality-controlled observations (40 CSV files, logged hours only)
│   │   └── NUS_CAMPUS_WS##_2025_Hourly.csv
│   └── imputed/                         # Gap-filled dataset (40 CSV files, 8,760 h each, + flag columns)
│       └── NUS_CAMPUS_WS##_2025_Hourly_imputed.csv
│
├── data_quality_assessment/
│   ├── qc_log.csv                       # Every value removed, corrected or flagged by quality control, with rule, action and reason
│   ├── data_quality_summary.csv         # Timestamp and variable completeness per station
│   ├── data_quality_network_summary.csv # Network-level completeness per variable
│   ├── imputation_counts.csv            # Values per variable by flag (Stage 1 / Stage 2 / not imputed / observed above 37 °C)
│   ├── imputation_evaluation.csv        # Leave-one-station-out validation, per station and gap length
│   ├── imputation_evaluation_summary.csv
│   ├── stage1_validation.csv            # Linear interpolation vs XGBoost on identical short gaps
│   ├── stage1_validation_summary.csv
│   └── imputation_plausibility.csv      # Range check of Stage 2 values
│
├── figures/                             # Manuscript figures (Fig1–Fig10) + additional_figures/
│
├── code/
│   ├── 0_quality_control.py             # Removes faulty values → data/raw/, qc_log.csv
│   ├── 1_data_quality_check.py          # Completeness → data_quality_assessment/
│   ├── 2_plot_data_completeness.py      # Fig 4
│   ├── 3_plot_weather_data.py           # Fig 6, Fig 7, Fig 8
│   ├── 4_plot_weather_radials.py        # Fig 5, Fig 9
│   ├── 5_impute_missing_data.py         # Gap filling + validation → data/imputed/, data_quality_assessment/
│   ├── 6_plot_imputation_examples.py    # Fig 10
│   └── preprocessing/                   # Historical only — not needed for re-use
│
├── requirements.txt                     # Package versions used for the release
├── LICENSE              # Pointer — repo is dual-licensed (see below)
├── LICENSE-CODE         # MIT — applies to code/ and scripts
├── LICENSE-DATA         # CC BY 4.0 — applies to data/, data_quality_assessment/, figures/
└── README.md
```

Figures 1–3 (photos and maps) are static image files. Figure 8 (wind-rose map) is produced by `code/3_plot_weather_data.py`; its basemap uses Mapbox tiles and needs an access token in the environment variable `MAPBOX_APIKEY`, and the roses are drawn without the basemap otherwise. The additional map figures (`AddFig4a`, `AddFig11`) also download a basemap at run time (CartoDB tiles, which now require an API key) and are drawn without it when tiles are unavailable.

---

## 📂 Data Files

### CSV column structure

| Column | Units | Description |
|---|---|---|
| `Datetime` | SGT (UTC+8) | Hourly timestamp, ISO 8601 |
| `Latitude` | °N WGS84 | Station latitude (constant within file) |
| `Longitude` | °E WGS84 | Station longitude (constant within file) |
| `Elevation (m)` | m | Station height above sea level (constant within file) |
| `Type` | – | Ground / Roof / Roof-3axis (constant within file) |
| `WindSpeed Ave (m/s)` | m s⁻¹ | Hourly vector-mean wind speed (magnitude of the mean wind vector; lower than the scalar mean when the direction varies within the hour) |
| `WindDir Ave (degrees)` | ° | Hourly vector-mean wind direction (0–360°) |
| `AirTemp Ave (C)` | °C | Hourly mean air temperature |
| `RelHum Ave (%)` | % | Hourly mean relative humidity |
| `AtmPress Ave (hPa)` | hPa | Hourly mean atmospheric pressure |
| `GlobalRad Ave (W/m2)` | W m⁻² | Hourly mean global horizontal irradiance at the sensor, including shading by nearby buildings and vegetation (not equivalent to unobstructed GHI) |
| `Rain Tot (mm)` | mm | Hourly rainfall total *(WS02, WS16, WS35 only)* |

Missing values are empty cells (`NaN` in pandas). The raw files hold one row for each hour the station logged: six stations lack 810 hourly rows in total, mostly at the end of December (WS06 317, WS38 435, WS17 43, WS31 12, WS20 2, WS30 1; see `Missing_Timestamps` in `data_quality_summary.csv`). The gap-filled files cover all 8,760 hours of 2025.

### Gap-filled CSV — flag columns

Each variable gains a companion `_flag` column (e.g. `AirTemp_flag`), and rainfall has `Rain_flag` at WS02, WS16 and WS35:

| Flag | Meaning |
|---|---|
| `0` | Original observation |
| `1` | Stage 1 — linear interpolation (short gap, variable-specific limit) |
| `2` | Stage 2 — XGBoost spatial regression |
| `3` | Not measured / not imputed — left empty (rainfall is never gap-filled) |
| `4` | Original air-temperature observation above 37.0 °C (Singapore's highest recorded temperature) — kept, but probably includes radiation error (rule T1) |

> To use only original data, keep rows where the flag is 0 (or 4 for air temperature), or use `observed_only=True` in the Python package.

### ⚠️ Quality control and known issues

`code/0_quality_control.py` removes faulty values from `data/raw/`, corrects the rainfall scale, and lists every removal, correction and flagged value in `data_quality_assessment/qc_log.csv`. The unmasked sensor values are preserved in release v1.0.0 (Zenodo [10.5281/zenodo.20477761](https://doi.org/10.5281/zenodo.20477761)). In total 32,236 values of the six main variables (1.5%) were removed, together with 1,180 rainfall values (see below):

- **Global irradiance, WS32 and WS39** — pyranometers non-functional (near-constant output with no diurnal cycle); whole year removed. Together with **WS38** (no pyranometer; fully shaded semi-outdoor site), these three stations have no irradiance data and are **not imputed** (flag 3).
- **Global irradiance, WS07 (Sep–Dec), WS19 (Feb–Mar), WS23 (Mar–Sep), WS26 (May, Aug–Dec)** — corrupted channel (spurious night-time readings up to 1,500 W m⁻²); affected months removed. Isolated night-time readings above 20 W m⁻² removed elsewhere.
- **Atmospheric pressure, WS17 (Jan–Apr)** — sensor fault (readings of 460–478 hPa); removed. Values from June onward are valid.
- **Wind, WS17 (December)** — anemometer malfunction: stalled records between 7 and 28 Dec, and implausible hourly means up to 17 m/s on 20–26 Dec; removed. Short runs of stalled wind records at WS01 and WS03 also removed.

- **Rainfall, WS02, WS16, WS35** — the hourly aggregation stored 0 mm in the 1,180 hours in which the station recorded no other variable (536 h at WS02, 642 h at WS35, 2 h at WS16, mostly April–May). From v1.2.0 these values are missing (rule R1; `Rain_flag` = 3). Rainfall is not gap-filled.
- **Rainfall amounts, WS02, WS16, WS35** — the loggers counted each tip of the 0.2-mm tipping bucket (LSI LASTEM DQA230.1) as 0.1 mm, so v1.0.0 and v1.1.0 hold half the true rainfall. From v1.2.0 all rainfall values are doubled (rule R2, action `corrected`). After the correction, annual totals over days recorded by both agree with the Meteorological Service Singapore gauge at Kent Ridge Road (S71) to within 2%, and daily totals correlate at r = 0.93–0.98.
- **Daytime air temperature** — the naturally ventilated radiation shields read high in strong sunshine and light wind. The 294 hourly values above 37.0 °C (15 stations, 163 of them at WS22; all between 10:00 and 17:00) are kept and flagged 4 (rule T1). Other midday readings in these conditions, at ground and rooftop stations, can also be biased high; treat daytime maxima, especially at WS22, as upper estimates.

Removed periods are then treated as ordinary gaps by the gap-filling pipeline (except the three irradiance stations above). The imputation procedure is described in the *Gap-filling methodology* section of the paper and implemented in `code/5_impute_missing_data.py`.

---

## 🔧 Gap-Filling Methodology

### Stage 1 — Linear interpolation (short gaps)
Gaps bounded by valid observations are interpolated linearly up to a **variable-specific limit**: 1 h for air temperature and relative humidity, 3 h for global irradiance, 12 h for wind speed and direction; atmospheric pressure is never interpolated. Wind direction is interpolated along the shortest circular arc. The limits are the longest gap lengths for which linear interpolation was at least as accurate as the spatial model on identical synthetic gaps (`stage1_validation_summary.csv`). **2,365 values filled.**

### Stage 2 — XGBoost spatial regression
All remaining gaps. One model per variable, trained on all stations. Targets are anomalies from each station's mean, removing calibration, elevation and exposure offsets between stations. Features: same-hour readings of all variables at the 8 nearest stations, station coordinates, cyclic hour-of-day and day-of-year terms. **103,751 values filled.** Station–variables with less than 10% of the year observed are not reconstructed (flag 3: irradiance at WS32, WS38, WS39; 26,280 values).

In total 106,116 values (5.05% of all 2,102,400 hourly values) are filled.

### Validation (leave-one-station-out)

Five complete stations (WS01, WS08, WS12, WS24, WS28) are withheld from training; synthetic gaps of 24–720 h are cut from their records and filled. Typical errors of filled values:

| Variable | flag = 1 (linear, within limit) | flag = 2 (spatial, 24–720 h gaps) | R² (flag 2) |
|---|---|---|---|
| Air temperature | 0.28 °C | 0.39–0.43 °C | 0.91–0.94 |
| Relative humidity | 1.0 % | 1.5–1.7 % | 0.92–0.95 |
| Atmospheric pressure | not used | 0.06–0.07 hPa | 0.995–0.998 |
| Global irradiance | 44–62 W m⁻² | 76–79 W m⁻² | 0.74–0.78 |
| Wind speed | 0.17–0.33 m s⁻¹ | 0.30–0.35 m s⁻¹ | negative (no skill) |
| Wind direction | 18–44° (circular) | 66–78° (circular) | – |

Values are mean absolute errors; see `imputation_evaluation_summary.csv` and `stage1_validation_summary.csv`.

---

## 🐍 Python Package

A companion package `nus_campus_weather` is available on PyPI ([source](https://github.com/City-Syntax/nus-campus-weather)).

```bash
pip install nus-campus-weather
```

```python
import nus_campus_weather as ncw          # data download from Zenodo on first use

df      = ncw.load_station("WS01")                       # single station, observations
df_imp  = ncw.load_station("WS01", imputed=True)         # gap-filled, with flag columns
df_obs  = ncw.load_station("WS01", imputed=True, observed_only=True)   # filled values → NaN
wide_df = ncw.load_all(variable="AirTemp Ave (C)", imputed=True)       # 8760 × 40
meta    = ncw.station_metadata()                         # lat, lon, elevation, type

ncw.plot_completeness(); ncw.plot_climatology(); ncw.plot_radials(); ncw.plot_imputation_examples()
```

To work from a local clone instead: `ncw.set_data_dir("/path/to/nus-campus-weather-paper")` or set `NUS_WEATHER_DATA`.

---

## ▶️ Reproducing the Analysis

Run the scripts in order from the repository root:

```bash
pip install -r requirements.txt
python code/0_quality_control.py           # → data/raw/ (cleaned), data_quality_assessment/qc_log.csv
python code/1_data_quality_check.py        # → data_quality_assessment/data_quality_*.csv
python code/2_plot_data_completeness.py    # → figures/Fig4
python code/3_plot_weather_data.py         # → figures/Fig6, Fig7, Fig8
python code/4_plot_weather_radials.py      # → figures/Fig5, Fig9
python code/5_impute_missing_data.py       # → data/imputed/, data_quality_assessment/
python code/6_plot_imputation_examples.py  # → figures/Fig10
```

`0_quality_control.py` is idempotent; starting from the v1.0.0 files it reproduces the cleaned values and the log exactly (the `Type` and `Elevation (m)` columns were added to the files for v1.1.0). XGBoost results can differ slightly between library versions; `requirements.txt` lists the versions used for this release.

---

## 📄 Citation

Lim, J. et al. *An hourly meteorological dataset from a high-density 40-station campus weather network in tropical Singapore.* Scientific Data (under review). Dataset: Zenodo, https://doi.org/10.5281/zenodo.20477760 (all versions; cite the version DOI shown on the Zenodo record you used).

---

## 📜 Licence

This repository is **dual-licensed**:

| Scope | Licence | File |
|---|---|---|
| Source code, scripts, documentation (`code/`, `README.md`, build config) | [MIT](https://opensource.org/licenses/MIT) | [`LICENSE-CODE`](LICENSE-CODE) |
| Dataset, derived results, figures (`data/`, `data_quality_assessment/`, `figures/`) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [`LICENSE-DATA`](LICENSE-DATA) |

See [`LICENSE`](LICENSE) for the top-level summary.

© 2026 City Syntax Lab and contributors.
