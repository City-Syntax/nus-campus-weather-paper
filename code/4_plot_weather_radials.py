import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patches import Circle
import warnings
warnings.filterwarnings('ignore')
import glob, os, calendar

# ── Style ─────────────────────────────────────────────────────────────────────
FONT = "Palatino Linotype"
plt.rcParams.update({
    "font.family":       FONT,
    "figure.dpi":        150,
    "savefig.facecolor": "white",
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIRIDIS    = cm.viridis
HOUR_CMAP  = cm.twilight_shifted      # circular: 00h and 23h share the same hue
VAR_COLORS = [VIRIDIS(x) for x in np.linspace(0.08, 0.92, 6)]
MONTH_ABBR = [calendar.month_abbr[m] for m in range(1, 13)]

VARS = [
    ("AirTemp Ave (C)",       "Air Temperature",   "°C",    VAR_COLORS[0]),
    ("RelHum Ave (%)",        "Relative Humidity", "%",     VAR_COLORS[1]),
    ("WindSpeed Ave (m/s)",   "Wind Speed",        "m/s",   VAR_COLORS[2]),
    ("AtmPress Ave (hPa)",    "Atm. Pressure",     "hPa",   VAR_COLORS[3]),
    ("GlobalRad Ave (W/m2)",  "Solar Radiation",   "W/m²",  VAR_COLORS[4]),
    ("WindDir Ave (degrees)", "Wind Direction",    "°",     VAR_COLORS[5]),
]

PRESSURE_OUTLIER = "WS17"
R_IN,  R_OUT = 0.30, 0.90    # inner / outer radius of data ring

GAUGE_STATIONS = ['WS02', 'WS16', 'WS35']

T_INNER_REF  = 18.0      # °C anchor for inner ring
T_OUTER_REF  = 42.0      # °C anchor for outer ring
R_INNER_BASE = 0.8       # radial units
R_OUTER_BASE = 3.0
SOLAR_VMIN   = 100       # W/m² colormap low clip
SOLAR_VMAX   = 550       # W/m² colormap high clip
RH_THRESHOLD = 82        # % — blobs only above this RH
RAIN_MIN     = 0.5       # mm — blobs only above this rainfall

SOLAR_CMAP = LinearSegmentedColormap.from_list(
    'solar', ['#fff3a0', '#ffee00', '#ff8c00', '#ff0000', '#8b0000'], N=256)
SOLAR_NORM = Normalize(vmin=SOLAR_VMIN, vmax=SOLAR_VMAX)

RAIN_COLOR = '#4a90d9'
RH_COLOR   = '#9b59b6'

MONTH_NAMES   = ['JAN','FEB','MAR','APR','MAY','JUN',
                 'JUL','AUG','SEP','OCT','NOV','DEC']
MONTH_MID_DOY = [16, 46, 74, 106, 136, 167, 197, 228, 259, 289, 320, 350]
MONTH_STARTS  = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]

# ── Load ──────────────────────────────────────────────────────────────────────
FILES = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "NUS_CAMPUS_WS*.csv")))
dfs = []
print("Loading data...")
for fpath in FILES:
    ws = os.path.basename(fpath).split("_")[2]
    df = pd.read_csv(fpath, parse_dates=["Datetime"])
    df["station"] = ws
    df["hour"]    = df["Datetime"].dt.hour
    df["doy"]     = df["Datetime"].dt.dayofyear
    dfs.append(df)

all_df   = pd.concat(dfs, ignore_index=True)
stations = sorted(all_df["station"].unique())
print(f"Loaded {len(stations)} stations, {len(all_df):,} rows.")

# ── Helpers ───────────────────────────────────────────────────────────────────
def circ_mean(series):
    rad = np.deg2rad(series.dropna())
    if len(rad) == 0:
        return np.nan
    return np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360

def doy_to_theta(doy):
    """Day-of-year (1-365) → clockwise radian from top (north = 1 Jan)."""
    return (np.asarray(doy, float) - 1) / 365 * 2 * np.pi

def to_radius(val, vmin, vmax):
    frac = np.clip((np.asarray(val, float) - vmin) / (vmax - vmin), 0, 1)
    return frac * (R_OUT - R_IN) + R_IN

def temp_to_r(T):
    """Map temperature (°C) to radial position."""
    return R_INNER_BASE + (T - T_INNER_REF) / (T_OUTER_REF - T_INNER_REF) * (R_OUTER_BASE - R_INNER_BASE)

def setup_polar(ax):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)
    ax.set_facecolor("white")
    tc = np.linspace(0, 2 * np.pi, 360)
    ax.plot(tc, np.full(360, R_IN),  color="#dddddd", lw=0.5, ls="--", zorder=1)
    ax.plot(tc, np.full(360, R_OUT), color="#dddddd", lw=0.5, ls="--", zorder=1)

def add_month_labels(ax, fontsize=6.5, r_text=1.035):
    for m in range(1, 13):
        doy_m = pd.Timestamp(f"2025-{m:02d}-01").timetuple().tm_yday
        t = doy_to_theta(doy_m)
        ax.text(t, r_text, MONTH_ABBR[m - 1], ha="center", va="bottom",
                fontsize=fontsize, fontfamily=FONT, color="#666666", zorder=9)
        ax.plot([t, t], [R_OUT, R_OUT + 0.025], color="#bbbbbb", lw=0.7, zorder=7)

def add_center_label(ax, lines, fontsize=7):
    ax.text(0, 0, "\n".join(lines), ha="center", va="center",
            fontsize=fontsize, fontfamily=FONT, color="#333333",
            multialignment="center", zorder=10)

def annotate_scale(ax, vmin_v, vmax_v, unit):
    # Oct position (right side of ring, less crowded)
    t_ref = doy_to_theta(274)
    ax.text(t_ref, R_IN - 0.065, f"{vmin_v:.1f} {unit}",
            ha="center", va="center", fontsize=5.2, fontfamily=FONT,
            color="#888888", zorder=10)
    ax.text(t_ref, R_OUT + 0.055, f"{vmax_v:.1f} {unit}",
            ha="center", va="center", fontsize=5.2, fontfamily=FONT,
            color="#888888", zorder=10)

def add_panel_legend(ax, lc, line_label, has_band=True, mean_color=None):
    mc = mean_color if mean_color is not None else lc
    handles = [
        Line2D([0], [0], color=mc, lw=1.3, alpha=0.85, label=line_label),
    ]
    if has_band:
        handles.append(Patch(facecolor=mc, alpha=0.28, label="±1 s.d. (40 stations)"))
    ax.legend(
        handles=handles,
        bbox_to_anchor=(0.5, -0.03), loc="upper center",
        fontsize=5.2, framealpha=0.92, edgecolor="#dddddd", fancybox=False,
        ncol=len(handles), columnspacing=0.7,
        handlelength=1.2, handletextpad=0.4, borderpad=0.4,
        prop={"family": FONT},
    )

def draw_hourly_wind_rose(ax, src_df, n_sectors=16):
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_facecolor("white")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.spines["polar"].set_visible(False)

    sector_deg  = 360 / n_sectors
    wd = src_df[["hour", "WindDir Ave (degrees)"]].dropna().copy()
    wd["sector"] = (wd["WindDir Ave (degrees)"] / sector_deg).astype(int) % n_sectors

    total = len(wd)
    pivot = (wd.groupby(["hour", "sector"]).size()
               .unstack(fill_value=0)
               .reindex(index=range(24), columns=range(n_sectors), fill_value=0)
               / total * 100)            # convert to %

    sector_theta = np.deg2rad(np.arange(n_sectors) * sector_deg + sector_deg / 2)
    bar_width    = np.deg2rad(sector_deg * 0.90)

    # Stacked bars: one layer per hour
    bottoms = np.zeros(n_sectors)
    for h in range(24):
        heights = pivot.loc[h].values
        ax.bar(sector_theta, heights, width=bar_width, bottom=bottoms,
               color=HOUR_CMAP(h / 23), alpha=0.88, edgecolor="none", zorder=3)
        bottoms += heights

    max_freq = bottoms.max()
    r_max    = max(np.ceil(max_freq / 2) * 2, 4)   # round up to nearest 2 %
    ax.set_ylim(0, r_max * 1.30)

    # Concentric reference circles + % labels
    tc = np.linspace(0, 2 * np.pi, 360)
    for pct in np.arange(2, r_max + 0.1, 2):
        ax.plot(tc, np.full(360, pct), color="#e8e8e8", lw=0.5, zorder=1)
    for pct in np.arange(4, r_max + 0.1, 4):
        ax.text(np.deg2rad(12), pct, f"{pct:.0f}%",
                fontsize=4.8, fontfamily=FONT, color="#aaaaaa",
                ha="left", va="bottom", zorder=6)

    # Compass labels
    for label_d, angle_d in zip(["N","NE","E","SE","S","SW","W","NW"],
                                  [0, 45, 90, 135, 180, 225, 270, 315]):
        ax.text(np.deg2rad(angle_d), r_max * 1.20, label_d,
                ha="center", va="center", fontsize=6.5,
                fontfamily=FONT, fontweight="bold", color="#555555", zorder=7)

    # White inner disc + centre label
    r_inner = r_max * 0.13
    ax.fill(tc, np.full(360, r_inner), color="white", zorder=5)
    ax.text(0, 0, "Wind\nDirection\n(% freq.)",
            ha="center", va="center", fontsize=6.5,
            fontfamily=FONT, color="#333333",
            multialignment="center", zorder=10)


def draw_radial(ax, doy, hour, val, vmin, vmax, lc,
                std=None, s=0.9, alpha=0.68,
                show_mean=True, mean_mask=None, mean_color=None):
    thetas = doy_to_theta(doy)
    r      = to_radius(val, vmin, vmax)

    # Scatter: all (doy, hour) points, colour = hour of day
    ax.scatter(thetas, r, c=hour, cmap=HOUR_CMAP, vmin=0, vmax=23,
               s=s, alpha=alpha, lw=0, zorder=3, rasterized=True)

    if show_mean or (std is not None):
        df_tmp = pd.DataFrame({
            "doy": doy, "val": val,
            "std": std if std is not None else np.zeros(len(doy))
        })
        # Optionally restrict mean/band to a subset of hours
        df_for_mean = df_tmp[mean_mask] if mean_mask is not None else df_tmp
        dg = df_for_mean.groupby("doy").agg(
            mv=("val", "mean"), ms=("std", "mean")
        ).reset_index()

        mv_loop = np.append(dg["mv"].values, dg["mv"].values[0])
        ms_loop = np.append(dg["ms"].values, dg["ms"].values[0])
        t_loop  = np.append(doy_to_theta(dg["doy"].values),
                            doy_to_theta(dg["doy"].values[0]))

        if std is not None:
            r_lo = to_radius(mv_loop - ms_loop, vmin, vmax)
            r_hi = to_radius(mv_loop + ms_loop, vmin, vmax)
            band_color = mean_color if mean_color is not None else lc
            ax.fill_between(t_loop, r_lo, r_hi, color=band_color, alpha=0.22, zorder=2)

        if show_mean:
            ax.plot(t_loop, to_radius(mv_loop, vmin, vmax),
                    color=mean_color if mean_color is not None else lc,
                    lw=0.9, alpha=0.85, zorder=4)

    # White inner disc — keeps centre clean for label
    tc = np.linspace(0, 2 * np.pi, 360)
    ax.fill(tc, np.full(360, R_IN), color="white", zorder=5)

# ── Per-variable plot config for Fig 9 ──────────────────────────────────────
# keys: center, line_label, has_band, show_mean, daytime_only
VAR_CFG = {
    "AirTemp Ave (C)": dict(
        center=["Air Temp.", "(°C)", "inner=min, outer=max"],
        line_label="Network daily mean",
        has_band=True, show_mean=True, daytime_only=False,
    ),
    "RelHum Ave (%)": dict(
        center=["Rel. Humidity", "(%)", "inner=min, outer=max"],
        line_label="Network daily mean",
        has_band=True, show_mean=True, daytime_only=False,
    ),
    "WindSpeed Ave (m/s)": dict(
        center=["Wind Speed", "(m/s)", "inner=min, outer=max"],
        line_label="Network daily mean",
        has_band=True, show_mean=True, daytime_only=False,
    ),
    "AtmPress Ave (hPa)": dict(
        center=["Atm. Pressure", "(hPa)", "WS17 excluded", "inner=min, outer=max"],
        line_label="Network daily mean (WS17 excl.)",
        has_band=True, show_mean=True, daytime_only=False,
    ),
    "GlobalRad Ave (W/m2)": dict(
        center=["Solar Rad.", "(W/m²)", "inner=min, outer=max"],
        line_label="Daytime mean  07–18h",
        has_band=True, show_mean=True, daytime_only=True,
    ),
    "WindDir Ave (degrees)": dict(
        center=["Wind Dir.", "(°)", "radius = direction", "0°/360°=N  90°=E", "180°=S  270°=W"],
        line_label=None,      # no mean line — circular data on linear axis
        has_band=False, show_mean=False, daytime_only=False,
    ),
}

# Mean line colors for Fig 9: dark end of each variable's own colormap
MEAN_COLORS_6A = {
    "AirTemp Ave (C)":       "#bd0026",   # YlOrRd dark red
    "RelHum Ave (%)":        "#225ea8",   # YlGnBu deep blue
    "WindSpeed Ave (m/s)":   "#006d2c",   # BuGn deep green
    "AtmPress Ave (hPa)":    "#023858",   # PuBu dark navy
    "GlobalRad Ave (W/m2)":  "#e05c00",   # afmhot orange
}

# ═════════════════════════════════════════════════════════════════════════════
# Fig 5 — Combined weather radial plot
# ═════════════════════════════════════════════════════════════════════════════
R_20 = temp_to_r(20)
R_35 = temp_to_r(35)

combineddf = all_df.copy()
combineddf['Datetime'] = pd.to_datetime(combineddf['Datetime'])
combineddf['Date']     = combineddf['Datetime'].dt.date
combineddf['Hour']     = combineddf['Datetime'].dt.hour

# Rooftop: 8 stations — temperature & solar
roof     = combineddf[(combineddf['Type'] == 'Roof') | (combineddf['Type'] == 'Roof-3axis')]
hourly_r = roof.groupby(['Date','Hour']).agg(
    AirTemp=('AirTemp Ave (C)','mean'), GlobalRad=('GlobalRad Ave (W/m2)','mean')).reset_index()
daily_rt = hourly_r.groupby('Date').agg(
    temp_max=('AirTemp','max'), temp_min=('AirTemp','min')).reset_index()
solar_r  = (hourly_r[hourly_r['Hour'].between(7,19)]
            .groupby('Date')['GlobalRad'].mean().rename('solar_day'))
daily_rt = daily_rt.merge(solar_r, on='Date')

# Rain: mean of 3 gauge stations
dfs_rain = {}
GAUGE_PATH = ROOT+'/data/imputed/NUS_CAMPUS_{ws}_2025_Hourly_imputed.csv'
for ws in GAUGE_STATIONS:
    d = pd.read_csv(GAUGE_PATH.format(ws=ws))
    d['Datetime'] = pd.to_datetime(d['Datetime'])
    d['Date']     = d['Datetime'].dt.date
    dfs_rain[ws]  = d
rain_avg = (pd.concat([d.groupby('Date')['Rain Tot (mm)'].sum().rename(ws)
                        for ws, d in dfs_rain.items()], axis=1)
              .mean(axis=1).rename('rain'))

daily_roof = daily_rt.merge(rain_avg, on='Date')
daily_roof['Date'] = pd.to_datetime(daily_roof['Date'])
daily_roof = daily_roof.sort_values('Date').reset_index(drop=True)

# Ground: 32 stations — temperature, solar, RH
ground   = combineddf[combineddf['Type'] == 'Ground']
hourly_g = ground.groupby(['Date','Hour']).agg(
    AirTemp=('AirTemp Ave (C)','mean'), RelHum=('RelHum Ave (%)','mean'),
    GlobalRad=('GlobalRad Ave (W/m2)','mean')).reset_index()
daily_gnd = hourly_g.groupby('Date').agg(
    temp_max=('AirTemp','max'), temp_min=('AirTemp','min'),
    rh_mean=('RelHum','mean')).reset_index()
solar_gnd = (hourly_g[hourly_g['Hour'].between(7,19)]
             .groupby('Date')['GlobalRad'].mean().rename('solar_day'))
daily_gnd = daily_gnd.merge(solar_gnd, on='Date')
daily_gnd['Date'] = pd.to_datetime(daily_gnd['Date'])
daily_gnd = daily_gnd.sort_values('Date').reset_index(drop=True)

# ── BLOB FUNCTIONS ────────────────────────────────────────────────────────────
RAIN_MAX   = daily_roof['rain'].max()
RH_MAX_VAL = daily_gnd['rh_mean'].max()

def rain_to_blob(r):
    return 0.0 if r < RAIN_MIN else 0.04 + 0.32 * (r / RAIN_MAX)

def rh_to_blob(rh):
    if rh < RH_THRESHOLD: return 0.0
    return 0.04 + 0.32 * ((rh - RH_THRESHOLD) / (RH_MAX_VAL - RH_THRESHOLD))

# ── RADIAL DRAW FUNCTION ──────────────────────────────────────────────────────
N      = 365
ANGLES = np.linspace(0, 2*np.pi, N, endpoint=False)

def draw_radial_combined(ax, daily, blob_col, blob_fn, blob_color, panel_label):
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    r_min = np.clip(temp_to_r(daily['temp_min'].values), R_INNER_BASE, R_OUTER_BASE)
    r_max = np.clip(temp_to_r(daily['temp_max'].values), R_INNER_BASE, R_OUTER_BASE)

    # temperature reference rings
    for T, lbl in [(20,'20°C'),(25,'25°C'),(30,'30°C'),(35,'35°C')]:
        r = temp_to_r(T)
        ax.plot(np.linspace(0,2*np.pi,500), [r]*500,
                color='#aaaaaa', linewidth=1.4, linestyle=(0,(4,3)), zorder=1)
        ax.text(np.radians(197), r, lbl, color='#888888', fontsize=14,
                ha='center', va='center', fontweight='bold', zorder=2)

    # daily temperature lines coloured by solar radiation
    for i, row in daily.iterrows():
        c = SOLAR_CMAP(SOLAR_NORM(row['solar_day']))
        ax.plot([ANGLES[i], ANGLES[i]], [r_min[i], r_max[i]],
                color=c, linewidth=2.0, alpha=0.92, solid_capstyle='round', zorder=3)

    # blob markers
    for i, row in daily.iterrows():
        rb = blob_fn(row[blob_col])
        if rb > 0:
            r_mid = (r_min[i] + r_max[i]) / 2
            x = r_mid * np.cos(np.pi/2 - ANGLES[i])
            y = r_mid * np.sin(np.pi/2 - ANGLES[i])
            ax.add_patch(Circle((x, y), rb, color=blob_color, alpha=0.30,
                                zorder=4, transform=ax.transData._b))

    # month separator lines: from inside 20°C ring to outside 35°C ring
    for doy in MONTH_STARTS:
        idx = min(doy-1, N-1)
        ax.plot([ANGLES[idx], ANGLES[idx]], [R_20 - 0.12, R_35 + 0.12],
                color='#bbbbbb', linewidth=0.8, zorder=2)

    ax.set_ylim(0, R_OUTER_BASE + 0.65)
    ax.set_yticks([]); ax.set_xticks([])
    ax.spines['polar'].set_visible(False)
    ax.text(0, 0, panel_label, color='#444444', fontsize=16,
            fontweight='bold', ha='center', va='center')

def add_month_labels_combined(fig, ax, label_r, fontsize=18):
    """Place month labels using display-space transform for correct rotation."""
    for doy, mname in zip(MONTH_MID_DOY, MONTH_NAMES):
        idx        = min(doy-1, N-1)
        t_m        = ANGLES[idx]
        xy_disp    = ax.transData.transform((t_m, label_r))
        cx, cy     = ax.transData.transform((0, 0))
        dx, dy     = xy_disp[0]-cx, xy_disp[1]-cy
        disp_angle = np.degrees(np.arctan2(dy, dx))
        rot        = disp_angle - 90
        if dy < 0:
            rot += 180
        xy_fig = fig.transFigure.inverted().transform(xy_disp)
        fig.text(xy_fig[0], xy_fig[1], mname,
                 color='#333333', fontsize=fontsize, fontweight='bold',
                 ha='center', va='center', rotation=rot)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Fig 5: Combined weather radial (two-panel chart)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
print("Building Fig 5 (combined weather radial)...")

fig = plt.figure(figsize=(24, 13), facecolor='white')

ax1 = fig.add_axes([0.03, 0.08, 0.43, 0.82], polar=True, facecolor='white')
ax2 = fig.add_axes([0.44, 0.08, 0.43, 0.82], polar=True, facecolor='white')
draw_radial_combined(ax1, daily_gnd,  'rh_mean', rh_to_blob, RH_COLOR,
            'GROUND LEVEL\n32 STATIONS\n(avg)')
draw_radial_combined(ax2, daily_roof, 'rain', rain_to_blob, RAIN_COLOR,
            'ROOFTOP\n8 STATIONS\n(avg)')

fig.canvas.draw()
label_r = R_35 + 0.30
add_month_labels_combined(fig, ax1, label_r, fontsize=18)
add_month_labels_combined(fig, ax2, label_r, fontsize=18)

# solar colorbar
sm = ScalarMappable(cmap=SOLAR_CMAP, norm=SOLAR_NORM)
sm.set_array([])
cbar_ax = fig.add_axes([0.85, 0.23, 0.012, 0.54])
cbar    = fig.colorbar(sm, cax=cbar_ax)
# cbar    = fig.colorbar(sm, location="right")
cbar.set_ticks([100,200,300,400,500])
cbar.set_ticklabels(['100','200','300','400','500+'])
cbar.set_label('Daytime mean solar (07–19h) (W/m²)', color='#444444',
               fontsize=18, labelpad=10)
cbar.ax.yaxis.set_tick_params(color='#666666')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#444444', fontsize=12)
cbar.outline.set_edgecolor('#cccccc')

# titles
fig.text(0.50, 0.855,
         'WEATHER RADIAL  ·  NUS KENT RIDGE CAMPUS  ·  2025',
         color='#222222', fontsize=18, fontweight='bold', ha='center', va='top')
fig.text(0.50, 0.826,
         'Line = daily temp range  ·  Colour = daytime solar radiation (07–19h)  ·  Blob = rainfall (left) / relative humidity >82% (right)',
         color='#888888', fontsize=12, ha='center', va='top')

rain_entries = [(rain_to_blob(5),  '5 mm'),
                (rain_to_blob(20), '20 mm'),
                (rain_to_blob(40), '40 mm'),
                (rain_to_blob(80), '80 mm')]
rh_entries   = [(rh_to_blob(83),  '83%'),
                (rh_to_blob(87),  '87%'),
                (rh_to_blob(91),  '91%'),
                (rh_to_blob(95),  '95%')]

def draw_blob_legend(ax, entries, color, title, unit_label):
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    max_b     = max(rb for rb, _ in entries)
    scale     = 0.25 / max_b
    ys        = np.linspace(0.82, 0.12, len(entries))
    cx        = 0.22
    for (rb, lbl), y in zip(entries, ys):
        ax.add_patch(Circle((cx, y), rb*0.9, color=color, alpha=0.30,
                                zorder=4, transform=ax.transData._b))
        ax.text(cx + max_b * scale + 0.25, y, lbl,
                color='#333333', fontsize=10, va='center', ha='left')
    ax.text(0.50, 0.97, title, color='#222222', fontsize=12,
            ha='center', va='top', fontweight='bold')
    ax.text(0.50, 0.90, unit_label, color='#888888', fontsize=8,
            ha='center', va='top')

    rect = plt.Rectangle((0.02,0.02), 0.96, 0.96,
                          fill=False, edgecolor='#dddddd', linewidth=0.8,
                          transform=ax.transAxes)
    ax.add_patch(rect)

blobax1 = fig.add_axes([0.03, 0.16, 0.0825, 0.25], facecolor='white')
blobax2 = fig.add_axes([0.435, 0.16, 0.0825, 0.25], facecolor='white')
draw_blob_legend(blobax1, rh_entries,  RH_COLOR,
                 'REL. HUMIDITY','daily mean (%)')
draw_blob_legend(blobax2, rain_entries, RAIN_COLOR,
                 'RAINFALL',     'daily total (mm)')

OUT_CHART  = ROOT+'/figures/Fig5_Combined_WeatherRadial.png'
plt.savefig(OUT_CHART, dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {OUT_CHART}")

# ═════════════════════════════════════════════════════════════════════════════
# Fig 9 — Option A: 6 radials, network mean, all variables
# ═════════════════════════════════════════════════════════════════════════════
print("Building Fig 9 (Option A — network radials)...")

fig9, axs6a = plt.subplots(2, 3, figsize=(18, 13),
                              subplot_kw={"projection": "polar"})
fig9.patch.set_facecolor("white")

for ax, (col, label, unit, lc) in zip(axs6a.flat, VARS):
    cfg = VAR_CFG[col]
    src = all_df.copy()
    if col == "AtmPress Ave (hPa)":
        src.loc[src["station"] == PRESSURE_OUTLIER, col] = np.nan

    # ── Wind Direction: hourly-stratified wind rose (different geometry) ─────
    if col == "WindDir Ave (degrees)":
        draw_hourly_wind_rose(ax, src)
        wr_handles = [
            Patch(facecolor=HOUR_CMAP(0.0),       alpha=0.88, label="00h (midnight)"),
            Patch(facecolor=HOUR_CMAP(12 / 23),   alpha=0.88, label="12h (noon)"),
            Patch(facecolor=HOUR_CMAP(18 / 23),   alpha=0.88, label="18h (dusk)"),
        ]
        ax.legend(handles=wr_handles,
                  title="Stacked by hour", title_fontsize=5.0,
                  bbox_to_anchor=(0.5, -0.03), loc="upper center",
                  fontsize=5.2, framealpha=0.92, edgecolor="#dddddd",
                  fancybox=False, ncol=3, columnspacing=0.5,
                  handlelength=0.9, handletextpad=0.4, borderpad=0.4,
                  prop={"family": FONT})
        continue   # skip the radial drawing below

    # ── All other variables: annual radial ────────────────────────────────────
    grp_m = src.groupby(["doy", "hour"])[col].mean().reset_index()
    grp_s = src.groupby(["doy", "hour"])[col].std().reset_index()
    grp_s[col] = grp_s[col].fillna(0)
    vmin_v = float(np.nanpercentile(grp_m[col], 2))
    vmax_v = float(np.nanpercentile(grp_m[col], 98))

    mean_mask = grp_m["hour"].between(7, 18).values if cfg["daytime_only"] else None

    mc = MEAN_COLORS_6A.get(col)
    setup_polar(ax)
    draw_radial(
        ax,
        doy        = grp_m["doy"].values,
        hour       = grp_m["hour"].values,
        val        = grp_m[col].values,
        vmin       = vmin_v,
        vmax       = vmax_v,
        lc         = lc,
        std        = grp_s[col].values,
        s          = 1.0, alpha=0.70,
        show_mean  = cfg["show_mean"],
        mean_mask  = mean_mask,
        mean_color = mc,
    )
    add_month_labels(ax)
    annotate_scale(ax, vmin_v, vmax_v, unit)
    add_center_label(ax, cfg["center"])
    add_panel_legend(ax, lc, cfg["line_label"], has_band=cfg["has_band"], mean_color=mc)

# Shared hour-of-day colorbar
sm_h = ScalarMappable(cmap=HOUR_CMAP, norm=Normalize(vmin=0, vmax=23))
sm_h.set_array([])
cax_a = fig9.add_axes([0.93, 0.14, 0.013, 0.72])
cb_a  = fig9.colorbar(sm_h, cax=cax_a)
cb_a.set_label("Hour of Day (all panels)", fontsize=8.5, fontfamily=FONT, labelpad=8)
cb_a.set_ticks([0, 6, 12, 18, 23])
cb_a.set_ticklabels(["00h", "06h", "12h", "18h", "23h"])
cb_a.ax.tick_params(labelsize=8)
for sp in cb_a.ax.spines.values():
    sp.set_linewidth(0.3)

fig9.suptitle(
    "Annual Weather Radials — NUS Campus Meteorological Network (2025)\n"
    "40 stations  ·  Jan at top, clockwise  ·  Dot colour = hour of day  ·  "
    "Radius = variable value  ·  Dashed circles = data min / max",
    fontsize=11, fontfamily=FONT, fontweight="bold", y=1.01
)
fig9.subplots_adjust(left=0.03, right=0.91, top=0.92, bottom=0.05,
                       hspace=0.18, wspace=0.08)
out9 = os.path.join(ROOT, "figures", "Fig9_WeatherRadials_Network.png")
fig9.savefig(out9, dpi=300, bbox_inches="tight")
print(f"Saved: {out9}")
plt.close(fig9)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 9af — Option B: 40 radials per station, one figure per variable (6 total)
# ═════════════════════════════════════════════════════════════════════════════

FIG9A_VARS = [
    dict(col="AirTemp Ave (C)",       label="Air Temperature",   unit="°C",   lc=VAR_COLORS[0],
         fname="AirTemp",   daytime_only=False, excl_ws17=False),
    dict(col="RelHum Ave (%)",        label="Relative Humidity", unit="%",    lc=VAR_COLORS[1],
         fname="RelHum",    daytime_only=False, excl_ws17=False),
    dict(col="WindSpeed Ave (m/s)",   label="Wind Speed",        unit="m/s",  lc=VAR_COLORS[2],
         fname="WindSpeed", daytime_only=False, excl_ws17=False),
    dict(col="AtmPress Ave (hPa)",    label="Atm. Pressure",     unit="hPa",  lc=VAR_COLORS[3],
         fname="AtmPress",  daytime_only=False, excl_ws17=True),
    dict(col="GlobalRad Ave (W/m2)",  label="Solar Radiation",   unit="W/m²", lc=VAR_COLORS[4],
         fname="SolarRad",  daytime_only=True,  excl_ws17=False),
    dict(col="WindDir Ave (degrees)", label="Wind Direction",    unit="°",    lc=VAR_COLORS[5],
         fname="WindDir",   daytime_only=False, excl_ws17=False),
]

N_COLS, N_ROWS = 8, 5
MEAN_GREY = "#aaaaaa"

for vcfg in FIG9A_VARS:
    col          = vcfg["col"]
    label        = vcfg["label"]
    unit         = vcfg["unit"]
    lc           = vcfg["lc"]
    fname        = vcfg["fname"]
    is_wind_dir  = (col == "WindDir Ave (degrees)")
    is_solar     = (col == "GlobalRad Ave (W/m2)")

    print(f"Building Fig 9a-f — {label} (40 stations)...")

    src = all_df.copy()
    if vcfg["excl_ws17"]:
        src.loc[src["station"] == PRESSURE_OUTLIER, col] = np.nan

    # Network-consistent scale (all stations, for direct comparison)
    if not is_wind_dir:
        net_grp = src.groupby(["doy", "hour"])[col].mean().reset_index()
        if is_solar:
            net_grp = net_grp[net_grp["hour"].between(7, 18)]
            vmin_b  = 0.0
        else:
            vmin_b  = float(np.nanpercentile(net_grp[col], 2))
        vmax_b = float(np.nanpercentile(net_grp[col], 98))
        print(f"  Scale: {vmin_b:.1f}–{vmax_b:.1f} {unit}")

    fig9af, axs6b = plt.subplots(N_ROWS, N_COLS, figsize=(27, 17),
                                  subplot_kw={"projection": "polar"})
    fig9af.patch.set_facecolor("white")

    for ax, ws in zip(axs6b.flat, stations):
        sta_df = src[src["station"] == ws]

        if is_wind_dir:
            draw_hourly_wind_rose(ax, sta_df)
            # Overwrite center label with station ID
            tc = np.linspace(0, 2 * np.pi, 360)
            # Estimate r_inner from station data (same formula as in draw_hourly_wind_rose)
            wd_tmp = sta_df[["WindDir Ave (degrees)"]].dropna()
            if len(wd_tmp) > 0:
                ax.text(0, 0, ws, ha="center", va="center",
                        fontsize=6.0, fontfamily=FONT, fontweight="bold",
                        color="#333333", zorder=11)
        else:
            grp_m     = sta_df.groupby(["doy", "hour"])[col].mean().reset_index()
            mean_mask = grp_m["hour"].between(7, 18).values if is_solar else None

            mc_b = MEAN_COLORS_6A.get(col)
            setup_polar(ax)
            draw_radial(
                ax,
                doy        = grp_m["doy"].values,
                hour       = grp_m["hour"].values,
                val        = grp_m[col].values,
                vmin       = vmin_b, vmax=vmax_b, lc=lc,
                std        = None, s=0.5, alpha=0.62,
                show_mean  = True, mean_mask=mean_mask,
                mean_color = mc_b,
            )
            add_month_labels(ax, fontsize=4.8, r_text=1.03)
            # Faulty-sensor label for WS17 pressure
            center_txt = ws if not (vcfg["excl_ws17"] and ws == PRESSURE_OUTLIER) \
                            else f"{ws}\n(faulty)"
            ax.text(0, 0, center_txt, ha="center", va="center",
                    fontsize=5.5, fontfamily=FONT, fontweight="bold",
                    color="#333333" if ws != PRESSURE_OUTLIER else "#E74C3C",
                    multialignment="center", zorder=10)

    # Shared legend
    if is_wind_dir:
        leg_handles = [
            Patch(facecolor=HOUR_CMAP(0.0),      alpha=0.88, label="00h (midnight)"),
            Patch(facecolor=HOUR_CMAP(12 / 23),  alpha=0.88, label="12h (noon)"),
            Patch(facecolor=HOUR_CMAP(18 / 23),  alpha=0.88, label="18h (dusk)"),
        ]
        fig9af.legend(handles=leg_handles,
                     title="Stacked by hour of day", title_fontsize=7.5,
                     loc="lower center", bbox_to_anchor=(0.45, 0.005),
                     fontsize=8, framealpha=0.95, edgecolor="#dddddd",
                     fancybox=False, ncol=3, columnspacing=1.0,
                     prop={"family": FONT})
    else:
        mean_note = "Daily mean (07–18h)" if is_solar else "Daily mean"
        leg_handles = [
            Line2D([0],[0], color=MEAN_COLORS_6A.get(col, MEAN_GREY), lw=1.2, alpha=0.85, label=mean_note),
            Line2D([0],[0], color="none", marker="o", markersize=3,
                   markerfacecolor=HOUR_CMAP(0.5), alpha=0.7, lw=0,
                   label="Dot colour = hour of day"),
        ]
        fig9af.legend(handles=leg_handles, loc="lower center",
                     bbox_to_anchor=(0.45, 0.005),
                     fontsize=8, framealpha=0.95, edgecolor="#dddddd",
                     fancybox=False, ncol=2, columnspacing=1.2,
                     handlelength=1.5, prop={"family": FONT})

    # Shared hour colorbar (applies to all variables)
    sm_b = ScalarMappable(cmap=HOUR_CMAP, norm=Normalize(vmin=0, vmax=23))
    sm_b.set_array([])
    cax_b = fig9af.add_axes([0.93, 0.10, 0.012, 0.78])
    cb_b  = fig9af.colorbar(sm_b, cax=cax_b)
    cb_b.set_label("Hour of Day", fontsize=9, fontfamily=FONT, labelpad=8)
    cb_b.set_ticks([0, 6, 12, 18, 23])
    cb_b.set_ticklabels(["00h", "06h", "12h", "18h", "23h"])
    cb_b.ax.tick_params(labelsize=8)
    for sp in cb_b.ax.spines.values():
        sp.set_linewidth(0.3)

    # Suptitle
    if is_wind_dir:
        subtitle = (f"Annual Wind Direction Roses — NUS Campus (2025)  |  "
                    f"One panel per weather station\n"
                    f"Bar height = directional frequency  ·  Bar colour = hour of day  ·  "
                    f"Jan reference ticks around each rose")
    else:
        scale_str = f"Consistent scale: {vmin_b:.1f}–{vmax_b:.1f} {unit}"
        sol_note  = "  ·  Mean line = daytime hours (07–18h)" if is_solar else ""
        ws17_note = "  ·  WS17 excluded (faulty sensor)" if vcfg["excl_ws17"] else ""
        subtitle  = (f"Annual {label} Radials — NUS Campus (2025)  |  "
                     f"One panel per weather station\n"
                     f"Jan at top, clockwise  ·  Dot colour = hour of day  ·  "
                     f"Radius = {label.lower()}  ·  {scale_str}{sol_note}{ws17_note}")

    fig9af.suptitle(subtitle, fontsize=11, fontfamily=FONT, fontweight="bold", y=1.003)
    fig9af.subplots_adjust(left=0.02, right=0.91, top=0.97, bottom=0.03,
                           hspace=0.06, wspace=0.06)

    FIG8_LETTERS = {"AirTemp": "a", "WindDir": "b", "RelHum": "c",
                    "WindSpeed": "d", "AtmPress": "e", "SolarRad": "f"}
    out9af = os.path.join(ROOT, "figures", "additional_figures",
                         f"AddFig9{FIG8_LETTERS[fname]}_{fname}_AllStations.png")
    fig9af.savefig(out9af, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out9af}")
    plt.close(fig9af)

print("\nAll radial figures complete.")
