import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter, StrMethodFormatter
from matplotlib.colors import Normalize, ListedColormap
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import contextily as ctx
import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import glob, os, calendar
import windrose

# ── Style ─────────────────────────────────────────────────────────────────────
from matplotlib import font_manager as _fm


def _pick_font(cands=("Palatino Linotype", "Palatino", "TeX Gyre Pagella",
                      "URW Palladio L", "P052")):
    """Palatino on Windows/macOS; metric-compatible clone elsewhere."""
    have = {f.name for f in _fm.fontManager.ttflist}
    return next((c for c in cands if c in have), "serif")


FONT = _pick_font()
plt.rcParams.update({
    "font.family":        FONT,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "figure.dpi":         150,
    "savefig.facecolor":  "white",
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

VIRIDIS   = cm.viridis
TWILIGHT  = cm.twilight_shifted
MONTH_LABELS = [calendar.month_abbr[m] for m in range(1, 13)]

VAR_COLORS = [VIRIDIS(x) for x in np.linspace(0.08, 0.92, 6)]

MAPBOX_APIKEY = None

# ── Variable definitions ──────────────────────────────────────────────────────
# (col, short_label, unit, line_color, fig6/fig10_cmap)
VARS = [
    ("AirTemp Ave (C)",       "Air Temperature",    "°C",    VAR_COLORS[0], cm.YlOrRd),
    ("RelHum Ave (%)",        "Relative Humidity",  "%",     VAR_COLORS[1], cm.YlGnBu),
    ("WindSpeed Ave (m/s)",   "Wind Speed",         "m/s",   VAR_COLORS[2], cm.BuGn),
    ("AtmPress Ave (hPa)",    "Atm. Pressure",      "hPa",   VAR_COLORS[3], cm.PuBu),
    ("GlobalRad Ave (W/m2)",  "Solar Radiation",    "W/m²",  VAR_COLORS[4], cm.afmhot),  # dark→yellow→white
    ("WindDir Ave (degrees)", "Wind Direction",     "°",     VAR_COLORS[5], TWILIGHT),
]

# Fig 3 and Fig 5 drop Wind Direction
VARS_FIG6 = [v for v in VARS if v[0] != "WindDir Ave (degrees)"]
VARS_FIG10 = VARS_FIG6

# Faulty WS17 pressure (Jan–Apr) is removed by 0_quality_control.py, so no
# station is excluded any more; set a station ID here to exclude one.
PRESSURE_OUTLIER = ""
EXPECTED_HOURS   = 8760

# ── Load all data ─────────────────────────────────────────────────────────────
FILES = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "NUS_CAMPUS_WS*.csv")))
station_meta = {}
dfs = []
print("Loading data...")
for fpath in FILES:
    ws = os.path.basename(fpath).split("_")[2]
    df = pd.read_csv(fpath, parse_dates=["Datetime"])
    df["station"] = ws
    df["hour"]    = df["Datetime"].dt.hour
    df["month"]   = df["Datetime"].dt.month
    station_meta[ws] = {"lat": df["Latitude"].iloc[0],
                         "lon": df["Longitude"].iloc[0]}
    dfs.append(df)

all_df   = pd.concat(dfs, ignore_index=True)
stations = sorted(station_meta.keys())
print(f"Loaded {len(stations)} stations, {len(all_df):,} rows.")

# Diagnostics
rad_max  = all_df["GlobalRad Ave (W/m2)"].max()
rad_mean = all_df["GlobalRad Ave (W/m2)"].mean()
print(f"Solar radiation — instantaneous max: {rad_max:.1f} W/m²  |  overall mean: {rad_mean:.1f} W/m²")

# ── Helpers ───────────────────────────────────────────────────────────────────
def circ_mean(series):
    rad = np.deg2rad(series.dropna())
    if len(rad) == 0:
        return np.nan
    return np.degrees(np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())) % 360

def style_cb(cb, unit, fontsize=8):
    cb.set_label(unit, fontsize=fontsize, fontfamily=FONT, labelpad=5)
    cb.ax.tick_params(labelsize=fontsize - 1)
    for sp in cb.ax.spines.values():
        sp.set_linewidth(0.3)

def style_ax(ax, xlabel="", ylabel="", title="", fontsize=9):
    if title:
        ax.set_title(title, fontsize=fontsize+1, fontfamily=FONT,
                     fontweight="bold", pad=7)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=fontsize, fontfamily=FONT, labelpad=4)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=fontsize, fontfamily=FONT, labelpad=4)
    ax.tick_params(labelsize=fontsize - 1)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)

def draw_station_windrose(ax, lon, lat, ws_series, wd_series,
                          radius=0.00085, n_sectors=16, cmap=cm.plasma, vmax=4.0):
    """Geographic wind rose at (lon, lat). Bar length = relative frequency; colour = mean speed."""
    common = ws_series.dropna().index.intersection(wd_series.dropna().index)
    ws = ws_series[common]
    wd = wd_series[common]
    if len(ws) < 20:
        return

    sector = 360 / n_sectors
    freqs, speeds = [], []
    for s in range(n_sectors):
        lo, hi = s * sector, (s + 1) * sector
        mask = (wd >= lo) & (wd < hi)
        freqs.append(mask.sum() / len(ws))
        speeds.append(ws[mask].mean() if mask.sum() > 0 else 0.0)

    max_freq = max(freqs) if max(freqs) > 0 else 1.0

    for s, (freq, spd) in enumerate(zip(freqs, speeds)):
        if freq < 0.005:
            continue
        lo, hi = s * sector, (s + 1) * sector
        # Meteorological → math angle (0=East, CCW)
        theta_lo = np.radians(90 - hi)
        theta_hi = np.radians(90 - lo)
        r = (freq / max_freq) * radius

        thetas = np.linspace(theta_lo, theta_hi, 8)
        xs = [lon] + list(lon + r * np.cos(thetas)) + [lon]
        ys = [lat] + list(lat + r * np.sin(thetas)) + [lat]
        color = cmap(min(spd / vmax, 1.0))
        ax.fill(xs, ys, color=color, alpha=0.90, zorder=4, lw=0)
        ax.plot(xs, ys, color="white", lw=0.2, zorder=5)

    ax.scatter([lon], [lat], s=5, c="white", zorder=6, linewidths=0)

def getWSid(num):
    if num >= 10:
        return "WS"+str(num)
    else:
        return "WS0"+str(num)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 6 — Diurnal × Seasonal Climatology  (Ladybug-style horizontal strips)
# ═════════════════════════════════════════════════════════════════════════════
print("Building Fig 6...")

clean_df = all_df.copy()
clean_df.loc[clean_df["station"] == PRESSURE_OUTLIER, "AtmPress Ave (hPa)"] = np.nan

fig6, axes3 = plt.subplots(5, 2, figsize=(15, 12), sharey=True)
fig6.patch.set_facecolor("white")

for axs, (col, label, unit, lc, cmap) in zip(axes3, VARS_FIG6):
    # print(ax, col)
    ax = axs[0]
    ax2 = axs[1]
    climate = clean_df.groupby(["month", "hour"])[col].mean().unstack(level="hour")
    climate_ground = clean_df[clean_df["Type"]=="Ground"].groupby(["month", "hour"])[col].mean().unstack(level="hour")
    climate_roof = clean_df[clean_df["Type"]!="Ground"].groupby(["month", "hour"])[col].mean().unstack(level="hour")
    use_cmap = cmap
    vmin, vmax_v = climate.values.min(), climate.values.max()

    # Solar radiation: force vmin=0 so night (0 W/m²) anchors to black on afmhot
    if col == "GlobalRad Ave (W/m2)":
        vmin = 0

    im = ax.imshow(climate_ground.values, aspect="auto", cmap=use_cmap,
                   vmin=vmin, vmax=vmax_v)
    im2 = ax2.imshow(climate_roof.values, aspect="auto", cmap=use_cmap,
                   vmin=vmin, vmax=vmax_v)

    # Thin white cell grid (Ladybug style)
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, linestyle="-")
    ax.tick_params(which="minor", length=0)
    ax2.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax2.set_yticks(np.arange(-0.5, 12, 1), minor=True)
    ax2.grid(which="minor", color="white", linewidth=0.4, linestyle="-")
    ax2.tick_params(which="minor", length=0)

    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 3)],
                       fontsize=8, fontfamily=FONT)
    ax2.set_xticks(range(0, 24, 3))
    ax2.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 3)],
                       fontsize=8, fontfamily=FONT)
    
    ax.set_yticks(range(12))
    ax.set_yticklabels(MONTH_LABELS, fontsize=8, fontfamily=FONT)
    ax2.yaxis.set_ticks_position('none') 

    cb = fig6.colorbar(im, ax=ax2, fraction=0.018, pad=0.01)
    style_cb(cb, unit, fontsize=10)

    ax.set_title(label+ " - Ground", fontsize=10, fontfamily=FONT, fontweight="bold",
                 loc="left", pad=5)
    ax2.set_title(label+ " - Rooftop", fontsize=10, fontfamily=FONT, fontweight="bold",
                 loc="left", pad=5)
    ax.set_ylabel("Month", fontsize=10, fontfamily=FONT)
    ax.tick_params(labelsize=8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)

# axes3[-1].set_xlabel("Hour of Day", fontsize=9, fontfamily=FONT, labelpad=6)
fig6.supxlabel("Hour of Day", fontsize=10, fontfamily=FONT)

fig6.suptitle(
    "Diurnal and Seasonal Climatology — NUS Campus Meteorological Network (2025)\n"
    "Mean across 40 stations",
    fontsize=12, fontfamily=FONT, fontweight="bold", y=1.005
)
fig6.tight_layout(h_pad=1.2)
out6 = os.path.join(ROOT, "figures", "Fig6_Climatology_DiurnalSeasonal.png")
fig6.savefig(out6, dpi=300, bbox_inches="tight")
print(f"Saved: {out6}")
plt.close(fig6)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 7 — Network Envelope Time Series  (monthly, with median)
# ═════════════════════════════════════════════════════════════════════════════
print("Building Fig 7...")

fig7, axes4 = plt.subplots(5, 2, figsize=(15, 12), sharex=True)
fig7.patch.set_facecolor("white")
xticks = list(range(12))

for idx, (axs, (col, label, unit, lc, _)) in enumerate(zip(axes4, VARS_FIG6)):
    ax = axs[0]
    ax2 = axs[1]
        
    ground_src = all_df.copy()
    ground_src = ground_src[ground_src["Type"]=="Ground"]
    roof_src = all_df.copy()
    roof_src = roof_src[roof_src["Type"]!="Ground"]
        
    if col == "AtmPress Ave (hPa)":
        ground_src.loc[ground_src["station"] == PRESSURE_OUTLIER, col] = np.nan
        roof_src.loc[roof_src["station"] == PRESSURE_OUTLIER, col] = np.nan

    if col == "WindDir Ave (degrees)":
        ground_monthly = (ground_src.groupby(["station", "month"])[col]
                   .apply(circ_mean).unstack(level="month"))
        roof_monthly = (roof_src.groupby(["station", "month"])[col]
                   .apply(circ_mean).unstack(level="month"))
    else:
        ground_monthly = ground_src.groupby(["station", "month"])[col].mean().unstack(level="month")
        roof_monthly = roof_src.groupby(["station", "month"])[col].mean().unstack(level="month")

    ground_stations = ground_monthly.index.to_list()
    roof_stations = roof_monthly.index.to_list()

    for ws in ground_stations:
        if ws not in ground_monthly.index:
            continue
        is_outlier = (col == "AtmPress Ave (hPa)" and ws == PRESSURE_OUTLIER)
        ax.plot(xticks, ground_monthly.loc[ws].values,
                color="#dddddd" if not is_outlier else "#E74C3C",
                lw=0.7 if not is_outlier else 1.2,
                alpha=0.6 if not is_outlier else 0.9,
                linestyle="-" if not is_outlier else "--",
                zorder=2 if not is_outlier else 5,
                label=f"{PRESSURE_OUTLIER} (faulty sensor)" if is_outlier else None)
    for ws in roof_stations:
        if ws not in roof_monthly.index:
            continue
        is_outlier = (col == "AtmPress Ave (hPa)" and ws == PRESSURE_OUTLIER)
        ax2.plot(xticks, roof_monthly.loc[ws].values,
                color="#dddddd" if not is_outlier else "#E74C3C",
                lw=0.7 if not is_outlier else 1.2,
                alpha=0.6 if not is_outlier else 0.9,
                linestyle="-" if not is_outlier else "--",
                zorder=2 if not is_outlier else 5,
                label=f"{PRESSURE_OUTLIER} (faulty sensor)" if is_outlier else None)

    ground_net        = ground_monthly.drop(index=PRESSURE_OUTLIER, errors="ignore")
    ground_net_mean   = ground_net.mean()
    ground_net_median = ground_net.median()
    ground_net_std    = ground_net.std()

    roof_net        = roof_monthly.drop(index=PRESSURE_OUTLIER, errors="ignore")
    roof_net_mean   = roof_net.mean()
    roof_net_median = roof_net.median()
    roof_net_std    = roof_net.std()

    min_net = min([ground_net.min().min(), roof_net.min().min()])
    max_net = max([ground_net.max().max(), roof_net.max().max()])

    ax.fill_between(xticks, ground_net_mean - ground_net_std, ground_net_mean + ground_net_std,
                    color=lc, alpha=0.20, zorder=3)
    ax.plot(xticks, ground_net_mean + ground_net_std, color=lc, lw=0.7,
            linestyle="--", alpha=0.55, zorder=3)
    ax.plot(xticks, ground_net_mean - ground_net_std, color=lc, lw=0.7,
            linestyle="--", alpha=0.55, zorder=3)
    ax.plot(xticks, ground_net_mean,   color=lc, lw=2.4, zorder=4)
    ax.plot(xticks, ground_net_median, color=lc, lw=1.4, linestyle=":",
            alpha=0.85, zorder=4)
    ax2.fill_between(xticks, roof_net_mean - roof_net_std, roof_net_mean + roof_net_std,
                    color=lc, alpha=0.20, zorder=3)
    ax2.plot(xticks, roof_net_mean + roof_net_std, color=lc, lw=0.7,
            linestyle="--", alpha=0.55, zorder=3)
    ax2.plot(xticks, roof_net_mean - roof_net_std, color=lc, lw=0.7,
            linestyle="--", alpha=0.55, zorder=3)
    ax2.plot(xticks, roof_net_mean,   color=lc, lw=2.4, zorder=4)
    ax2.plot(xticks, roof_net_median, color=lc, lw=1.4, linestyle=":",
            alpha=0.85, zorder=4)
    ax.set_ylim([min_net, max_net])
    ax2.set_ylim([min_net, max_net])
    ax.set_ylabel(f"{label} ({unit})", fontsize=8.5, fontfamily=FONT, labelpad=5)
    ax.tick_params(labelsize=8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)
    for sp in ax2.spines.values():
        sp.set_linewidth(0.5)

    ax2.yaxis.set_ticks_position('none')
    ax2.set_yticklabels([])

    if col == "AtmPress Ave (hPa)" and PRESSURE_OUTLIER:
        ax.annotate(f"{PRESSURE_OUTLIER}: faulty pressure sensor",
                    xy=(0.02, 0.08), xycoords="axes fraction",
                    fontsize=10, fontfamily=FONT, color="#E74C3C",
                    fontstyle="italic")

legend_elements = [
    Line2D([0],[0], color="#cccccc", lw=1.2, label="Individual stations (n=40)"),
    Line2D([0],[0], color=VAR_COLORS[0], lw=2.4, label="Network mean"),
    Line2D([0],[0], color=VAR_COLORS[0], lw=1.4, linestyle=":",
           alpha=0.85, label="Network median"),
    Patch(facecolor=VAR_COLORS[0], alpha=0.25, label="±1 std. dev."),
]
fig7.legend(handles=legend_elements, fontsize=8, loc="upper right",
                framealpha=0.88, edgecolor="#dddddd", fancybox=False,
                prop={"family": FONT})

for ax in axes4[-1]:
    ax.set_xticks(xticks)
    ax.set_xticklabels(MONTH_LABELS, fontsize=9, fontfamily=FONT)
fig7.supxlabel("Month (2025)", fontsize=10, fontfamily=FONT)

axes4[0,0].set_title("Ground Stations", fontsize=10, fontfamily=FONT, fontweight="bold", pad=15)
axes4[0,1].set_title("Rooftop Stations", fontsize=10, fontfamily=FONT, fontweight="bold", pad=15)

fig7.suptitle(
    "Monthly Network Time Series — NUS Campus Meteorological Observation Network (2025)\n"
    "All 40 stations (grey)  |  Network mean (solid) and median (dotted)  |  Shaded band = ±1 std. dev.",
    fontsize=11.5, fontfamily=FONT, fontweight="bold", y=1.005
)
fig7.tight_layout(h_pad=0.6)
out7 = os.path.join(ROOT, "figures", "Fig7_NetworkTimeSeries.png")
fig7.savefig(out7, dpi=300, bbox_inches="tight")
print(f"Saved: {out7}")
plt.close(fig7)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 8 — Windrose
# ═════════════════════════════════════════════════════════════════════════════
print("Building Fig 8...")
fig8 = plt.figure(figsize=(32, 12))
minlon, maxlon, minlat, maxlat = (103.7685, 103.784, 1.2885, 1.31)
proj = ccrs.PlateCarree()

main_ax_ground = fig8.add_subplot(1, 2, 1, projection=proj)
main_ax_ground.set_extent([minlon, maxlon, minlat, maxlat], crs=proj)
plt.setp(main_ax_ground.spines.values(), linewidth=0.5)
main_ax_ground.set_title("Ground Stations", loc="left", fontsize=16)

main_ax_roof = fig8.add_subplot(1, 1, 1, projection=proj)
main_ax_roof.set_extent([minlon, maxlon, minlat, maxlat], crs=proj)
plt.setp(main_ax_roof.spines.values(), linewidth=0.5)
main_ax_roof.set_title("Rooftop Stations", loc="left", fontsize=16)

try: 
    request = cimgt.MapboxTiles(MAPBOX_APIKEY,"light-v11")
    main_ax_ground.add_image(request, 16)
    main_ax_roof.add_image(request, 16)
except Exception as e:
    print(f'  basemap skipped: {e}')

bins = np.arange(0, 4, 1)
colors = [ (115/255,238/255,220/255,1), (115/255,194/255,190/255,1), (119/255,104/255,133/255,1), (95/255,26/255,55/255,1) ]
cmap = ListedColormap(colors)

excl_stations = [3, 5, 21, 26, 29, 31, 38]
wslist = [i for i in range(1,41) if i not in excl_stations]
for i in wslist:
    wsid = getWSid(i)
    wsdf = all_df.copy()
    wsdf = wsdf[wsdf["station"]==wsid]
    wstype = wsdf['Type'].to_list()[0]
    ws_lon = wsdf['Longitude'].to_list()[0]
    ws_lat = wsdf['Latitude'].to_list()[0]

    ax_curr = {}
    if wstype == "Ground":
        ax_curr = main_ax_ground
    elif wstype.startswith("Roof"):
        ax_curr = main_ax_roof

    df.dropna(subset=["WindSpeed Ave (m/s)", "WindDir Ave (degrees)"])
    windspeed = wsdf["WindSpeed Ave (m/s)"].values
    winddir = wsdf["WindDir Ave (degrees)"].values

    wrax_curr = inset_axes(
        ax_curr,
        width=0.8,
        height=0.8,
        loc="center",
        bbox_to_anchor=(ws_lon, ws_lat),
        bbox_transform=ax_curr.transData,
        axes_class=windrose.WindroseAxes,
    )
    wrax_curr.bar(winddir, windspeed, normed=True, nsector=8, opening=1, cmap=cmap, bins=bins)
    wrax_curr.tick_params(labelleft=False, labelbottom=False)
    wrax_curr.set_yticks(np.arange(10, 60, step=10))
    wrax_curr.set_yticklabels(np.arange(10, 60, step=10))
    wrax_curr.yaxis.grid(alpha=0.2)
    wrax_curr.xaxis.grid(alpha=0.4)
    wrax_curr.set_facecolor((1, 1, 1, 0))
    plt.setp(wrax_curr.spines.values(), linewidth=0.2)

## LEGEND
wrax_legend = inset_axes(
    main_ax_roof,
    width=2,  # size in inches
    height=2,  # size in inches
    bbox_to_anchor=(-0.04,-0.04,1,1),
    axes_class=windrose.WindroseAxes,  # specify the class of the axe
    bbox_transform=main_ax_roof.transAxes
)
sample_wd = [120,120,120,120,120,120,120,120,120,120,120,120, 270,270,270,270,270,270,270,270]
sample_ws = [0,0,1,1,1,1,1,2,2,2,2,3, 0,0,1,1,1,2,2,3]
wrax_legend.bar(sample_wd, sample_ws, normed=True, nsector=8, opening=1, cmap=cmap, bins=bins)
wrax_legend.set_yticks(np.arange(10, 60, step=10))
wrax_legend.set_yticklabels(np.arange(10, 60, step=10), fontsize="8")
wrax_legend.yaxis.set_major_formatter(StrMethodFormatter("{x}%"))
wrax_legend.yaxis.grid(alpha=0.2)
wrax_legend.xaxis.grid(alpha=0.4)
wrax_legend.set_legend(title="Wind Speed (m/s)", bbox_to_anchor=(0.4, -0.7))

out8 = os.path.join(ROOT, "figures", "Fig8_WindRose_Map.png")
fig8.savefig(out8, bbox_inches = "tight")
print(f"Saved: {out8}")
plt.close(fig8)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 11 — Spatial Variability Maps  (5 panels: Wind Dir removed — encoded in rose)
#   Color = annual mean value  (unique colormap per variable)
#   Size  = data completeness for that variable at each station
#   Wind Speed panel = geographic mini wind roses (speed + direction)
# ═════════════════════════════════════════════════════════════════════════════
print("Building Fig 10...")

annual       = {}
completeness = {}
for col, *_ in VARS:
    src = all_df.copy()
    if col == "AtmPress Ave (hPa)":
        src.loc[src["station"] == PRESSURE_OUTLIER, col] = np.nan
    if col == "WindDir Ave (degrees)":
        annual[col] = src.groupby("station")[col].apply(circ_mean)
    else:
        annual[col] = src.groupby("station")[col].mean()
    completeness[col] = src.groupby("station")[col].apply(
        lambda s: s.notna().sum() / EXPECTED_HOURS
    )

lons = [station_meta[ws]["lon"] for ws in stations]
lats = [station_meta[ws]["lat"] for ws in stations]
pad  = 0.004
xlim = (min(lons) - pad, max(lons) + pad)
ylim = (min(lats) - pad, max(lats) + pad)

# 3-top + 2-bottom-centred layout via GridSpec
fig10 = plt.figure(figsize=(18, 12))
fig10.patch.set_facecolor("white")
gs = GridSpec(2, 6, figure=fig10, hspace=0.38, wspace=0.30)
axes5 = [
    fig10.add_subplot(gs[0, 0:2]),   # Air Temperature
    fig10.add_subplot(gs[0, 2:4]),   # Relative Humidity
    fig10.add_subplot(gs[0, 4:6]),   # Wind Speed & Direction (rose)
    fig10.add_subplot(gs[1, 1:3]),   # Atm. Pressure  (centred)
    fig10.add_subplot(gs[1, 3:5]),   # Solar Radiation (centred)
]

WINDROSE_VMAX = 4.0   # m/s colour-scale ceiling

for ax, (col, label, unit, lc, cmap5) in zip(axes5, VARS_FIG10):

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    try:  # basemap tiles need internet access; figure still renders without
        ctx.add_basemap(ax, crs="EPSG:4326",
                    source=ctx.providers.CartoDB.Positron,
                    alpha=0.40, zorder=0, attribution=False)
    except Exception as e:
        print(f'  basemap skipped: {e}')
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # ── Wind Speed panel: geographic wind roses ───────────────────────────────
    if col == "WindSpeed Ave (m/s)":
        for ws in stations:
            sta = all_df[all_df["station"] == ws]
            draw_station_windrose(
                ax,
                station_meta[ws]["lon"], station_meta[ws]["lat"],
                sta["WindSpeed Ave (m/s)"], sta["WindDir Ave (degrees)"],
                radius=0.00085, n_sectors=16,
                cmap=cmap5, vmax=WINDROSE_VMAX
            )

        sm = ScalarMappable(cmap=cmap5, norm=Normalize(vmin=0, vmax=WINDROSE_VMAX))
        sm.set_array([])
        cb = fig10.colorbar(sm, ax=ax, fraction=0.033, pad=0.02, aspect=22)
        style_cb(cb, unit, fontsize=8)

        style_ax(ax, xlabel="Longitude (°E)", ylabel="Latitude (°N)",
                 title="Wind Speed & Direction", fontsize=8)
        ax.annotate("Bar length = relative frequency  |  Colour = mean speed",
                    xy=(0.5, 0.015), xycoords="axes fraction", ha="center",
                    fontsize=6, fontfamily=FONT, fontstyle="italic", color="#555555")
        continue

    # ── All other variables: scatter map ─────────────────────────────────────
    vals  = [annual[col].get(ws, np.nan) for ws in stations]
    compl = [completeness[col].get(ws, 0.0) for ws in stations]
    sizes_v = [max(c**1.5 * 370 + 50, 60) for c in compl]

    normal_idx = [i for i, ws in enumerate(stations)
                  if not (col == "AtmPress Ave (hPa)" and ws == PRESSURE_OUTLIER)]
    faulty_idx = [i for i, ws in enumerate(stations)
                  if col == "AtmPress Ave (hPa)" and ws == PRESSURE_OUTLIER]

    sc = ax.scatter(
        [lons[i] for i in normal_idx], [lats[i] for i in normal_idx],
        c=[vals[i] for i in normal_idx], cmap=cmap5,
        s=[sizes_v[i] for i in normal_idx],
        edgecolors="white", linewidths=0.7, zorder=3, alpha=0.90
    )

    for i in faulty_idx:
        ax.scatter(lons[i], lats[i], c="red", marker="x",
                   s=120, linewidths=2, zorder=6)
        ax.annotate(f"{PRESSURE_OUTLIER}*", xy=(lons[i], lats[i]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6, color="red", fontfamily=FONT)

    cb = fig10.colorbar(sc, ax=ax, fraction=0.033, pad=0.02, aspect=22)
    style_cb(cb, unit, fontsize=8)
    style_ax(ax, xlabel="Longitude (°E)", ylabel="Latitude (°N)",
             title=label, fontsize=8)

    for pct in [0.6, 0.8, 1.0]:
        ax.scatter([], [], s=max(pct**1.5 * 370 + 50, 60),
                   c="#888888", alpha=0.7, edgecolors="white",
                   linewidths=0.5, label=f"{int(pct*100)}%")
    ax.legend(title="Data\ncompleteness", title_fontsize=6,
              fontsize=6.5, loc="lower right", framealpha=0.88,
              edgecolor="#dddddd", fancybox=False,
              labelspacing=0.8, handletextpad=0.7, borderpad=0.8,
              prop={"family": FONT})

fig10.text(0.01, 0.003,
          (f"* {PRESSURE_OUTLIER} excluded from Atm. Pressure panel (faulty sensor).  " if PRESSURE_OUTLIER else "")
          + "Circle size = per-variable data completeness.  Wind direction encoded in rose bar length.",
          fontsize=7.5, fontfamily=FONT, fontstyle="italic", color="#555555")

fig10.suptitle(
    "Spatial Variability of Annual Mean Meteorological Variables\n"
    "NUS Campus Observation Network (2025)  |  40 stations  |  Circle size = data completeness",
    fontsize=13, fontfamily=FONT, fontweight="bold", y=1.01
)
out10 = os.path.join(ROOT, "figures", "additional_figures", "AddFig11_SpatialVariability.png")
fig10.savefig(out10, dpi=300, bbox_inches="tight")
print(f"Saved: {out10}")
plt.close(fig10)

print("\nAll figures complete.")
