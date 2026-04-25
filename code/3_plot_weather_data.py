import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import seaborn as sns
import contextily as ctx
import glob, os, calendar

# ── Style ─────────────────────────────────────────────────────────────────────
FONT = "Palatino Linotype"
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

# ── Variable definitions ──────────────────────────────────────────────────────
# (col, short_label, unit, line_color, fig3/fig5_cmap)
VARS = [
    ("AirTemp Ave (C)",       "Air Temperature",    "°C",    VAR_COLORS[0], cm.YlOrRd),
    ("RelHum Ave (%)",        "Relative Humidity",  "%",     VAR_COLORS[1], cm.YlGnBu),
    ("WindSpeed Ave (m/s)",   "Wind Speed",         "m/s",   VAR_COLORS[2], cm.BuGn),
    ("AtmPress Ave (hPa)",    "Atm. Pressure",      "hPa",   VAR_COLORS[3], cm.PuBu),
    ("GlobalRad Ave (W/m2)",  "Solar Radiation",    "W/m²",  VAR_COLORS[4], cm.afmhot),  # dark→yellow→white
    ("WindDir Ave (degrees)", "Wind Direction",     "°",     VAR_COLORS[5], TWILIGHT),
]

# Fig 3 and Fig 5 drop Wind Direction
VARS_FIG3 = [v for v in VARS if v[0] != "WindDir Ave (degrees)"]
VARS_FIG5 = VARS_FIG3

PRESSURE_OUTLIER = "WS17"
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

# ═════════════════════════════════════════════════════════════════════════════
# Fig 3 — Diurnal × Seasonal Climatology  (Ladybug-style horizontal strips)
# ═════════════════════════════════════════════════════════════════════════════
print("Building Fig 3...")

clean_df = all_df.copy()
clean_df.loc[clean_df["station"] == PRESSURE_OUTLIER, "AtmPress Ave (hPa)"] = np.nan

fig3, axes3 = plt.subplots(5, 1, figsize=(18, 14))
fig3.patch.set_facecolor("white")

for ax, (col, label, unit, lc, cmap) in zip(axes3, VARS_FIG3):
    climate = clean_df.groupby(["month", "hour"])[col].mean().unstack(level="hour")
    use_cmap = cmap
    vmin, vmax_v = climate.values.min(), climate.values.max()

    # Solar radiation: force vmin=0 so night (0 W/m²) anchors to black on afmhot
    if col == "GlobalRad Ave (W/m2)":
        vmin = 0

    im = ax.imshow(climate.values, aspect="auto", cmap=use_cmap,
                   vmin=vmin, vmax=vmax_v, interpolation="bilinear", origin="upper")

    # Thin white cell grid (Ladybug style)
    ax.set_xticks(np.arange(-0.5, 24, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 12, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, linestyle="-")
    ax.tick_params(which="minor", length=0)

    ax.set_xticks(range(0, 24, 3))
    ax.set_xticklabels([f"{h:02d}h" for h in range(0, 24, 3)],
                       fontsize=7.5, fontfamily=FONT)
    ax.set_yticks(range(12))
    ax.set_yticklabels(MONTH_LABELS, fontsize=8, fontfamily=FONT)

    cb = fig3.colorbar(im, ax=ax, fraction=0.018, pad=0.01, aspect=14)
    style_cb(cb, unit, fontsize=7.5)

    ax.set_title(label, fontsize=9.5, fontfamily=FONT, fontweight="bold",
                 loc="left", pad=5)
    ax.set_ylabel("Month", fontsize=8, fontfamily=FONT, labelpad=4)
    ax.tick_params(labelsize=7.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.4)

axes3[-1].set_xlabel("Hour of Day", fontsize=9, fontfamily=FONT, labelpad=6)

fig3.suptitle(
    "Diurnal and Seasonal Climatology — NUS Campus Meteorological Network (2025)\n"
    "Mean across 40 stations",
    fontsize=12, fontfamily=FONT, fontweight="bold", y=1.005
)
fig3.tight_layout(h_pad=1.2)
out3 = os.path.join(ROOT, "figures", "Fig6b_Climatology_DiurnalSeasonal.png")
fig3.savefig(out3, dpi=300, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close(fig3)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 4 — Network Envelope Time Series  (monthly, with median)
# ═════════════════════════════════════════════════════════════════════════════
print("Building Fig 4...")

fig4, axes4 = plt.subplots(6, 1, figsize=(14, 18), sharex=True)
fig4.patch.set_facecolor("white")
xticks = list(range(12))

for idx, (ax, (col, label, unit, lc, _)) in enumerate(zip(axes4, VARS)):

    src = all_df.copy()
    if col == "AtmPress Ave (hPa)":
        src.loc[src["station"] == PRESSURE_OUTLIER, col] = np.nan

    if col == "WindDir Ave (degrees)":
        monthly = (src.groupby(["station", "month"])[col]
                   .apply(circ_mean).unstack(level="month"))
    else:
        monthly = src.groupby(["station", "month"])[col].mean().unstack(level="month")

    for ws in stations:
        if ws not in monthly.index:
            continue
        is_outlier = (col == "AtmPress Ave (hPa)" and ws == PRESSURE_OUTLIER)
        ax.plot(xticks, monthly.loc[ws].values,
                color="#dddddd" if not is_outlier else "#E74C3C",
                lw=0.7 if not is_outlier else 1.2,
                alpha=0.6 if not is_outlier else 0.9,
                linestyle="-" if not is_outlier else "--",
                zorder=2 if not is_outlier else 5,
                label=f"{PRESSURE_OUTLIER} (faulty sensor)" if is_outlier else None)

    net        = monthly.drop(index=PRESSURE_OUTLIER, errors="ignore")
    net_mean   = net.mean()
    net_median = net.median()
    net_std    = net.std()

    ax.fill_between(xticks, net_mean - net_std, net_mean + net_std,
                    color=lc, alpha=0.20, zorder=3)
    ax.plot(xticks, net_mean + net_std, color=lc, lw=0.7,
            linestyle="--", alpha=0.55, zorder=3)
    ax.plot(xticks, net_mean - net_std, color=lc, lw=0.7,
            linestyle="--", alpha=0.55, zorder=3)
    ax.plot(xticks, net_mean,   color=lc, lw=2.4, zorder=4)
    ax.plot(xticks, net_median, color=lc, lw=1.4, linestyle=":",
            alpha=0.85, zorder=4)

    ax.set_ylabel(f"{label} ({unit})", fontsize=8.5, fontfamily=FONT, labelpad=5)
    ax.tick_params(labelsize=8)
    for sp in ax.spines.values():
        sp.set_linewidth(0.5)

    if col == "AtmPress Ave (hPa)":
        ax.annotate(f"{PRESSURE_OUTLIER}: faulty pressure sensor",
                    xy=(0.01, 0.08), xycoords="axes fraction",
                    fontsize=7.5, fontfamily=FONT, color="#E74C3C",
                    fontstyle="italic")

legend_elements = [
    Line2D([0],[0], color="#cccccc", lw=1.2, label="Individual stations (n=40)"),
    Line2D([0],[0], color=VAR_COLORS[0], lw=2.4, label="Network mean"),
    Line2D([0],[0], color=VAR_COLORS[0], lw=1.4, linestyle=":",
           alpha=0.85, label="Network median"),
    Patch(facecolor=VAR_COLORS[0], alpha=0.25, label="±1 std. dev."),
]
axes4[0].legend(handles=legend_elements, fontsize=8, loc="upper right",
                framealpha=0.88, edgecolor="#dddddd", fancybox=False,
                prop={"family": FONT})

axes4[-1].set_xticks(xticks)
axes4[-1].set_xticklabels(MONTH_LABELS, fontsize=9, fontfamily=FONT)
axes4[-1].set_xlabel("Month (2025)", fontsize=10, fontfamily=FONT, labelpad=8)

fig4.suptitle(
    "Monthly Network Time Series — NUS Campus Meteorological Observation Network (2025)\n"
    "All 40 stations (grey)  |  Network mean (solid) and median (dotted)  |  Shaded band = ±1 std. dev.",
    fontsize=11.5, fontfamily=FONT, fontweight="bold", y=1.005
)
fig4.tight_layout(h_pad=0.6)
out4 = os.path.join(ROOT, "figures", "Fig7_NetworkTimeSeries.png")
fig4.savefig(out4, dpi=300, bbox_inches="tight")
print(f"Saved: {out4}")
plt.close(fig4)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 5 — Spatial Variability Maps  (5 panels: Wind Dir removed — encoded in rose)
#   Color = annual mean value  (unique colormap per variable)
#   Size  = data completeness for that variable at each station
#   Wind Speed panel = geographic mini wind roses (speed + direction)
# ═════════════════════════════════════════════════════════════════════════════
print("Building Fig 5...")

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
fig5 = plt.figure(figsize=(18, 12))
fig5.patch.set_facecolor("white")
gs = GridSpec(2, 6, figure=fig5, hspace=0.38, wspace=0.30)
axes5 = [
    fig5.add_subplot(gs[0, 0:2]),   # Air Temperature
    fig5.add_subplot(gs[0, 2:4]),   # Relative Humidity
    fig5.add_subplot(gs[0, 4:6]),   # Wind Speed & Direction (rose)
    fig5.add_subplot(gs[1, 1:3]),   # Atm. Pressure  (centred)
    fig5.add_subplot(gs[1, 3:5]),   # Solar Radiation (centred)
]

WINDROSE_VMAX = 4.0   # m/s colour-scale ceiling

for ax, (col, label, unit, lc, cmap5) in zip(axes5, VARS_FIG5):

    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ctx.add_basemap(ax, crs="EPSG:4326",
                    source=ctx.providers.CartoDB.Positron,
                    alpha=0.40, zorder=0, attribution=False)
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
        cb = fig5.colorbar(sm, ax=ax, fraction=0.033, pad=0.02, aspect=22)
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
        ax.annotate("WS17*", xy=(lons[i], lats[i]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=6, color="red", fontfamily=FONT)

    cb = fig5.colorbar(sc, ax=ax, fraction=0.033, pad=0.02, aspect=22)
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

fig5.text(0.01, 0.003,
          "* WS17 excluded from Atm. Pressure panel (faulty sensor, mean = 812 hPa)."
          "  Circle size = per-variable data completeness.  Wind direction encoded in rose bar length.",
          fontsize=7.5, fontfamily=FONT, fontstyle="italic", color="#555555")

fig5.suptitle(
    "Spatial Variability of Annual Mean Meteorological Variables\n"
    "NUS Campus Observation Network (2025)  |  40 stations  |  Circle size = data completeness",
    fontsize=13, fontfamily=FONT, fontweight="bold", y=1.01
)
out5 = os.path.join(ROOT, "figures", "Fig9_SpatialVariability.png")
fig5.savefig(out5, dpi=300, bbox_inches="tight")
print(f"Saved: {out5}")
plt.close(fig5)

print("\nAll 3 figures complete.")
