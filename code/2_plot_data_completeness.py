import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns
import copy
import contextily as ctx
import glob
import os
import calendar

# ── Typography & global style ─────────────────────────────────────────────────
FONT = "Palatino Linotype"
plt.rcParams.update({
    "font.family":        FONT,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    "axes.linewidth":     0.7,
    "xtick.major.width":  0.7,
    "ytick.major.width":  0.7,
    "figure.dpi":         150,
    "savefig.facecolor":  "white",
})

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MAKO = sns.color_palette("mako", as_cmap=True)
VMIN, VMAX = 65, 100

EXPECTED_HOURS = 8760
VAR_COLS = {
    "WindSpeed Ave (m/s)":   "Wind Speed",
    "WindDir Ave (degrees)": "Wind Dir",
    "AirTemp Ave (C)":       "Air Temp",
    "RelHum Ave (%)":        "Rel Hum",
    "AtmPress Ave (hPa)":    "Atm Press",
    "GlobalRad Ave (W/m2)":  "Solar Rad",
}
FILES = sorted(glob.glob(os.path.join(ROOT, "data", "raw", "NUS_CAMPUS_WS*.csv")))

# ── Load data ─────────────────────────────────────────────────────────────────
station_meta = {}
monthly_data = {}

for fpath in FILES:
    ws = os.path.basename(fpath).split("_")[2]
    df = pd.read_csv(fpath, parse_dates=["Datetime"])
    lat = df["Latitude"].iloc[0]
    lon = df["Longitude"].iloc[0]
    valid_cols = [c for c in VAR_COLS if c in df.columns]
    # Mean of per-variable completeness (same metric as Fig 2). Gives credit for
    # working sensors while still penalising partial-sensor failures (e.g. WS38
    # pyranometer, which stays dead all year while the other five vars work).
    per_var_pct = [df[c].notna().sum() / EXPECTED_HOURS * 100 for c in valid_cols]
    overall = sum(per_var_pct) / len(per_var_pct)
    station_meta[ws] = {"lat": lat, "lon": lon, "overall": overall}

    df["month"] = df["Datetime"].dt.month
    monthly = {}
    for m in range(1, 13):
        expected = calendar.monthrange(2025, m)[1] * 24
        sub = df[df["month"] == m]
        # Mean of per-variable completeness: gives credit for working sensors
        # while still penalising partial-sensor failures (e.g. WS38 pyranometer).
        per_var_pct = [sub[c].notna().sum() / expected * 100 for c in valid_cols]
        pct = sum(per_var_pct) / len(per_var_pct)
        monthly[m] = round(min(pct, 100), 2)
    monthly_data[ws] = monthly

stations = sorted(station_meta.keys())

# ── Force-based spread to nudge overlapping bubbles apart ────────────────────
def spread_positions(orig_lons, orig_lats,
                     min_dist=0.0010, max_disp=0.0009, iterations=300):
    # Visual only — does not alter coordinates in the data.
    lons = np.array(orig_lons, dtype=float)
    lats = np.array(orig_lats, dtype=float)
    olons = np.array(orig_lons, dtype=float)
    olats = np.array(orig_lats, dtype=float)
    n = len(lons)

    for _ in range(iterations):
        for i in range(n):
            for j in range(i + 1, n):
                dlon = lons[j] - lons[i]
                dlat = lats[j] - lats[i]
                dist = np.sqrt(dlon**2 + dlat**2)
                if 0 < dist < min_dist:
                    push = (min_dist - dist) / dist * 0.4
                    lons[i] -= dlon * push / 2
                    lats[i] -= dlat * push / 2
                    lons[j] += dlon * push / 2
                    lats[j] += dlat * push / 2

        # Clamp each station to max_disp from its original position
        for i in range(n):
            disp = np.sqrt((lons[i] - olons[i])**2 + (lats[i] - olats[i])**2)
            if disp > max_disp:
                angle = np.arctan2(lats[i] - olats[i], lons[i] - olons[i])
                lons[i] = olons[i] + max_disp * np.cos(angle)
                lats[i] = olats[i] + max_disp * np.sin(angle)

    return lons, lats

orig_lons = [station_meta[ws]["lon"] for ws in stations]
orig_lats = [station_meta[ws]["lat"] for ws in stations]
vals      = [station_meta[ws]["overall"] for ws in stations]
sizes     = [max(((v / 100) ** 2.2) * 700 + 65, 220) for v in vals]

plot_lons, plot_lats = spread_positions(orig_lons, orig_lats)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 1 — Bubble Map
# ═════════════════════════════════════════════════════════════════════════════
fig1, ax1 = plt.subplots(figsize=(11, 9))
fig1.patch.set_facecolor("white")

pad_lon, pad_lat = 0.004, 0.004
ax1.set_xlim(min(orig_lons) - pad_lon, max(orig_lons) + pad_lon)
ax1.set_ylim(min(orig_lats) - pad_lat, max(orig_lats) + pad_lat)

ctx.add_basemap(ax1, crs="EPSG:4326",
                source=ctx.providers.CartoDB.Positron,
                alpha=0.45, zorder=0, attribution=False)

sc = ax1.scatter(
    plot_lons, plot_lats, s=sizes, c=vals,
    cmap=MAKO, vmin=VMIN, vmax=VMAX,
    edgecolors="white", linewidths=0.9,
    zorder=3, alpha=0.88
)

# #ID + completeness % inside each bubble — adaptive text colour
for lon, lat, ws, val in zip(plot_lons, plot_lats, stations, vals):
    txt_col = "white" if val < 88 else "#1a1a1a"
    ax1.text(
        lon, lat, f"#{ws.replace('WS','')}\n{val:.0f}%",
        ha="center", va="center", zorder=5,
        fontsize=5.2, fontweight="bold", linespacing=1.35,
        color=txt_col, fontfamily=FONT,
    )

# Colorbar
cbar = fig1.colorbar(sc, ax=ax1, fraction=0.028, pad=0.02, aspect=30)
cbar.set_label("Mean per-variable completeness (%)", fontsize=10,
               fontfamily=FONT, labelpad=10)
cbar.ax.tick_params(labelsize=8.5)
for spine in cbar.ax.spines.values():
    spine.set_linewidth(0.5)

# Size legend — upper left
leg_handles = [
    ax1.scatter([], [], s=((p/100)**2.2)*700+65, c="#888888",
                alpha=0.7, edgecolors="white", linewidths=0.8)
    for p in [75, 88, 100]
]
leg = ax1.legend(
    leg_handles, ["75%", "88%", "100%"],
    title="Completeness", title_fontsize=9,
    fontsize=8.5, loc="upper left", framealpha=0.93,
    edgecolor="#cccccc", fancybox=False,
    labelspacing=1.8, handletextpad=1.4, borderpad=1.1,
)
leg.get_title().set_fontfamily(FONT)
for txt in leg.get_texts():
    txt.set_fontfamily(FONT)

ax1.xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
ax1.yaxis.set_major_formatter(FormatStrFormatter("%.4f"))
ax1.set_xlabel("Longitude (°E)", fontsize=10.5, fontfamily=FONT, labelpad=8)
ax1.set_ylabel("Latitude (°N)",  fontsize=10.5, fontfamily=FONT, labelpad=8)
ax1.tick_params(labelsize=8.5)
ax1.set_title(
    "Spatial Distribution and Data Completeness of the NUS Campus\n"
    "Meteorological Observation Network (2025)",
    fontsize=12.5, fontfamily=FONT, fontweight="bold", pad=16, loc="left"
)

fig1.tight_layout()
out1 = os.path.join(ROOT, "figures", "Fig4_BubbleMap_Completeness.png")
fig1.savefig(out1, dpi=300, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close(fig1)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 2 — Heatmap
# ═════════════════════════════════════════════════════════════════════════════
NO_DATA_THRESH = 5

matrix = pd.DataFrame(
    {ws: [monthly_data[ws][m] for m in range(1, 13)] for ws in stations},
    index=[calendar.month_abbr[m] for m in range(1, 13)]
).T

row_order = stations  # natural WS01 -> WS40 order for cross-referencing with other figures
matrix = matrix.loc[row_order]

mako_nd = copy.copy(MAKO)
mako_nd.set_bad("#d4d4d4")
masked_matrix = np.ma.masked_where(matrix.values < NO_DATA_THRESH, matrix.values)

fig2, ax2 = plt.subplots(figsize=(14, 12))
fig2.patch.set_facecolor("white")

im = ax2.imshow(
    masked_matrix, aspect="auto",
    cmap=mako_nd, vmin=35, vmax=100,
    interpolation="nearest", alpha=0.92
)

ax2.set_xticks(range(12))
ax2.set_xticklabels(matrix.columns, fontsize=9.5, fontfamily=FONT)
ax2.set_yticks(range(len(row_order)))
ax2.set_yticklabels(row_order, fontsize=8.5, fontfamily=FONT)

for i in range(len(row_order)):
    for j in range(12):
        v = matrix.values[i, j]
        if v < NO_DATA_THRESH:
            ax2.text(j, i, "N/A", ha="center", va="center",
                     fontsize=5.8, fontweight="bold",
                     color="#555555", fontfamily=FONT)
        elif v < 95:
            txt_color = "white" if v < 75 else "#1a1a1a"
            ax2.text(j, i, f"{v:.0f}%", ha="center", va="center",
                     fontsize=6.0, fontweight="bold",
                     color=txt_color, fontfamily=FONT)

ax2.set_xticks(np.arange(-0.5, 12, 1), minor=True)
ax2.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
ax2.grid(which="minor", color="white", linewidth=1.0)
ax2.tick_params(which="minor", length=0)
for spine in ax2.spines.values():
    spine.set_linewidth(0.5)

cbar2 = fig2.colorbar(im, ax=ax2, fraction=0.022, pad=0.015, aspect=35)
cbar2.set_label("Mean per-variable completeness (%)", fontsize=10,
                fontfamily=FONT, labelpad=10)
cbar2.ax.tick_params(labelsize=8.5)
for spine in cbar2.ax.spines.values():
    spine.set_linewidth(0.5)

nd_patch = Patch(facecolor="#d4d4d4", edgecolor="#aaaaaa",
                 linewidth=0.5, label="No data (<5%)")
ax2.legend(handles=[nd_patch], loc="lower right", fontsize=8.5,
           framealpha=0.93, edgecolor="#cccccc", fancybox=False,
           prop={"family": FONT})

ax2.set_xlabel("Month", fontsize=10.5, fontfamily=FONT, labelpad=10)
ax2.set_ylabel("Weather Station", fontsize=10.5, fontfamily=FONT, labelpad=10)
ax2.set_title(
    "Monthly Mean Per-Variable Completeness —\n"
    "NUS Campus Meteorological Observation Network (2025)",
    fontsize=12.5, fontfamily=FONT, fontweight="bold", pad=16, loc="left"
)

fig2.tight_layout()
out2 = os.path.join(ROOT, "figures", "Fig5_Heatmap_MonthlyCompleteness.png")
fig2.savefig(out2, dpi=300, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close("all")
print("\nBoth figures complete.")
