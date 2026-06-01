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
VAR_COLS_WS38 = {k: v for k, v in VAR_COLS.items() if k != "GlobalRad Ave (W/m2)"}
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
    if ws == "WS38":
        valid_cols = [c for c in VAR_COLS_WS38 if c in df.columns]
    # Mean of per-variable completeness (same metric as Fig 2). Gives credit for
    # working sensors while still penalising partial-sensor failures 
    # (excl. WS38 GlobalRad, which has no pyranometer).
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
df_stationmeta = pd.DataFrame(station_meta)

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
# Fig 4a — Bubble Map
# ═════════════════════════════════════════════════════════════════════════════
fig4a, ax1 = plt.subplots(figsize=(11, 9))
fig4a.patch.set_facecolor("white")

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
cbar = fig4a.colorbar(sc, ax=ax1, fraction=0.028, pad=0.02, aspect=30)
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

fig4a.tight_layout()
out4a = os.path.join(ROOT, "figures", "additional_figures", "AddFig4a_BubbleMap_Completeness.png")
fig4a.savefig(out4a, dpi=300, bbox_inches="tight")
print(f"Saved: {out4a}")
plt.close(fig4a)

# ═════════════════════════════════════════════════════════════════════════════
# Fig 4b — Heatmap
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

fig4b, ax2 = plt.subplots(figsize=(14, 12))
fig4b.patch.set_facecolor("white")

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

cbar2 = fig4b.colorbar(im, ax=ax2, fraction=0.022, pad=0.015, aspect=35)
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

fig4b.tight_layout()
out4b = os.path.join(ROOT, "figures", "additional_figures", "AddFig4b_Heatmap_MonthlyCompleteness.png")
fig4b.savefig(out4b, dpi=300, bbox_inches="tight")
print(f"Saved: {out4b}")
plt.close("all")


###### ═════════════════════════════════════════════════════════════════════════════
# Fig 4 — Heatmap (Condensed)
# ═════════════════════════════════════════════════════════════════════════════
NO_DATA_THRESH = 15

matrix = pd.DataFrame(
    {ws: [monthly_data[ws][m] for m in range(1, 13)] for ws in stations},
    index=[calendar.month_abbr[m] for m in range(1, 13)]
)

row_order = stations  # natural WS01 -> WS40 order for cross-referencing with other figures

mako_nd = copy.copy(MAKO)
mako_nd.set_bad("#d4d4d4")
masked_matrix = np.ma.masked_where(matrix.values < NO_DATA_THRESH, matrix.values)

fig4, fig4_ax = plt.subplots(2,1, figsize=(15, 5.5), gridspec_kw={'height_ratios': [1, 12]}, sharex=True, layout='constrained')
ax3 = fig4_ax[0]
ax4 = fig4_ax[1]
fig4.patch.set_facecolor("white")

im_overall = ax3.imshow(
    df_stationmeta.drop(index=['lat', 'lon']), aspect="auto",
    cmap=mako_nd, vmin=30, vmax=100,
    interpolation="nearest", alpha=0.92,
)

ax3.set_yticks(range(1))
ax3.set_yticklabels(["Overall"], fontsize=12, fontfamily=FONT)
ax3.set_xticks(np.arange(-0.5, len(row_order), 1), minor=True)
ax3.xaxis.set_ticks_position('none') 
ax3.grid(which="minor", color="white", linewidth=1.0)
ax3.tick_params(which="minor", length=0)
for spine in ax3.spines.values():
    spine.set_visible(False)

for j in range(len(row_order)):
    v = df_stationmeta.drop(index=['lat', 'lon']).values[0,j]
    txt_color = "white" if v < 70 else "#1a1a1a"
    ax3.text(j, 0, f"{v:.0f}", ha="center", va="center",
             fontsize=10, fontweight="bold",
             color=txt_color, fontfamily=FONT)

im = ax4.imshow(
    masked_matrix, aspect="auto",
    # cmap=newcmp, vmin=-0.5, vmax=8.5,
    cmap=mako_nd, vmin=30, vmax=100,
    interpolation="nearest", alpha=0.92,
)

ax4.set_yticks(range(12))
ax4.set_yticklabels(matrix.index, fontsize=12, fontfamily=FONT)
ax4.set_xticks(range(len(row_order)))
xlabels = [x[2:] for x in row_order]
ax4.set_xticklabels(xlabels, fontsize=12, fontfamily=FONT)

ax4.set_yticks(np.arange(-0.5, 12, 1), minor=True)
ax4.set_xticks(np.arange(-0.5, len(row_order), 1), minor=True)
ax4.grid(which="minor", color="white", linewidth=1.0)
ax4.tick_params(which="minor", length=0)
for spine in ax4.spines.values():
    spine.set_linewidth(0.5)

cbar3 = fig4b.colorbar(im, fraction=0.022, pad=0.015, aspect=35)
cbar3.set_label("Mean per-variable completeness (%)", fontsize=12,
                fontfamily=FONT, labelpad=10)
cbar3.ax.tick_params(labelsize=10)
for spine in cbar3.ax.spines.values():
    spine.set_linewidth(0.5)

nd_patch2 = Patch(facecolor="#d4d4d4", edgecolor="#aaaaaa",
                 linewidth=0.5, label="N/A (<15%)")
fig4.legend(handles=[nd_patch2], loc="lower right", bbox_to_anchor=(1.025, 0.03), fontsize=8,
           framealpha=0, edgecolor="#cccccc", fancybox=False,
           prop={"family": FONT})

ax4.set_ylabel("Month", fontsize=12, fontfamily=FONT, labelpad=8)
ax4.set_xlabel("Weather Station", fontsize=12, fontfamily=FONT, labelpad=8)
fig4.suptitle(
    "Monthly Mean Per-Variable Completeness —\n"
    "NUS Campus Meteorological Observation Network (2025)",
    fontsize=12.5, fontfamily=FONT, fontweight="bold", ha="left", x=0.05
)

out4 = os.path.join(ROOT, "figures", "Fig4_Heatmap_Completeness.png")
fig4.savefig(out4, dpi=300, bbox_inches="tight")
print(f"Saved: {out4}")
plt.close("all")

print("\nAll figures complete.")
