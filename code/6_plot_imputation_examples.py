import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

ROOT        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(ROOT, "data", "raw")
IMPUTED_DIR = os.path.join(ROOT, "data", "imputed")

VAR_COLOR = {
    "AirTemp Ave (C)":      "#bd0026",
    "RelHum Ave (%)":       "#225ea8",
    "WindSpeed Ave (m/s)":  "#006d2c",
    "AtmPress Ave (hPa)":   "#023858",
    "GlobalRad Ave (W/m2)": "#e05c00",
}
VAR_LABEL = {
    "AirTemp Ave (C)":      "Air Temperature (°C)",
    "RelHum Ave (%)":       "Relative Humidity (%)",
    "WindSpeed Ave (m/s)":  "Wind Speed (m s\u207b\u00b9)",
    "AtmPress Ave (hPa)":   "Atmospheric Pressure (hPa)",
    "GlobalRad Ave (W/m2)": "Global Radiation (W m\u207b\u00b2)",
}
FLAG_COL = {v: v.split()[0] + "_flag" for v in VAR_COLOR}

# Three panels: (station, variable, context_h, max_display_h)
# WS40 AirTemp  : 141h gap  — medium gap, best-performing variable
# WS02 AtmPress : 401h gap  — longer gap, showcases anomaly-corrected pressure fill
# WS34 GlobalRad: 1646h gap — very long gap, diurnal cycle preservation (first 720h shown)
EXAMPLES = [
    ("WS40", "AirTemp Ave (C)",      96,  500),
    ("WS02", "AtmPress Ave (hPa)",  120,  720),
    ("WS34", "GlobalRad Ave (W/m2)", 96,  720),
]


def load_pair(ws, var):
    orig = pd.read_csv(
        os.path.join(DATA_DIR, f"NUS_CAMPUS_{ws}_2025_Hourly.csv"),
        parse_dates=["Datetime"],
    ).set_index("Datetime")[var]
    imp = pd.read_csv(
        os.path.join(IMPUTED_DIR, f"NUS_CAMPUS_{ws}_2025_Hourly_imputed.csv"),
        parse_dates=["Datetime"],
    ).set_index("Datetime")
    return orig, imp[var], imp[FLAG_COL[var]]


def find_longest_s2_run(flags):
    s2 = (flags == 2).values
    runs = []
    start = None
    for i, v in enumerate(s2):
        if v and start is None:
            start = i
        elif not v and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(s2) - 1))
    if not runs:
        return None, None
    return max(runs, key=lambda r: r[1] - r[0])


COL_FILL = "#ebebeb"    # light grey — gap region shading

fig, axes = plt.subplots(3, 1, figsize=(10, 9))
fig.subplots_adjust(hspace=0.55)

panel_labels = ["a", "b", "c"]

for ax, (ws, var, ctx_h, max_disp_h), panel in zip(axes, EXAMPLES, panel_labels):
    orig, imp_vals, flags = load_pair(ws, var)
    col_obs = VAR_COLOR[var]

    run_start, run_end = find_longest_s2_run(flags)
    if run_start is None:
        ax.set_visible(False)
        continue

    gap_len  = run_end - run_start + 1
    disp_end = run_start + min(gap_len, max_disp_h)

    win_s = max(0, run_start - ctx_h)
    win_e = min(len(flags) - 1, disp_end + ctx_h)

    idx   = flags.index
    t     = idx[win_s : win_e + 1]
    obs_w = orig.iloc[win_s : win_e + 1]
    imp_w = imp_vals.iloc[win_s : win_e + 1]
    fl_w  = flags.iloc[win_s : win_e + 1]

    gap_t0 = idx[run_start]
    gap_t1 = idx[min(run_end, win_e)]

    # Shaded gap region
    ax.axvspan(gap_t0, gap_t1, color=COL_FILL, zorder=0)

    # Stage 2 imputed — dashed, same variable colour as observed
    gap_mask = (t >= gap_t0) & (t <= gap_t1)
    ax.plot(t[gap_mask], imp_w[gap_mask], color=col_obs, lw=1.3,
            linestyle="--", alpha=0.85, zorder=2)

    # Observed — solid variable colour; NaN where not observed so line breaks at gap
    obs_plot = obs_w.copy().where(fl_w == 0)
    ax.plot(t, obs_plot, color=col_obs, lw=1.5, linestyle="-", zorder=3)

    ax.set_ylabel(VAR_LABEL[var], fontsize=8.5)
    ax.tick_params(axis="both", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(minticks=4, maxticks=8))
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")

    # Gap annotation
    mid_t  = gap_t0 + (gap_t1 - gap_t0) / 2
    y0, y1 = ax.get_ylim()
    y_ann  = y0 + 0.94 * (y1 - y0)
    trunc  = gap_len > max_disp_h
    ann    = (f"{gap_len} h gap  (first {max_disp_h} h shown)"
              if trunc else f"{gap_len} h gap")
    ax.text(mid_t, y_ann, ann, ha="center", va="top",
            fontsize=7.5, color="#555555", style="italic")

    ax.text(0.01, 0.97, f"({panel})  {ws}", transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")

# Shared legend
legend_elements = [
    Line2D([0], [0], color="#888888", lw=1.5, linestyle="-",
           label="Observed"),
    Line2D([0], [0], color="#888888", lw=1.3, linestyle="--",
           label="Imputed — XGBoost spatial regression"),
    mpatches.Patch(facecolor=COL_FILL, edgecolor="#aac4de",
                   label="Gap region"),
]
fig.legend(handles=legend_elements, loc="lower center", ncol=2,
           fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, -0.04))

fig.suptitle("Representative gap-fill examples", fontsize=11, y=1.01)

out_base = os.path.join(ROOT, "figures", "Fig10_Imputation_Examples")
plt.savefig(out_base + ".png", dpi=200, bbox_inches="tight")
plt.savefig(out_base + ".pdf", bbox_inches="tight")
print(f"Saved {out_base}.png / .pdf")
plt.close()
